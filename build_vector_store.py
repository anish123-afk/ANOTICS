"""
One-Time Contract Ingestion and Vector Store Builder
===================================================
Pipeline:
1. Loads contract paths and metadata from manifest.json
2. Checks if Chroma persistent database already exists. If documents already exist,
   skips the embedding process (enforces single-time creation).
3. Uses PyMuPDF (pymupdf) to read each PDF page-by-page.
4. Uses RecursiveCharacterTextSplitter to split text into section & paragraph chunks.
5. Extracts and tracks clause IDs (e.g., Section 1.1, Article 4, §2, etc.).
6. Generates embeddings via local Ollama bge-m3 model.
7. Stores chunks and metadata (contract_id, document_name, clause_id, page_number, chunk_id)
   into a persistent ChromaDB collection.
"""

import os
import sys
import json
import re
import argparse
from typing import List, Dict, Any, Optional
import urllib.request
import urllib.error

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pymupdf
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ----------------------------------------------------------------------
# Ollama bge-m3 Embedding Function for ChromaDB
# ----------------------------------------------------------------------
class OllamaBgeM3Embedder(EmbeddingFunction[Documents]):
    """
    Embedding function that calls local Ollama server running bge-m3.
    Compatible with ChromaDB's EmbeddingFunction interface.
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "bge-m3", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        # 1. Try batch /api/embed endpoint (modern Ollama >= 0.1.33)
        embed_url = f"{self.base_url}/api/embed"
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            embed_url,
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if "embeddings" in data:
                    return data["embeddings"]
        except urllib.error.HTTPError as e:
            # Fall back to single /api/embeddings if /api/embed is not available (404)
            if e.code != 404:
                raise RuntimeError(
                    f"Ollama embedding API error (HTTP {e.code}): {e.read().decode('utf-8', errors='ignore')}"
                ) from e
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                f"Ensure Ollama is running ('ollama serve') and model '{self.model}' is installed ('ollama pull {self.model}'). "
                f"Original error: {e}"
            ) from e

        # 2. Fallback to /api/embeddings per item
        fallback_url = f"{self.base_url}/api/embeddings"
        embeddings: List[List[float]] = []
        for text in texts:
            p = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
            r = urllib.request.Request(fallback_url, data=p, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(r, timeout=self.timeout) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    embeddings.append(res["embedding"])
            except urllib.error.URLError as err:
                raise ConnectionError(
                    f"Could not connect to Ollama at {self.base_url}. Error: {err}"
                ) from err

        return embeddings

    def __call__(self, input: Documents) -> Embeddings:
        """Embed a list of documents (required by ChromaDB EmbeddingFunction)."""
        if not input:
            return []
        
        # Batch requests in chunks of 32 for optimal Ollama performance
        batch_size = 32
        all_embeddings: List[List[float]] = []
        for i in range(0, len(input), batch_size):
            batch = input[i : i + batch_size]
            emb_batch = self._embed_batch(batch)
            all_embeddings.extend(emb_batch)
        return all_embeddings


# ----------------------------------------------------------------------
# Clause Identification Helper
# ----------------------------------------------------------------------
CLAUSE_PATTERNS = [
    # Special Terms §1, §3.1
    (re.compile(r"Special\s+Terms\s+§\s*(\d+(?:\.\d+)*)", re.IGNORECASE), lambda m: f"Special Terms §{m.group(1)}"),
    # §3.1 or § 4
    (re.compile(r"§\s*(\d+(?:\.\d+)*)"), lambda m: f"§{m.group(1)}"),
    # SECTION 1 — SERVICE or Section 2.1 or SECTION A
    (re.compile(r"\bSECTION\s+([0-9]+(?:\.[0-9]+)*)\b", re.IGNORECASE), lambda m: f"Section {m.group(1)}"),
    (re.compile(r"\bSection\s+([A-Z])\b"), lambda m: f"Section {m.group(1)}"),
    # 1.1 Access and Use or 12.3 Governing Law
    (re.compile(r"(?:^|\n)\s*(\d+\.\d+(?:\.\d+)*)\s+[A-Z]"), lambda m: f"Section {m.group(1)}"),
    # Exhibit A, Schedule 1, Addendum B
    (re.compile(r"\b(Exhibit|Schedule|Addendum)\s+([0-9A-Z]+)\b", re.IGNORECASE), lambda m: f"{m.group(1).title()} {m.group(2).upper()}"),
]

def extract_clause_id(text: str, current_clause: Optional[str] = None) -> str:
    """
    Scans chunk text for clause or section references.
    If none found, carries over the current active clause or defaults to 'General'.
    """
    for pat, formatter in CLAUSE_PATTERNS:
        match = pat.search(text)
        if match:
            return formatter(match)

    return current_clause if current_clause else "General"


# ----------------------------------------------------------------------
# PDF Extraction & Chunking
# ----------------------------------------------------------------------
def process_pdf_contract(
    contract_info: Dict[str, Any],
    text_splitter: RecursiveCharacterTextSplitter,
    base_dir: str = "."
) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page with PyMuPDF, splits into section & paragraph chunks,
    and assigns all required metadata.
    """
    contract_id = contract_info["contract_id"]
    doc_name = contract_info.get("document_name", contract_id)
    pdf_rel_path = contract_info["pdf_path"]
    pdf_path = os.path.normpath(os.path.join(base_dir, pdf_rel_path))

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Contract PDF not found: {pdf_path}")

    print(f"[*] Reading: [{contract_id}] {doc_name} from {pdf_path}")
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    print(f"    Total pages: {total_pages}")

    all_chunks: List[Dict[str, Any]] = []
    current_clause_tracker = "General"
    chunk_counter = 0

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page = doc[page_idx]
        page_text = page.get_text("text").strip()

        if not page_text:
            continue

        # Split text on section/paragraph boundaries
        page_chunks = text_splitter.split_text(page_text)

        for chunk_text in page_chunks:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunk_counter += 1
            chunk_id = f"{contract_id}_p{page_num}_c{chunk_counter:03d}"

            # Detect clause reference or maintain active clause
            clause_id = extract_clause_id(chunk_text, current_clause_tracker)
            if clause_id != "General":
                current_clause_tracker = clause_id

            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "contract_id": str(contract_id),
                    "document_name": str(doc_name),
                    "clause_id": str(clause_id),
                    "page_number": int(page_num),
                    "chunk_id": str(chunk_id),
                }
            })

    print(f"   Generated {len(all_chunks)} chunks across {total_pages} pages.")
    return all_chunks


# ----------------------------------------------------------------------
# Vector Store Ingestion
# ----------------------------------------------------------------------
def build_vector_store(
    manifest_path: str = "manifest.json",
    db_dir: str = "chroma_db",
    collection_name: str = "contract_clauses",
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    ollama_url: str = "http://localhost:11434",
    model: str = "bge-m3",
    force_rebuild: bool = False
):
    """
    Loads manifest, checks if vector store exists (single-time creation),
    chunks PDFs, embeds with Ollama bge-m3, and persists in ChromaDB.
    """
    print("=" * 70)
    print(" QUILLHUB CONTRACT INTELLIGENCE — VECTOR STORE BUILDER")
    print("=" * 70)

    # 1. Initialize persistent Chroma client
    client = chromadb.PersistentClient(path=db_dir)
    embedder = OllamaBgeM3Embedder(base_url=ollama_url, model=model)

    # 2. Check if collection already exists
    existing_collections = [c.name for c in client.list_collections()]
    
    if collection_name in existing_collections:
        col = client.get_collection(name=collection_name, embedding_function=embedder)
        doc_count = col.count()
        if doc_count > 0 and not force_rebuild:
            print(f"\n[INFO] Vector store already exists at '{db_dir}' with {doc_count} chunks.")
            print("       Enforcing SINGLE-TIME creation: embedding step is SKIPPED.")
            print("       Ready for queries. Use retrieve.py or pass --force to rebuild.\n")
            return col

        # If empty (from an interrupted run) or force_rebuild is True, delete and recreate
        if doc_count == 0 or force_rebuild:
            if force_rebuild:
                print(f"\n[WARN] --force flag passed. Deleting existing collection '{collection_name}'...")
            else:
                print(f"\n[INFO] Found incomplete/empty collection '{collection_name}'. Resetting...")
            client.delete_collection(collection_name)

    # 3. Create collection with custom Ollama bge-m3 embedding function
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"\nCreated Chroma collection '{collection_name}' (distance: cosine).")

    # 4. Load Manifest
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    contracts = manifest_data.get("contracts", [])
    if not contracts:
        raise ValueError(f"No contracts found in '{manifest_path}'.")

    print(f"Loaded manifest with {len(contracts)} contracts to process.\n")

    # 5. Initialize RecursiveCharacterTextSplitter with section and paragraph delimiters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n================================================================================\n",
            "\n\n--------------------------------------------------------------------------------\n",
            "\n\nSECTION ",
            "\n\nSection ",
            "\n\n§",
            "\n\n\n",
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )

    # 6. Process and extract all chunks
    all_documents: List[str] = []
    all_metadatas: List[Dict[str, Any]] = []
    all_ids: List[str] = []

    base_dir = os.path.dirname(os.path.abspath(manifest_path))

    for contract in contracts:
        chunks = process_pdf_contract(contract, text_splitter, base_dir=base_dir)
        for c in chunks:
            all_documents.append(c["text"])
            all_metadatas.append(c["metadata"])
            all_ids.append(c["chunk_id"])

    total_chunks = len(all_documents)
    print(f"\nTotal chunks to embed and store: {total_chunks}")

    # 7. Add documents to Chroma collection in batches (handles Ollama embeddings automatically)
    batch_size = 64
    print(f"Connecting to Ollama at {ollama_url} (model: {model}) and storing embeddings in Chroma...")
    
    for i in range(0, total_chunks, batch_size):
        end_idx = min(i + batch_size, total_chunks)
        batch_docs = all_documents[i:end_idx]
        batch_metas = all_metadatas[i:end_idx]
        batch_ids = all_ids[i:end_idx]

        print(f"  -> Indexing chunks {i + 1} to {end_idx} of {total_chunks}...")
        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids
        )

    print("\n" + "=" * 70)
    print(f" SUCCESS: Vector store created with {collection.count()} chunks in '{db_dir}'.")
    print("=" * 70)
    return collection


# ----------------------------------------------------------------------
# CLI Entry Point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Ingest contract PDFs and build persistent Chroma vector store with Ollama bge-m3 embeddings."
    )
    parser.add_argument("--manifest", default="manifest.json", help="Path to manifest.json (default: manifest.json)")
    parser.add_argument("--db-dir", default="chroma_db", help="Directory for ChromaDB persistence (default: chroma_db)")
    parser.add_argument("--collection", default="contract_clauses", help="Collection name (default: contract_clauses)")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size in characters (default: 800)")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap in characters (default: 150)")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server URL (default: http://localhost:11434)")
    parser.add_argument("--model", default="bge-m3", help="Ollama embedding model (default: bge-m3)")
    parser.add_argument("--force", action="store_true", help="Force rebuild vector store even if it already exists")

    args = parser.parse_args()

    try:
        build_vector_store(
            manifest_path=args.manifest,
            db_dir=args.db_dir,
            collection_name=args.collection,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            ollama_url=args.ollama_url,
            model=args.model,
            force_rebuild=args.force
        )
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
