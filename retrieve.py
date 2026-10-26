"""
Contract Intelligence Retriever
===============================
Queries the persistent Chroma vector store without rebuilding or re-embedding.
Uses the same Ollama bge-m3 embedding function to perform semantic search across
all contracts or filtered by contract_id.
"""

import sys
import argparse
from typing import Optional, Dict, Any

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import chromadb
from langchain_ollama import OllamaEmbeddings
from build_vector_store import OllamaBgeM3Embedder

# Standard LangChain OllamaEmbeddings instance for bge-m3
embeddings = OllamaEmbeddings(
    model="bge-m3"
)


def retrieve_relevant_clauses(
    query_text: str,
    n_results: int = 5,
    contract_id: Optional[str] = None,
    clause_id: Optional[str] = None,
    db_dir: str = "chroma_db",
    collection_name: str = "contract_clauses",
    ollama_url: str = "http://localhost:11434",
    model: str = "bge-m3"
) -> Dict[str, Any]:
    """
    Connects to existing Chroma vector store and retrieves relevant contract chunks.
    """
    client = chromadb.PersistentClient(path=db_dir)
    embedder = OllamaBgeM3Embedder(base_url=ollama_url, model=model)

    try:
        collection = client.get_collection(name=collection_name, embedding_function=embedder)
    except Exception as e:
        raise RuntimeError(
            f"Could not load Chroma collection '{collection_name}' from '{db_dir}'. "
            f"Have you run 'python build_vector_store.py' first? Error: {e}"
        ) from e

    # Build metadata filter if specified
    where_filter = None
    filters = []
    if contract_id:
        filters.append({"contract_id": contract_id})
    if clause_id:
        filters.append({"clause_id": clause_id})

    if len(filters) == 1:
        where_filter = filters[0]
    elif len(filters) > 1:
        where_filter = {"$and": filters}

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )
    return results


def print_search_results(results: Dict[str, Any], query: str):
    """Prints retrieved results in clean human-readable format."""
    print("=" * 80)
    print(f" QUERY: \"{query}\"")
    print("=" * 80)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs:
        print("No matching contract clauses found.")
        return

    for idx, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        # Cosine distance: similarity = 1 - distance
        sim_score = max(0.0, 1.0 - dist)
        print(f"\n[{idx}] Document:  {meta.get('document_name')} (ID: {meta.get('contract_id')})")
        print(f"    Clause:    {meta.get('clause_id')}")
        print(f"    Location:  Page {meta.get('page_number')} | Chunk ID: {meta.get('chunk_id')}")
        print(f"    Score:     Similarity {sim_score:.4f} (Cosine distance: {dist:.4f})")
        print("    " + "-" * 76)
        # Indent document text preview
        preview_lines = doc.strip().split("\n")
        for line in preview_lines:
            print(f"    {line}")
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Query the persistent Chroma vector store of contracts using Ollama bge-m3 embeddings."
    )
    parser.add_argument("query", nargs="?", default="What are the uptime SLA targets and service credits?",
                        help="Query string to search for (default: SLA uptime & service credits)")
    parser.add_argument("--n-results", "-n", type=int, default=5, help="Number of results to retrieve (default: 5)")
    parser.add_argument("--contract-id", "-c", default=None, help="Optional filter by contract ID (e.g., A1, A2, STD)")
    parser.add_argument("--db-dir", default="chroma_db", help="Chroma database directory (default: chroma_db)")
    parser.add_argument("--collection", default="contract_clauses", help="Collection name (default: contract_clauses)")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--model", default="bge-m3", help="Ollama embedding model (default: bge-m3)")

    args = parser.parse_args()

    try:
        results = retrieve_relevant_clauses(
            query_text=args.query,
            n_results=args.n_results,
            contract_id=args.contract_id,
            db_dir=args.db_dir,
            collection_name=args.collection,
            ollama_url=args.ollama_url,
            model=args.model
        )
        print_search_results(results, args.query)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
