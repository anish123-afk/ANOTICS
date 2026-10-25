"""
Thor & Loki Multi-Agent Contract Extraction System
===================================================
1. Common State: OdinState (query, retrieved_document, reExtract = False)
2. Agent Thor:
   - Local state: query, retrieved_documents, iteration=1, max_iteration=3
   - Converts query into embedding using local Ollama bge-m3 model.
   - Retrieves top chunk (k=1) from ChromaDB using cosine similarity.
   - Extracts clause_id, document_name, contract_id, page_number, chunk_id.
   - Updates OdinState and its local state.
3. Agent Loki:
   - Inspects retrieved document in OdinState (e.g., B2).
   - Checks if next document (e.g., B3, A2, etc.) is present in manifest.
   - If present, sets OdinState.reExtract = True.
   - Loop continues until iteration == max_iteration.
   - Iteration increases by 1 on every loop step.
4. Final Output:
   - JSON containing: clause_id, document_name, contract_id, page_number, chunk_id.
"""

import os
import sys
import json
import re
import argparse
from typing import Optional, Dict, Any, List, Tuple
import urllib.request
import urllib.error

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import chromadb
from pydantic import BaseModel, Field

from odin_state import OdinState


# ----------------------------------------------------------------------
# Loki Prompt Placeholder (Insert your custom prompt here)
# ----------------------------------------------------------------------
LOKI_PROMPT = """You are LOKI, the Adversarial Review and Legal Attack Agent for Project Odin (Multi-Agent Contract Intelligence System).

Your role is to embody the counterparty's most aggressive, cut-throat legal counsel. You do not review contracts to see if things are going well; you review contracts to find every conceivable legal vulnerability, hidden trapdoor, cross-contract contradiction, and unrecoverable liability that could be weaponized against Quillhub.

================================================================================
CORE MANDATE & RESPONSIBILITIES (FR4)
================================================================================

1. ADVERSARIAL CONTRACT REVIEW (FR4.1)
   - Read contracts from the perspective of an opposing litigator looking to sue Quillhub, extract maximum damages, or escape unfavorable terms.
   - Actively identify:
     * One-sided terms: Asymmetrical termination rights, unilateral fee changes, or uncapped indemnification obligations.
     * Vague or weaponizable standards: Ambiguous definitions of "commercially reasonable efforts," "material breach," or subjective satisfaction criteria.
     * Silent defaults: Inaction traps, such as auto-renewal windows that renew automatically unless Quillhub serves notice under strict formalities.
     * Severe penalty triggers: Liquidated damages, service credits, penalty interest compounding, and forfeiture of intellectual property.

2. CROSS-CONTRACT CONFLICT DETECTION (FR4.2)
   - Analyze relationships between upstream providers (e.g., StratoNode A1/A2/A3) and downstream customers (e.g., Harbor Retail B1/B2, Logix C1, Contractors D1).
   - Uncover catastrophic liability gaps:
     * Uptime / SLA Mismatch: Quillhub promising 99.99% availability to Customer B1 while its cloud host A1 only guarantees 99.5% uptime.
     * Limitation of Liability Contradictions: Quillhub accepting uncapped data breach liability downstream while its upstream host's liability is capped at $5,000.
     * Territorial / Exclusivity Clashes: Granting exclusive distribution rights to Customer B1 that conflict with vendor terms in Agreement C1.
     * Amendment Precedence Traps: Determining whether an amendment (e.g. A3) quietly stripped away protections granted in master agreement A1 or SLA A2.

3. ATTACK PATH MODELING & RISK SCORING (FR4.3)
   - For every vulnerability you uncover, construct a reasoned "Attack Path":
     * Step-by-step narrative of how opposing counsel will frame the claim in litigation or arbitration.
     * The exact financial or operational damages Quillhub faces.
   - Assign two independent scores to each finding:
     * Severity: LOW | MEDIUM | HIGH | CRITICAL (potential damage scale).
     * Likelihood: LOW | MEDIUM | HIGH (probability of being triggered/enforced).

================================================================================
STRICT OPERATIONAL RULES & EVIDENCE STANDARDS
================================================================================

1. EVIDENCE ANCHORING (ZERO HALLUCINATION):
   - Every attack, trapdoor, and conflict you cite MUST reference exact Clause IDs (e.g., "B1-Sec-11.4", "A1-Sec-8.2") and quote the governing source text verbatim.
   - You are prohibited from inventing fictional breach clauses or fabricating indemnification obligations.

2. TIMELINE GROUNDING (RESPECT NORNS):
   - Ground your attacks in the current simulated checkpoint provided by Norns.
   - Do NOT declare an active breach on an obligation if Norns indicates the deadline is still upcoming. 
   - Attack upcoming deadlines by modeling the imminent disaster if Quillhub fails to act before the window shuts.

3. DISPUTE READINESS AGAINST HEIMDALL:
   - Formulate your risk claims with precise legal citations so they can stand up to scrutiny when Heimdall (Event Monitoring) attempts to defend Quillhub with operational event logs.

================================================================================
OUTPUT FORMAT REQUIREMENT
================================================================================
You must always output your findings as valid, parseable JSON conforming strictly to the requested schema. Never output conversational prose outside the JSON structure."""


# ----------------------------------------------------------------------
# Atreus Contract ID Lookup Helper
# ----------------------------------------------------------------------
def get_contract_id_from_atreus_csv(doc_name: str, csv_path: str = "contract_details_summary.csv") -> str:
    """
    Matches the document name (e.g. A1, B2) against the Atreus CSV
    and returns the corresponding formal legal contract_id (e.g. QH-STRATO-2026-001, QH-HARBOR-2026-DPA).
    """
    clean_doc = doc_name.strip().upper()

    # Check primary CSV and fallback v2 CSV
    candidate_paths = [csv_path, "contract_details_summary_v2.csv", "contract_details_summary.csv"]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                import csv
                with open(path, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        csv_doc = row.get("document_name", "").strip().upper()
                        cid = row.get("contract_or_document_id", "").strip()
                        if csv_doc == clean_doc and cid:
                            return cid
            except Exception:
                pass

    # Fallback to standard canonical portfolio mapping if CSV not yet created
    canonical_mapping = {
        "STD": "STD",
        "A1": "QH-STRATO-2026-001",
        "A2": "QH-STRATO-2026-SLA",
        "A3": "QH-STRATO-2026-AMD1",
        "B1": "QH-HARBOR-2026-001",
        "B2": "QH-HARBOR-2026-DPA",
        "B3": "QH-HARBOR-2026-AMD1",
        "C1": "QH-VANTAGE-2026-001",
        "D1": "QH-PIXEL-2026-001",
    }
    return canonical_mapping.get(clean_doc, f"QH-{clean_doc}-2026-001")


# ----------------------------------------------------------------------
# Thor's Local State
# ----------------------------------------------------------------------
class ThorLocalState(BaseModel):
    """
    Thor's local agent state.
    """
    query: str = Field(default="", description="The active query being processed")
    retrieved_documents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of all chunks retrieved across iterations"
    )
    iteration: int = Field(default=1, description="Current iteration number (starts at 1)")
    max_iteration: int = Field(default=3, description="Maximum iterations permitted (default: 3)")


# ----------------------------------------------------------------------
# Ollama Embedding Helper (bge-m3)
# ----------------------------------------------------------------------
def embed_query_with_ollama(
    text: str,
    base_url: str = "http://localhost:11434",
    model: str = "bge-m3",
    timeout: int = 30
) -> Optional[List[float]]:
    """
    Generates embedding for a single query using local Ollama bge-m3.
    """
    url = f"{base_url.rstrip('/')}/api/embed"
    payload = json.dumps({"model": model, "input": [text]}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "embeddings" in data and len(data["embeddings"]) > 0:
                return data["embeddings"][0]
    except Exception:
        # Fallback to single /api/embeddings
        fallback_url = f"{base_url.rstrip('/')}/api/embeddings"
        fb_payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
        fb_req = urllib.request.Request(fallback_url, data=fb_payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(fb_req, timeout=timeout) as fb_resp:
                fb_data = json.loads(fb_resp.read().decode("utf-8"))
                if "embedding" in fb_data:
                    return fb_data["embedding"]
        except Exception:
            return None

    return None


# ----------------------------------------------------------------------
# Agent Thor: Clause Extraction Agent
# ----------------------------------------------------------------------
class ThorAgent:
    """
    Agent Thor:
    1. Embeds query using local Ollama bge-m3.
    2. Searches ChromaDB vector store with cosine similarity (k=1).
    3. Extracts clause_id, document_name, contract_id, page_number, chunk_id.
    4. Updates local state and OdinState.
    """

    def __init__(
        self,
        db_dir: str = "chroma_db",
        collection_name: str = "contract_clauses",
        ollama_url: str = "http://localhost:11434",
        model: str = "bge-m3",
        max_iteration: int = 3
    ):
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.ollama_url = ollama_url
        self.model = model
        self.local_state = ThorLocalState(max_iteration=max_iteration)
        self.chroma_client: Optional[chromadb.PersistentClient] = None
        self.collection = None

        self._init_chroma()

    def _init_chroma(self):
        """Initializes connection to persistent Chroma store."""
        try:
            if os.path.exists(self.db_dir):
                self.chroma_client = chromadb.PersistentClient(path=self.db_dir)
                existing = [c.name for c in self.chroma_client.list_collections()]
                if self.collection_name in existing:
                    self.collection = self.chroma_client.get_collection(name=self.collection_name)
        except Exception as e:
            print(f"[Thor WARN] Could not connect to ChromaDB: {e}")

    def extract_clause(
        self,
        odin_state: OdinState,
        target_contract_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes a single extraction step (k=1) using bge-m3 embedding and cosine similarity.
        Updates OdinState and Thor's local state.
        """
        query = odin_state.query
        self.local_state.query = query

        print(f"\n[⚡ THOR] Iteration {self.local_state.iteration}/{self.local_state.max_iteration}")
        print(f"         Processing Query: '{query}'")
        if target_contract_id:
            print(f"         Targeting Document: [{target_contract_id}]")

        # 1. Convert query to embedding via Ollama bge-m3
        query_embedding = embed_query_with_ollama(
            query, base_url=self.ollama_url, model=self.model
        )

        extracted_doc: Optional[Dict[str, Any]] = None

        # 2. Search Chroma vector store (k=1)
        if self.collection is not None and query_embedding is not None:
            try:
                where_filter = {"contract_id": target_contract_id} if target_contract_id else None
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=1,
                    where=where_filter,
                    include=["metadatas", "documents", "distances"]
                )

                if results["metadatas"] and len(results["metadatas"][0]) > 0:
                    meta = results["metadatas"][0][0]
                    dist = results["distances"][0][0] if "distances" in results else 0.0
                    doc_code = target_contract_id or meta.get("contract_id", "UNKNOWN")
                    extracted_doc = {
                        "clause_id": meta.get("clause_id", "General"),
                        "document_name": doc_code,  # Document name like A1, A2, B2
                        "contract_id": doc_code,    # Temporary code, mapped to Atreus formal ID in output
                        "page_number": int(meta.get("page_number", 1)),
                        "chunk_id": meta.get("chunk_id", "chunk_001"),
                        "cosine_similarity": round(1.0 - dist, 4),
                        "snippet": results["documents"][0][0][:200]
                    }
            except Exception as e:
                print(f"[Thor WARN] Chroma query error: {e}")

        # 3. Fallback extraction (if Chroma/Ollama not running or empty)
        if extracted_doc is None:
            extracted_doc = self._heuristic_clause_lookup(query, target_contract_id)

        # 4. Update Thor's Local State
        self.local_state.retrieved_documents.append(extracted_doc)

        # 5. Update Common State (OdinState) on each iteration
        odin_state.update_retrieval(query=query, document=extracted_doc)
        odin_state.set_re_extract(False)  # Reset flag for Loki to inspect

        print(f"         -> Retrieved Document Name: [{extracted_doc['document_name']}]")
        print(f"            Series Root (First Doc): [{odin_state.first_document_name}]")
        print(f"            Clause ID:               {extracted_doc['clause_id']}")
        print(f"            Location:                Page {extracted_doc['page_number']} | Chunk {extracted_doc['chunk_id']}")

        return extracted_doc

    def _heuristic_clause_lookup(
        self,
        query: str,
        target_contract_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deterministic fallback retrieval if vector store is offline.
        Uses manifest and contract text files.
        """
        # Load manifest to find contracts
        cid = target_contract_id or "B2"
        doc_code = cid.upper()

        if os.path.exists("manifest.json"):
            try:
                with open("manifest.json", "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                contracts = manifest.get("contracts", [])
                for c in contracts:
                    if target_contract_id and c["contract_id"].upper() == target_contract_id.upper():
                        doc_code = c["contract_id"].upper()
                        break
                    elif not target_contract_id and ("b2" in query.lower() or "dpa" in query.lower() or "security" in query.lower()):
                        if c["contract_id"] == "B2":
                            doc_code = "B2"
                            break
            except Exception:
                pass

        clause = "Section 2.1" if "security" in query.lower() or "b2" in doc_code.lower() else "Section 1.1"
        return {
            "clause_id": clause,
            "document_name": doc_code,
            "contract_id": doc_code,
            "page_number": 2 if doc_code == "B2" else 1,
            "chunk_id": f"{doc_code}_p1_c002",
            "cosine_similarity": 0.9421,
            "snippet": f"Fallback retrieval excerpt for {doc_code} under query '{query}'."
        }

    def get_final_json(self) -> str:
        """
        Returns final JSON formatted strictly with required fields:
        clause_id, document_name, contract_id, page_number, chunk_id.
        Matches document_name against Atreus CSV to populate the formal legal contract_id.
        """
        if not self.local_state.retrieved_documents:
            return json.dumps({
                "clause_id": "N/A",
                "document_name": "N/A",
                "contract_id": "N/A",
                "page_number": 0,
                "chunk_id": "N/A"
            }, indent=2)

        last_doc = self.local_state.retrieved_documents[-1]
        doc_name = str(last_doc.get("document_name", "UNKNOWN")).upper()
        
        # Match document name (e.g. A1, B2) against Atreus CSV to get formal legal contract_id
        formal_contract_id = get_contract_id_from_atreus_csv(doc_name)

        output = {
            "clause_id": last_doc.get("clause_id"),
            "document_name": doc_name,
            "contract_id": formal_contract_id,
            "page_number": last_doc.get("page_number"),
            "chunk_id": last_doc.get("chunk_id")
        }
        return json.dumps(output, indent=2)


# ----------------------------------------------------------------------
# Agent Loki: Document Sequence & Amendment Verification Agent
# ----------------------------------------------------------------------
class LokiAgent:
    """
    Agent Loki:
    1. Inspects extracted document from OdinState (e.g., B2).
    2. Checks if next document (e.g., B3, A2, etc.) is present in manifest.json or catalog.
       If yes, makes reExtract in OdinState True.
    3. Interprets the final retrieved document using Claude Haiku and updates OdinState.loki_interpretation.
    """

    def __init__(
        self,
        manifest_path: str = "manifest.json",
        extra_catalog: Optional[List[str]] = None,
        model_name: str = "claude-3-5-haiku-20241022",
        api_key: Optional[str] = None,
        use_mock: bool = False
    ):
        self.manifest_path = manifest_path
        self.catalog = self._load_catalog(extra_catalog)
        self.model_name = model_name
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.use_mock = use_mock
        self.llm = None

        if not self.use_mock and self.api_key:
            try:
                from langchain_anthropic import ChatAnthropic
                self.llm = ChatAnthropic(
                    model_name=self.model_name,
                    anthropic_api_key=self.api_key,
                    temperature=0.0
                )
            except Exception as e:
                print(f"[Loki WARN] Could not initialize ChatAnthropic ({model_name}): {e}")
                self.use_mock = True
        else:
            self.use_mock = True

    def _load_catalog(self, extra_catalog: Optional[List[str]]) -> List[str]:
        """Loads available contract IDs from manifest.json."""
        ids: List[str] = []
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for c in data.get("contracts", []):
                    ids.append(c["contract_id"].upper())
            except Exception as e:
                print(f"[Loki WARN] Could not read manifest: {e}")

        if extra_catalog:
            for item in extra_catalog:
                ids.append(item.upper())
        return ids

    def compute_next_document_id(self, current_doc_id: str) -> Optional[str]:
        """
        Determines the next expected sequential document ID.
        Examples:
          'B2' -> 'B3'
          'A1' -> 'A2'
          'A2' -> 'A3'
          'B1' -> 'B2'
        """
        match = re.search(r"([A-Za-z]+)(\d+)", current_doc_id)
        if match:
            prefix = match.group(1).upper()
            num = int(match.group(2))
            next_id = f"{prefix}{num + 1}"
            return next_id
        return None

    def evaluate(self, odin_state: OdinState) -> Tuple[bool, Optional[str]]:
        """
        Evaluates OdinState.retrieved_document.
        Checks if the next document (e.g., B3) is present in the catalog.
        If yes, sets odin_state.reExtract = True.
        Returns: (reExtract_bool, next_doc_id)
        """
        if not odin_state.retrieved_document:
            odin_state.set_re_extract(False)
            return False, None

        current_id = str(odin_state.retrieved_document.get("document_name") or odin_state.retrieved_document.get("contract_id", "")).upper()
        next_id = self.compute_next_document_id(current_id)

        print(f"\n[🦊 LOKI] Evaluating retrieved document: [{current_id}]")
        print(f"         Checking for subsequent document: [{next_id}] in catalog...")

        if next_id and next_id in self.catalog:
            print(f"         -> Found [{next_id}] in manifest! Setting reExtract = True in OdinState.")
            odin_state.set_re_extract(True)
            return True, next_id
        else:
            print(f"         -> [{next_id}] is NOT present in manifest. reExtract remains False.")
            odin_state.set_re_extract(False)
            return False, None

    def interpret_final_document(self, odin_state: OdinState) -> str:
        """
        Gets the final retrieved document from the updated OdinState,
        invokes Claude Haiku using LOKI_PROMPT, and updates odin_state.loki_interpretation.
        """
        final_doc = odin_state.retrieved_document or {}
        doc_name = final_doc.get("document_name", "Unknown")
        contract_id = final_doc.get("contract_id", "")
        clause_id = final_doc.get("clause_id", "General")
        snippet = final_doc.get("snippet", "")
        query = odin_state.query

        print(f"\n[🦊 LOKI] Interpreting final retrieved document [{doc_name}] with Claude Haiku...")

        # Prompt placeholder designated by user
        system_prompt = LOKI_PROMPT

        if self.use_mock or self.llm is None:
            interpretation = (
                f"[Loki Interpretation ({self.model_name})]: Final document '{doc_name}' ({contract_id}) "
                f"analyzed under query '{query}'. Target clause is '{clause_id}'."
            )
        else:
            try:
                from langchain_core.prompts import ChatPromptTemplate
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "Final Retrieved Document Information:\n"
                              "- Document Name: {doc_name}\n"
                              "- Contract ID: {contract_id}\n"
                              "- Clause ID: {clause_id}\n"
                              "- Page: {page_number}\n"
                              "- Chunk ID: {chunk_id}\n"
                              "- Relevant Text Excerpt:\n```\n{snippet}\n```\n\nQuery: {query}")
                ])
                chain = prompt | self.llm
                resp = chain.invoke({
                    "doc_name": doc_name,
                    "contract_id": contract_id,
                    "clause_id": clause_id,
                    "page_number": final_doc.get("page_number", 1),
                    "chunk_id": final_doc.get("chunk_id", ""),
                    "snippet": snippet,
                    "query": query
                })
                interpretation = resp.content.strip()
            except Exception as e:
                interpretation = f"[Loki Interpretation Fallback]: Evaluated {doc_name} for '{query}'. (Error: {e})"

        # Update common state with the output of LLM
        odin_state.update_loki_interpretation(interpretation)
        print("         -> Added 'loki_interpretation' to OdinState successfully!")
        return interpretation


# ----------------------------------------------------------------------
# Orchestration Pipeline (Thor <-> Loki Loop)
# ----------------------------------------------------------------------
def run_thor_loki_loop(
    initial_query: str,
    manifest_path: str = "manifest.json",
    db_dir: str = "chroma_db",
    max_iteration: int = 3,
    extra_catalog: Optional[List[str]] = None,
    model_name: str = "claude-3-5-haiku-20241022",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the collaborative loop between Thor and Loki:
    1. Thor extracts clause from vector store using bge-m3 + cosine similarity.
    2. Updates OdinState on each iteration.
    3. Loki checks if next document (e.g., B2 -> B3) exists.
    4. If yes, sets reExtract = True, and Thor runs again.
    5. Loop stops when iteration == max_iteration or reExtract is False.
    6. Loki interprets the final retrieved document using Claude Haiku (LOKI_PROMPT).
    7. Updates OdinState with loki_interpretation.
    8. Returns Thor's final JSON output.
    """
    print("=" * 75)
    print(" ODIN ORCHESTRATOR: THOR (EXTRACTOR) & LOKI (VERIFIER & INTERPRETER)")
    print("=" * 75)

    # 1. Initialize Common State (OdinState)
    odin_state = OdinState(query=initial_query, retrieved_document=None, reExtract=False)
    print(f"Initial State: {odin_state}\n")

    # 2. Initialize Agents
    thor = ThorAgent(db_dir=db_dir, max_iteration=max_iteration)
    loki = LokiAgent(manifest_path=manifest_path, extra_catalog=extra_catalog, model_name=model_name, api_key=api_key)

    target_next_doc = None

    # 3. Collaborative loop
    while thor.local_state.iteration <= thor.local_state.max_iteration:
        # Thor extracts
        thor.extract_clause(odin_state, target_contract_id=target_next_doc)
        print(f"[State Update] {odin_state}")

        # Loki evaluates retrieved document
        should_reextract, next_doc = loki.evaluate(odin_state)
        print(f"[State Update] {odin_state}")

        # Check termination condition
        if not should_reextract:
            print(f"\n[ORCHESTRATOR] Loki reported no subsequent document to extract. Exiting loop.")
            break

        if thor.local_state.iteration >= thor.local_state.max_iteration:
            print(f"\n[ORCHESTRATOR] Reached max_iteration ({thor.local_state.max_iteration}). Exiting loop.")
            break

        # Prepare for next iteration
        target_next_doc = next_doc
        thor.local_state.iteration += 1

    # 4. Loki interprets final retrieved document from OdinState
    loki.interpret_final_document(odin_state)
    print(f"\n[State Update with Loki Interpretation]\n{odin_state}")
    print(f"\n[🦊 LOKI INTERPRETATION OUTPUT]\n{odin_state.loki_interpretation}")

    # 5. Final Output from Thor
    final_json = thor.get_final_json()
    print("\n" + "=" * 75)
    print(" FINAL JSON OUTPUT OF THOR:")
    print("=" * 75)
    print(final_json)
    print("=" * 75)

    return json.loads(final_json)


# ----------------------------------------------------------------------
# CLI Interface
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Thor & Loki Multi-Agent Contract Extraction with OdinState."
    )
    parser.add_argument("query", nargs="?", default="What are the data security and encryption requirements?",
                        help="Query to extract from contract vector store")
    parser.add_argument("--manifest", default="manifest.json", help="Path to manifest.json")
    parser.add_argument("--db-dir", default="chroma_db", help="Path to ChromaDB directory")
    parser.add_argument("--max-iter", type=int, default=3, help="Max iterations for Thor (default: 3)")
    parser.add_argument("--start-doc", default=None,
                        help="Optional specific starting document ID (e.g., A1, B2) to initiate extraction")
    parser.add_argument("--simulate-b3", action="store_true",
                        help="Simulate B3 being present in catalog to test full B2 -> B3 reExtract loop")
    parser.add_argument("--model", default="claude-3-5-haiku-20241022", help="Claude Haiku model for Loki")
    parser.add_argument("--api-key", default=None, help="Anthropic API key")

    args = parser.parse_args()

    extra = ["B3"] if args.simulate_b3 else None
    
    # Run orchestrator
    print("=" * 75)
    print(" ODIN ORCHESTRATOR: THOR (EXTRACTOR) & LOKI (VERIFIER & INTERPRETER)")
    print("=" * 75)

    odin_state = OdinState(query=args.query, retrieved_document=None, reExtract=False)
    print(f"Initial State: {odin_state}\n")

    thor = ThorAgent(db_dir=args.db_dir, max_iteration=args.max_iter)
    loki = LokiAgent(manifest_path=args.manifest, extra_catalog=extra, model_name=args.model, api_key=args.api_key)

    target_next_doc = args.start_doc

    while thor.local_state.iteration <= thor.local_state.max_iteration:
        # Thor extracts
        thor.extract_clause(odin_state, target_contract_id=target_next_doc)
        print(f"[State Update] {odin_state}")

        # Loki evaluates retrieved document
        should_reextract, next_doc = loki.evaluate(odin_state)
        print(f"[State Update] {odin_state}")

        # Check termination condition
        if not should_reextract:
            print(f"\n[ORCHESTRATOR] Loki reported no subsequent document to extract. Exiting loop.")
            break

        if thor.local_state.iteration >= thor.local_state.max_iteration:
            print(f"\n[ORCHESTRATOR] Reached max_iteration ({thor.local_state.max_iteration}). Exiting loop.")
            break

        # Prepare for next iteration
        target_next_doc = next_doc
        thor.local_state.iteration += 1

    # Loki interprets final retrieved document
    loki.interpret_final_document(odin_state)
    print(f"\n[State Update with Loki Interpretation]\n{odin_state}")
    print(f"\n[🦊 LOKI INTERPRETATION OUTPUT]\n{odin_state.loki_interpretation}")

    final_json = thor.get_final_json()
    print("\n" + "=" * 75)
    print(" FINAL JSON OUTPUT OF THOR:")
    print("=" * 75)
    print(final_json)
    print("=" * 75)


if __name__ == "__main__":
    main()
