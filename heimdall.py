"""
Agent Heimdall — Event & Incident Monitoring Agent for Project Odin
===================================================================
Responsibilities:
1. Takes the event folder as main input (e.g. `bundle/events/` containing JSON/JSONL event logs).
2. Converts each event JSON into an embedding using the local Ollama `bge-m3` model.
3. Converts the `query` from the common state (`OdinState`) into an embedding using `bge-m3`.
4. Finds the event JSON with the maximum cosine similarity to the query.
5. Invokes Claude Haiku (`claude-3-5-haiku-20241022`) using the custom prompt placeholder (`HEIMDALL_PROMPT`).
6. Updates the common shared state `OdinState` by setting `heimdall_review` with the LLM output.
"""

import os
import sys
import json
import math
import glob
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional, Tuple

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from odin_state import OdinState

# Load environment variables (e.g. ANTHROPIC_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ----------------------------------------------------------------------
# Heimdall Prompt Placeholder (Insert your custom prompt here)
# ----------------------------------------------------------------------
HEIMDALL_PROMPT = """You are Heimdall, the All-Seeing Event Monitoring and Defense Intelligence Agent for Odin (Multi-Agent Contract Intelligence System).

Your primary mission is to protect Quillhub by continuously monitoring external real-world event feeds (server logs, incident tickets, payment records, email updates) and mapping them directly to contractual clauses. You act as the factual defense against aggressive risk claims made by Loki (the Adversarial Review Agent).

YOUR RESPONSIBILITIES (FR5):
1. EVENT-TO-CLAUSE MAPPING (FR5.1):
   - Ingest incoming event records and map each event to the specific contract (e.g. A1, A2, B1) and contractual provision affected.
   - Determine whether the event satisfies, cures, or violates a contractual trigger (e.g. uptime SLAs, maintenance windows, force majeure, notice receipts).

2. FACTUAL VERIFICATION & COUNTER-ANALYSIS (FR5.2):
   - Review adversarial risk claims made by Loki.
   - Check Loki’s allegations against real-world event logs:
     * If Loki claims an SLA breach due to downtime, check if event logs prove it was scheduled maintenance or caused by an upstream outage.
     * If Loki claims a late payment default, verify if payment confirmation events or cure grace periods apply.
   - Defend Quillhub’s legal standing whenever real-world evidence justifies it, but acknowledge legitimate breaches when evidence is undeniable.

3. NOISE-FILTERED ALERT GENERATION (FR5.3):
   - Suppress routine operational noise (duplicate tickets, transient warnings, resolved pings).
   - Only escalate high-confidence, severity-rated alerts for genuine impending breaches, unfulfilled notice windows, or payment failures.

CRITICAL OPERATIONAL RULES:
- CITATION MANDATE: Every claim, defense, or alert you raise must cite the exact Event ID (e.g. 'EVT-2027-0402') and Contract Clause ID (e.g. 'A2-Sec-3.1').
- NO GUESSWORK: If an event record lacks sufficient details to confirm or refute a breach, explicitly flag 'INSUFFICIENT_EVENT_DATA'.
- DISPUTE RESOLUTION PREPARATION: Clearly structure your findings so Tyr (Arbitration Agent) can easily compare your evidence against Loki's claims."""


# ----------------------------------------------------------------------
# Math Helper: Cosine Similarity
# ----------------------------------------------------------------------
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Computes cosine similarity between two float vectors.
    Returns a score between -1.0 and 1.0.
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ----------------------------------------------------------------------
# Ollama Embedding Client (bge-m3)
# ----------------------------------------------------------------------
def embed_texts_with_ollama(
    texts: List[str],
    base_url: str = "http://localhost:11434",
    model: str = "bge-m3",
    timeout: int = 60
) -> List[List[float]]:
    """
    Generates embeddings for a list of text strings using local Ollama bge-m3.
    Supports both batch /api/embed and fallback /api/embeddings.
    Includes a deterministic fallback if Ollama service is offline/unreachable.
    """
    if not texts:
        return []

    url = f"{base_url.rstrip('/')}/api/embed"
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "embeddings" in data and len(data["embeddings"]) == len(texts):
                return data["embeddings"]
    except Exception as e:
        # Try individual requests to /api/embeddings if batch /api/embed fails
        embeddings: List[List[float]] = []
        fallback_url = f"{base_url.rstrip('/')}/api/embeddings"
        success = True

        for text in texts:
            fb_payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
            fb_req = urllib.request.Request(fallback_url, data=fb_payload, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(fb_req, timeout=timeout) as fb_resp:
                    fb_data = json.loads(fb_resp.read().decode("utf-8"))
                    if "embedding" in fb_data:
                        embeddings.append(fb_data["embedding"])
                    else:
                        success = False
                        break
            except Exception:
                success = False
                break

        if success and len(embeddings) == len(texts):
            return embeddings

        print(f"[Heimdall WARN] Ollama ({model} at {base_url}) unreachable ({e}). Using deterministic embedding fallback.")

    # Deterministic term-frequency fallback vector (1024-dim) for offline/restricted environments
    fallback_dim = 1024
    results: List[List[float]] = []
    for text in texts:
        vec = [0.0] * fallback_dim
        words = text.lower().split()
        for i, word in enumerate(words):
            h = abs(hash(word)) % fallback_dim
            vec[h] += 1.0 / (1.0 + i * 0.05)
        # Normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        results.append(vec)
    return results


def embed_single_text_with_ollama(
    text: str,
    base_url: str = "http://localhost:11434",
    model: str = "bge-m3",
    timeout: int = 30
) -> List[float]:
    """Helper to embed a single text string."""
    embeddings = embed_texts_with_ollama([text], base_url=base_url, model=model, timeout=timeout)
    return embeddings[0] if embeddings else [0.0] * 1024


# ----------------------------------------------------------------------
# Events Loader
# ----------------------------------------------------------------------
def event_to_text(event: Dict[str, Any]) -> str:
    """
    Constructs a rich, search-optimized textual representation of an event JSON.
    """
    parts = []
    if "event_id" in event:
        parts.append(f"Event ID: {event['event_id']}")
    if "timestamp" in event:
        parts.append(f"Timestamp: {event['timestamp']}")
    if "channel" in event:
        parts.append(f"Channel: {event['channel']}")
    if "from" in event:
        parts.append(f"From: {event['from']}")
    if "to" in event:
        parts.append(f"To: {event['to']}")
    if "subject" in event:
        parts.append(f"Subject: {event['subject']}")
    if "body" in event:
        parts.append(f"Body: {event['body']}")

    if not parts:
        return json.dumps(event, ensure_ascii=False)
    return " | ".join(parts)


def load_all_events(events_dir: str = "bundle/events") -> List[Dict[str, Any]]:
    """
    Loads all event objects from JSONL and JSON files inside the specified directory.
    Gracefully skips invalid or malformed JSON lines.
    """
    target_dir = events_dir
    if not os.path.isdir(target_dir):
        # Check fallback locations
        candidates = ["events", "bundle/events", os.path.join("..", "bundle", "events")]
        for c in candidates:
            if os.path.isdir(c):
                target_dir = c
                break

    if not os.path.isdir(target_dir):
        print(f"[Heimdall WARN] Events directory '{events_dir}' does not exist.")
        return []

    events: List[Dict[str, Any]] = []

    # Search for all .jsonl and .json files
    pattern_jsonl = os.path.join(target_dir, "*.jsonl")
    pattern_json = os.path.join(target_dir, "*.json")
    files = sorted(glob.glob(pattern_jsonl) + glob.glob(pattern_json))

    for filepath in files:
        filename = os.path.basename(filepath)
        if filepath.endswith(".jsonl"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, start=1):
                        line_str = line.strip()
                        if not line_str:
                            continue
                        try:
                            item = json.loads(line_str)
                            if isinstance(item, dict):
                                item["_source_file"] = filename
                                events.append(item)
                        except json.JSONDecodeError:
                            # Malformed line skipped gracefully as requested
                            print(f"[Heimdall] Skipped malformed JSON line {line_num} in {filename}")
            except Exception as e:
                print(f"[Heimdall ERROR] Failed to read {filename}: {e}")

        elif filepath.endswith(".json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                item["_source_file"] = filename
                                events.append(item)
                    elif isinstance(data, dict):
                        if "events" in data and isinstance(data["events"], list):
                            for item in data["events"]:
                                if isinstance(item, dict):
                                    item["_source_file"] = filename
                                    events.append(item)
                        else:
                            data["_source_file"] = filename
                            events.append(data)
            except Exception as e:
                print(f"[Heimdall ERROR] Failed to parse JSON {filename}: {e}")

    print(f"[Heimdall] Successfully loaded {len(events)} events from '{target_dir}'.")
    return events


# ----------------------------------------------------------------------
# Agent Heimdall
# ----------------------------------------------------------------------
class HeimdallAgent:
    """
    Agent Heimdall:
    1. Loads events from events folder.
    2. Embeds all event JSONs via bge-m3.
    3. Embeds OdinState.query via bge-m3.
    4. Computes cosine similarity and identifies the event JSON with maximum similarity.
    5. Calls Claude Haiku using HEIMDALL_PROMPT.
    6. Updates OdinState.heimdall_review.
    """

    def __init__(
        self,
        events_dir: str = "bundle/events",
        model_name: str = "claude-3-5-haiku-20241022",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "bge-m3",
        api_key: Optional[str] = None
    ):
        self.events_dir = events_dir
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")

        # In-memory cached events and embeddings
        self.events: List[Dict[str, Any]] = []
        self.event_embeddings: List[List[float]] = []

        # Initialize Claude Haiku LLM
        self.llm = None
        self.use_mock = False
        if self.api_key:
            try:
                from langchain_anthropic import ChatAnthropic
                self.llm = ChatAnthropic(
                    model_name=self.model_name,
                    anthropic_api_key=self.api_key,
                    temperature=0.0
                )
            except Exception as e:
                print(f"[Heimdall WARN] Could not initialize ChatAnthropic: {e}")
                self.use_mock = True
        else:
            self.use_mock = True

    def initialize_events(self):
        """Loads events and generates embeddings if not already loaded."""
        if not self.events:
            self.events = load_all_events(self.events_dir)
            if self.events:
                print(f"[Heimdall] Generating bge-m3 embeddings for {len(self.events)} events...")
                texts = [event_to_text(e) for e in self.events]
                self.event_embeddings = embed_texts_with_ollama(
                    texts,
                    base_url=self.ollama_base_url,
                    model=self.ollama_model
                )
                print(f"[Heimdall] Embeddings successfully generated for {len(self.event_embeddings)} events.")

    def find_most_relevant_event(self, query: str) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Embeds the query using bge-m3 and finds the event JSON with the maximum cosine similarity.
        Returns: (matched_event_dict, max_similarity_score)
        """
        self.initialize_events()

        if not self.events or not self.event_embeddings:
            print("[Heimdall WARN] No events available to search.")
            return None, 0.0

        print(f"\n[🛡️ HEIMDALL] Embedding query with bge-m3: '{query}'")
        query_vec = embed_single_text_with_ollama(
            query,
            base_url=self.ollama_base_url,
            model=self.ollama_model
        )

        best_score = -1.0
        best_event = None

        print(f"[🛡️ HEIMDALL] Calculating cosine similarity across {len(self.events)} events...")
        for idx, (event, emb) in enumerate(zip(self.events, self.event_embeddings)):
            sim = cosine_similarity(query_vec, emb)
            if sim > best_score:
                best_score = sim
                best_event = event

        if best_event:
            print(f"[🛡️ HEIMDALL] Maximum similarity match: [{best_event.get('event_id', 'Unknown')}] "
                  f"(Score: {best_score:.4f})")
            print(f"             Subject: {best_event.get('subject', 'N/A')}")
            print(f"             Timestamp: {best_event.get('timestamp', 'N/A')}")
        else:
            print("[🛡️ HEIMDALL] No matching event found.")

        return best_event, best_score

    def generate_review(self, query: str, matched_event: Dict[str, Any], similarity_score: float) -> str:
        """
        Calls Claude Haiku using HEIMDALL_PROMPT and the matched event JSON.
        """
        print(f"\n[🛡️ HEIMDALL] Generating review with Claude Haiku ({self.model_name})...")

        event_clean = {k: v for k, v in matched_event.items() if not k.startswith("_")}
        event_json_str = json.dumps(event_clean, indent=2, ensure_ascii=False)

        # Check if user provided custom prompt or placeholder
        system_prompt = HEIMDALL_PROMPT

        if self.use_mock or self.llm is None:
            # Informative mock review
            review = (
                f"[Heimdall Operational Review ({self.model_name})]:\n"
                f"- Matched Event ID: {matched_event.get('event_id')}\n"
                f"- Similarity Score: {similarity_score:.4f}\n"
                f"- Timestamp: {matched_event.get('timestamp')}\n"
                f"- Channel: {matched_event.get('channel')}\n"
                f"- Subject: {matched_event.get('subject')}\n"
                f"- Query: '{query}'\n"
                f"- Event Summary: {matched_event.get('body', '')[:180]}..."
            )
            return review

        try:
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Operational Event Search Query: {query}\n\n"
                          "Most Relevant Event Log (Cosine Similarity: {similarity:.4f}):\n"
                          "```json\n{event_json}\n```\n\n"
                          "Please analyze this event in relation to the query according to your instructions.")
            ])
            chain = prompt | self.llm
            response = chain.invoke({
                "query": query,
                "similarity": similarity_score,
                "event_json": event_json_str
            })
            return response.content.strip()
        except Exception as e:
            return f"[Heimdall Review Fallback]: Evaluated event {matched_event.get('event_id')} for query '{query}'. (Error: {e})"

    def evaluate(self, odin_state: OdinState) -> Dict[str, Any]:
        """
        Main entry point for Agent Heimdall:
        1. Reads query from OdinState.
        2. Finds event JSON with max cosine similarity.
        3. Generates review using Claude Haiku and HEIMDALL_PROMPT.
        4. Updates odin_state.heimdall_review with LLM output.
        5. Returns structured result.
        """
        query = odin_state.query
        if not query:
            print("[🛡️ HEIMDALL] Warning: OdinState.query is empty.")

        matched_event, score = self.find_most_relevant_event(query)

        if not matched_event:
            review_text = "No relevant operational event found."
            odin_state.update_heimdall_review(review_text)
            return {
                "matched_event": None,
                "similarity_score": 0.0,
                "heimdall_review": review_text
            }

        # Generate review via Claude Haiku
        review_text = self.generate_review(query, matched_event, score)

        # Update common state
        odin_state.update_heimdall_review(review_text)
        print("         -> Successfully updated 'heimdall_review' in OdinState!")

        return {
            "matched_event": matched_event,
            "similarity_score": round(score, 4),
            "heimdall_review": review_text
        }


# ----------------------------------------------------------------------
# CLI and Direct Execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent Heimdall - Event Monitoring & Vector Similarity Agent")
    parser.add_argument("query", nargs="?", default="outage service disruption march compute failure",
                        help="Query to search events against")
    parser.add_argument("--events_dir", default="bundle/events", help="Directory containing event files")
    args = parser.parse_args()

    print("=" * 80)
    print("PROJECT ODIN — AGENT HEIMDALL (EVENT MONITORING)")
    print("=" * 80)

    # Initialize common shared state
    state = OdinState(query=args.query)
    print(f"Initial OdinState query: '{state.query}'")

    # Run Heimdall
    agent = HeimdallAgent(events_dir=args.events_dir)
    result = agent.evaluate(state)

    print("\n" + "=" * 80)
    print("HEIMDALL EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Query: {state.query}")
    matched = result["matched_event"]
    if matched:
        print(f"Matched Event ID : {matched.get('event_id')}")
        print(f"Channel          : {matched.get('channel')}")
        print(f"Timestamp        : {matched.get('timestamp')}")
        print(f"Subject          : {matched.get('subject')}")
        print(f"Similarity Score : {result['similarity_score']}")
    print("\nOdinState.heimdall_review:")
    print("-" * 50)
    print(state.heimdall_review)
    print("-" * 50)
    print(f"\nFinal OdinState JSON:")
    print(json.dumps(state.to_dict(), indent=2))

