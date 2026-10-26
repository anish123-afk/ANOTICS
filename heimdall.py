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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings

# Standard LangChain OllamaEmbeddings instance for bge-m3
embeddings = OllamaEmbeddings(
    model="bge-m3"
)

# Load environment variables (e.g. ANTHROPIC_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ----------------------------------------------------------------------
# Heimdall Prompt Template
# Strictly constrained prompt template yielding ONLY the required answers:
# - query: active user query from OdinState
# - content: full retrieved document content from OdinState
# - retrieved_clause_content: extracted operative clause text from OdinState
# - event_json: extracted event JSON with the highest cosine similarity
# (Also available: retrived_clause_content, matched_event_json, event_id, similarity)
# ----------------------------------------------------------------------
HEIMDALL_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You are HEIMDALL, the Event & Operational Intelligence Agent for Project Odin.

Your task is to cross-examine operational incident logs and real-world event records against retrieved contractual clauses, providing ONLY the required, factual answer to the query.

STRICT OPERATIONAL RULES:
1. ONLY THE REQUIRED ANSWER: Deliver the factual determination immediately and concisely. Do not provide conversational filler, pleasantries, or introductory chatter (e.g., no "Certainly!", "Based on the event log...", "Here is the summary...").
2. EVIDENCE ANCHORED: Ground your findings strictly in the matched event JSON and the retrieved clause content. Cite the Event ID, channel, and exact timestamp.
3. CONCISE & OBJECTIVE: State whether real-world events prove a breach, satisfy a requirement, qualify as scheduled maintenance, or trigger financial remedies.
4. DEFENSE / OPERATIONAL IMPACT: Explain in one concise sentence the operational reality (e.g. root cause, whether advance notice was given, or vendor fault).

FORMAT YOUR RESPONSE IN EXACTLY THIS THREE-PART STRUCTURE (NO OTHER TEXT):
• DIRECT FACTUAL FINDING: <Immediate, factual determination answering the query using the event record>
• EVENT EVIDENCE: <Event ID, timestamp, channel, and verbatim excerpt proving the fact>
• CONTRACTUAL IMPACT: <1 concise sentence on whether this event triggers, cures, or mitigates clause liability>"""),
    ("human", """User Query:
{query}

Retrieved Contract Clause Content:
```
{retrieved_clause_content}
```

Full Document Context:
```
{content}
```

Extracted Operational Event Log (Highest Similarity Match: {similarity}):
```json
{event_json}
```

Provide ONLY the required answer conforming strictly to the requested three-part structure.""")
])

# Backwards compatibility alias
HEIMDALL_PROMPT = HEIMDALL_PROMPT_TEMPLATE


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
    Generates embeddings for a list of text strings using local Ollama bge-m3 via langchain_ollama.
    Supports both batch /api/embed and fallback /api/embeddings.
    Includes a deterministic fallback if Ollama service is offline/unreachable.
    """
    if not texts:
        return []

    try:
        emb_client = OllamaEmbeddings(model=model, base_url=base_url)
        return emb_client.embed_documents(texts)
    except Exception as e:
        # Fallback to direct HTTP /api/embed if needed
        url = f"{base_url.rstrip('/')}/api/embed"
        payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if "embeddings" in data and len(data["embeddings"]) == len(texts):
                    return data["embeddings"]
        except Exception:
            # Try individual requests to /api/embeddings
            embed_results: List[List[float]] = []
            fallback_url = f"{base_url.rstrip('/')}/api/embeddings"
            success = True

            for text in texts:
                fb_payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
                fb_req = urllib.request.Request(fallback_url, data=fb_payload, headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(fb_req, timeout=timeout) as fb_resp:
                        fb_data = json.loads(fb_resp.read().decode("utf-8"))
                        if "embedding" in fb_data:
                            embed_results.append(fb_data["embedding"])
                        else:
                            success = False
                            break
                except Exception:
                    success = False
                    break

            if success and len(embed_results) == len(texts):
                return embed_results

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
    """Helper to embed a single text string using langchain_ollama OllamaEmbeddings."""
    try:
        emb_client = OllamaEmbeddings(model=model, base_url=base_url)
        return emb_client.embed_query(text)
    except Exception:
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
        api_key: Optional[str] = None,
        prompt_template: Optional[Any] = None
    ):
        self.events_dir = events_dir
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.prompt_template = prompt_template or HEIMDALL_PROMPT_TEMPLATE

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

    def generate_review(
        self,
        odin_state_or_query: Any,
        matched_event: Dict[str, Any],
        similarity_score: float
    ) -> str:
        """
        Calls Claude Haiku using HEIMDALL_PROMPT_TEMPLATE with dynamic input variables:
        - query
        - content
        - retrieved_clause_content
        - event_json
        """
        if isinstance(odin_state_or_query, OdinState):
            odin_state = odin_state_or_query
            query = odin_state.query
            retrieved_doc = odin_state.retrieved_document or {}
            content = (
                retrieved_doc.get("content") or 
                retrieved_doc.get("clause_text") or 
                retrieved_doc.get("snippet") or 
                ""
            )
            retrieved_clause_content = (
                odin_state.retrieved_clause_content or 
                content or 
                "No clause content available in state."
            )
        else:
            query = str(odin_state_or_query)
            content = "No content provided."
            retrieved_clause_content = "No clause content provided."

        print(f"\n[🛡️ HEIMDALL] Generating review with Prompt Template ({self.model_name})...")
        print(f"         - Input query: '{query}'")
        print(f"         - Input content: {len(content)} chars")
        print(f"         - Input retrieved_clause_content: {len(retrieved_clause_content)} chars")

        event_clean = {k: v for k, v in matched_event.items() if not k.startswith("_")}
        event_json_str = json.dumps(event_clean, indent=2, ensure_ascii=False)

        template_inputs = {
            "query": query,
            "content": content,
            "retrieved_clause_content": retrieved_clause_content,
            "retrived_clause_content": retrieved_clause_content,  # spelling alias
            "event_json": event_json_str,
            "matched_event_json": event_json_str,
            "event_id": matched_event.get("event_id", "N/A"),
            "similarity": f"{similarity_score:.4f}",
            "similarity_score": similarity_score,
            "subject": matched_event.get("subject", "N/A"),
            "timestamp": matched_event.get("timestamp", "N/A"),
            "channel": matched_event.get("channel", "N/A")
        }

        if self.use_mock or self.llm is None:
            return (
                f"• DIRECT FACTUAL FINDING: Operational event {matched_event.get('event_id')} relates directly to '{query}'. "
                f"Log confirms '{matched_event.get('subject')}'.\n"
                f"• EVENT EVIDENCE: Event {matched_event.get('event_id')} ({matched_event.get('timestamp')}, {matched_event.get('channel')}) — "
                f"Excerpt: \"{matched_event.get('body', '')[:140].strip()}...\"\n"
                f"• CONTRACTUAL IMPACT: This event log serves as verifiable real-world evidence for evaluating compliance against the governing clause."
            )

        try:
            chain = self.prompt_template | self.llm | StrOutputParser()
            response = chain.invoke(template_inputs)
            return response.strip()
        except Exception as e:
            return (
                f"• DIRECT FACTUAL FINDING: Matched event {matched_event.get('event_id')} for query '{query}'.\n"
                f"• EVENT EVIDENCE: Event {matched_event.get('event_id')} ({matched_event.get('timestamp')}).\n"
                f"• CONTRACTUAL IMPACT: Event details require reconciliation with contractual obligations. (Error: {e})"
            )

    def evaluate(self, odin_state: OdinState) -> Dict[str, Any]:
        """
        Main entry point for Agent Heimdall:
        1. Reads query from OdinState.
        2. Finds event JSON with max cosine similarity.
        3. Generates review using Claude Haiku and HEIMDALL_PROMPT_TEMPLATE.
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

        # Generate review via Claude Haiku using dynamic template inputs from state
        review_text = self.generate_review(odin_state, matched_event, score)

        # Update common state
        odin_state.update_heimdall_review(review_text)
        odin_state.update_matched_event(matched_event)
        print("         -> Successfully updated 'heimdall_review' and 'matched_event' in OdinState!")

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

