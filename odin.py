"""
Odin Orchestrator — Main Pipeline & Multi-Agent Executive Controller
====================================================================
Architecture Chain (as specified in system workflow):
  [Odin] 
    │
    ▼
  [Thor] ◄───────────────┐
    │                    │ (if not approved: reExtract == True)
    ▼                    │
  [Loki] ────────────────┘
    │
    │ (if approved: reExtract == False)
    ▼
  [Norns]
    │
    ▼
  [Loki] (Contract & Legal Interpretation)
    │
    ▼
  [Heimdall] (Operational Event & Incident Logs)
    │
    ├─── [Loki Interpretation]
    └─── [Heimdall Review]
    │
    ▼
  [Tyr] (Supreme Arbiter: Final Judgment)
    │
    ▼
  [Final Ans] (Definitive Resolution + Human Approval)
"""

import os
import sys
import json
import argparse
from typing import Optional, Dict, Any, List

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Project Odin Components
from odin_state import OdinState
from thor_loki import ThorAgent, LokiAgent
from norns import NornsAgent
from heimdall import HeimdallAgent
from tyr import TyrAgent


class OdinOrchestrator:
    """
    Main Orchestrator for Project Odin.
    Coordinates the multi-agent chain across Thor, Loki, Norns, Heimdall, and Tyr.
    """

    def __init__(
        self,
        manifest_path: str = "manifest.json",
        db_dir: str = "chroma_db",
        clock_path: str = "clock.json",
        atreus_csv_path: str = "contract_details_summary_v2.csv",
        events_dir: str = "bundle/events",
        model_name: str = "claude-3-5-haiku-20241022",
        api_key: Optional[str] = None,
        max_loop_iterations: int = 3
    ):
        self.manifest_path = manifest_path
        self.db_dir = db_dir
        self.clock_path = clock_path
        self.atreus_csv_path = atreus_csv_path
        self.events_dir = events_dir
        self.model_name = model_name
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.max_loop_iterations = max_loop_iterations

        # Initialize Sub-Agents
        print("[⚡ ODIN] Initializing Agent Ecosystem...")
        self.thor = ThorAgent(db_dir=self.db_dir, max_iteration=self.max_loop_iterations)
        self.loki = LokiAgent(
            manifest_path=self.manifest_path,
            model_name=self.model_name,
            api_key=self.api_key
        )
        self.norns = NornsAgent(
            model_name=self.model_name,
            api_key=self.api_key
        )
        self.heimdall = HeimdallAgent(
            events_dir=self.events_dir,
            model_name=self.model_name,
            api_key=self.api_key
        )
        self.tyr = TyrAgent(
            model_name=self.model_name,
            api_key=self.api_key
        )
        print("[⚡ ODIN] All agents (Thor, Loki, Norns, Heimdall, Tyr) ready.\n")

    def run(
        self,
        query: str,
        checkpoint_val: Any = 0,
        interactive_approval: bool = True,
        auto_approve: Optional[bool] = None
    ) -> OdinState:
        """
        Executes the full pipeline chain according to the workflow specification.
        """
        print("=" * 80)
        print("⚡ PROJECT ODIN: MULTI-AGENT ORCHESTRATION PIPELINE ⚡")
        print("=" * 80)
        print(f"Query: \"{query}\"")
        print(f"Clock Checkpoint: {checkpoint_val}\n")

        # --------------------------------------------------------------
        # 0. Initialize Common State (OdinState)
        # --------------------------------------------------------------
        state = OdinState(query=query)
        print(f"[OdinState Initialized]: {state}\n")

        # --------------------------------------------------------------
        # 1. Loop: Thor <──> Loki (Extraction & Sequential Verification)
        # --------------------------------------------------------------
        print("=" * 80)
        print("STAGE 1: THOR <──> LOKI EXTRACTION LOOP")
        print("=" * 80)

        target_doc = None
        while self.thor.local_state.iteration <= self.thor.local_state.max_iteration:
            iter_num = self.thor.local_state.iteration
            print(f"\n--- [Iteration {iter_num}/{self.thor.local_state.max_iteration}] ---")

            # Thor extracts from vector store
            print(f"[🔨 THOR] Extracting relevant clause for query...")
            self.thor.extract_clause(state, target_contract_id=target_doc)

            retrieved = state.retrieved_document or {}
            doc_name = retrieved.get("document_name", "Unknown")
            clause_id = retrieved.get("clause_id", "Unknown")
            print(f"         -> Retrieved: [{doc_name}] Clause: {clause_id}")

            # Loki evaluates whether next document exists
            should_reextract, next_doc = self.loki.evaluate(state)

            if should_reextract:
                # "if not approved" -> loop back to Thor
                print(f"[🦊 LOKI] [NOT APPROVED] Subsequent document [{next_doc}] detected!")
                print(f"         -> Triggering reExtract loop back to Thor.")
                if iter_num >= self.thor.local_state.max_iteration:
                    print(f"[⚡ ODIN] Reached max loop iterations ({self.thor.local_state.max_iteration}). Proceeding.")
                    break
                target_doc = next_doc
                self.thor.local_state.iteration += 1
            else:
                # "if approved" -> extraction approved!
                print(f"[🦊 LOKI] [APPROVED] Final document sequence verified ({doc_name}). No further re-extraction needed.")
                break

        print("\nThor's Final Extraction JSON:")
        print(self.thor.get_final_json())

        # --------------------------------------------------------------
        # 2. Stage 2: Norns (Timeline & Non-Renewal Notice Analysis)
        # --------------------------------------------------------------
        print("\n" + "=" * 80)
        print("STAGE 2: NORNS (FINANCIAL & DEADLINE COMPLIANCE)")
        print("=" * 80)
        print("[⏳ NORNS] Evaluating contractual end dates against clock checkpoint...")

        self.norns.evaluate_portfolio(
            atreus_csv_path=self.atreus_csv_path,
            clock_path=self.clock_path,
            checkpoint_val=checkpoint_val,
            odin_state=state
        )

        print("[⏳ NORNS] Portfolio financial review complete. Common state updated.")

        # --------------------------------------------------------------
        # 3. Stage 3: Loki (Legal Interpretation & Attack Surface)
        # --------------------------------------------------------------
        print("\n" + "=" * 80)
        print("STAGE 3: LOKI (CONTRACT & LEGAL INTERPRETATION)")
        print("=" * 80)
        print("[🦊 LOKI] Analyzing final document under adversarial lens with Claude Haiku...")

        loki_output = self.loki.interpret_final_document(state)
        print(f"[🦊 LOKI] Interpretation generated ({len(loki_output)} characters).")

        # --------------------------------------------------------------
        # 4. Stage 4: Heimdall (Operational Events & Incident Logs)
        # --------------------------------------------------------------
        print("\n" + "=" * 80)
        print("STAGE 4: HEIMDALL (OPERATIONAL INCIDENT & EVENT MONITORING)")
        print("=" * 80)
        print("[🛡️ HEIMDALL] Matching operational event logs using bge-m3 cosine similarity...")

        heimdall_res = self.heimdall.evaluate(state)
        matched_event = heimdall_res.get("matched_event")
        if matched_event:
            print(f"[🛡️ HEIMDALL] Matched Incident Event: [{matched_event.get('event_id')}] "
                  f"'{matched_event.get('subject')}' (Sim: {heimdall_res.get('similarity_score')})")

        # --------------------------------------------------------------
        # 5. Stage 5: Tyr (Supreme Arbiter & Final Answer)
        # --------------------------------------------------------------
        print("\n" + "=" * 80)
        print("STAGE 5: TYR (SYNTHESIS, FINAL JUDGMENT & FINAL ANSWER)")
        print("=" * 80)
        print("[⚖️ TYR] Synthesizing Loki Legal Interpretation + Heimdall Incident Review...")

        # Tyr synthesizes judgment, derives final_ans, and requests human approval
        self.tyr.evaluate(
            state,
            interactive_approval=interactive_approval,
            auto_approve=auto_approve
        )

        # --------------------------------------------------------------
        # 6. Pipeline Complete
        # --------------------------------------------------------------
        print("\n" + "=" * 80)
        print("⚡ ODIN ORCHESTRATION PIPELINE COMPLETE ⚡")
        print("=" * 80)
        print(f"Final Target Query  : {state.query}")
        print(f"Extracted Document  : {state.retrieved_document.get('document_name') if state.retrieved_document else 'N/A'}")
        print(f"Contract ID         : {state.retrieved_document.get('contract_id') if state.retrieved_document else 'N/A'}")
        print(f"Human Approval      : {state.human_approval} (Feedback: {state.human_feedback})")
        print("\nDEFINITIVE FINAL ANSWER (odin_state.final_ans):")
        print("-" * 60)
        print(state.final_ans)
        print("-" * 60)

        return state


# ----------------------------------------------------------------------
# CLI Interface
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Odin Orchestrator: Multi-Agent Contract & Operations Intelligence System"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="Harbor Retail SLA uptime commitments, service credits, and outage remedies",
        help="Query to evaluate through the Odin ecosystem"
    )
    parser.add_argument("--checkpoint", default="0", help="Clock checkpoint (0, 1, 2 or YYYY-MM-DD)")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve Tyr final answer without waiting for terminal prompt")
    parser.add_argument("--manifest", default="manifest.json", help="Path to manifest.json")
    parser.add_argument("--db", default="chroma_db", help="Path to ChromaDB directory")
    parser.add_argument("--clock", default="clock.json", help="Path to clock.json")
    parser.add_argument("--events", default="bundle/events", help="Directory of events")

    args = parser.parse_args()

    orchestrator = OdinOrchestrator(
        manifest_path=args.manifest,
        db_dir=args.db,
        clock_path=args.clock,
        events_dir=args.events,
    )

    final_state = orchestrator.run(
        query=args.query,
        checkpoint_val=args.checkpoint,
        interactive_approval=True,
        auto_approve=True if args.auto_approve else None
    )

    print("\n[+] Full OdinState Object Dump:")
    print(json.dumps(final_state.to_dict(), indent=2))


if __name__ == "__main__":
    main()

