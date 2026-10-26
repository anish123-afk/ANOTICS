"""
Agent Tyr — Supreme Arbiter & Judgment Agent for Project Odin
=============================================================
Responsibilities:
1. Takes as input from OdinState:
   - `loki_interpretation` (Adversarial contract review and legal attack analysis)
   - `heimdall_review` (Operational incident, SLA downtime, and event log monitoring)
2. Synthesizes both perspectives using Claude Haiku and the first prompt (`TYR_JUDGMENT_PROMPT`)
   to produce the `final_judgment` of Tyr.
3. Updates `odin_state.final_judgment`.
4. Analyzes the `final_judgment` using Claude Haiku and the second prompt (`TYR_FINAL_ANSWER_PROMPT`)
   to derive the definitive `final_ans`.
5. Updates `odin_state.final_ans`.
6. Presents the final answer and requests human approval (Human-in-the-Loop),
   updating `odin_state.human_approval` and `odin_state.human_feedback`.
"""

import os
import sys
import json
from typing import Optional, Dict, Any

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from odin_state import OdinState
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (e.g. ANTHROPIC_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ----------------------------------------------------------------------
# Prompt 1: Tyr Final Judgment Prompt Template
# Strictly constrained prompt template yielding ONLY the required judgment:
# Inputs:
# - query: active user query from OdinState
# - content: full retrieved document content from OdinState
# - retrieved_clause_content: extracted operative clause text from OdinState
# - loki_interpretation: Loki's adversarial legal interpretation
# - heimdall_review: Heimdall's operational event analysis
# - event_json: extracted event JSON with highest similarity
# ----------------------------------------------------------------------
TYR_JUDGMENT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You are TYR, the Supreme Arbitrator and Judicial Ruling Agent for Project Odin.

Your task is to impartially weigh Loki's adversarial legal interpretation against Heimdall's operational event defense, evaluated in the context of the governing contract clause, and render ONLY the required judicial ruling.

STRICT OPERATIONAL RULES:
1. ONLY THE REQUIRED ANSWER: Deliver the judicial ruling directly and factually. Do not provide conversational filler, pleasantries, or introductory chatter (e.g., no "Certainly!", "As Supreme Arbiter...", "Here is my judgment...").
2. EVIDENCE ANCHORED: Whichever position is proven by the contract clause text and the verified event telemetry prevails.
3. CONCISE & DEFINITIVE: State the verdict, synthesize the conflict in 1-2 tight sentences, cite the key clause and event IDs, and specify the exact financial exposure or credit amount.

FORMAT YOUR RESPONSE IN EXACTLY THIS FOUR-PART STRUCTURE (NO OTHER TEXT):
• VERDICT: <FINDING_SUPPORTED | FINDING_REJECTED | FINDING_REVISED | COMPROMISE>
• FACTUAL & LEGAL SYNTHESIS: <Direct 1-2 sentence ruling balancing Loki's legal claim against Heimdall's event proof and the contract clause>
• GOVERNING EVIDENCE: <Specific Clause ID and Event ID that determined this ruling>
• CONFIRMED LIABILITY / REMEDY: <Exact financial exposure, service credit percentage/amount, or contractual penalty determined>"""),
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

=== LOKI'S ADVERSARIAL LEGAL INTERPRETATION ===
{loki_interpretation}

=== HEIMDALL'S OPERATIONAL EVENT REVIEW ===
{heimdall_review}

=== TOP MATCHED OPERATIONAL EVENT JSON ===
```json
{event_json}
```

Provide ONLY the required judicial ruling conforming strictly to the requested four-part structure.""")
])

# Backwards compatibility alias
TYR_JUDGMENT_PROMPT = TYR_JUDGMENT_PROMPT_TEMPLATE


# ----------------------------------------------------------------------
# Prompt 2: Tyr Final Answer Prompt Template
# Strictly constrained prompt template yielding ONLY the required answer:
# Inputs:
# - query: active user query from OdinState
# - content: full retrieved document content from OdinState
# - retrieved_clause_content: extracted operative clause text from OdinState
# - final_judgment: synthesized ruling from Tyr
# - event_json: extracted event JSON with highest similarity
# ----------------------------------------------------------------------
TYR_FINAL_ANSWER_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You are TYR in your capacity as Senior Legal Counsel and Final Decision Engine for Project Odin.

Your task is to convert Tyr's judicial ruling (`final_judgment`) into the definitive, actionable resolution that answers the user's query.

STRICT OPERATIONAL RULES:
1. ONLY THE REQUIRED ANSWER: Output the definitive resolution and actionable next step immediately. No preambles, pleasantries, or conversational filler.
2. CONCISE & ACTIONABLE: Provide the direct settlement answer, the immediate operational/legal action required, and the governing citations.
3. ACCURATE TO JUDGMENT: Ground the answer strictly in the ruling, clause content, and event facts.

FORMAT YOUR RESPONSE IN EXACTLY THIS THREE-PART STRUCTURE (NO OTHER TEXT):
• DEFINITIVE RESOLUTION: <Clear, decisive answer resolving the user's specific query based on the ruling>
• IMMEDIATE ACTION REQUIRED: <Exact legal, financial, or operational action to execute>
• GOVERNING CITATION: <Contract ID, Clause ID, and Event ID governing the resolution>"""),
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

=== TYR'S FINAL JUDGMENT ===
{final_judgment}

=== TOP MATCHED OPERATIONAL EVENT JSON ===
```json
{event_json}
```

Provide ONLY the required final answer conforming strictly to the requested three-part structure.""")
])

# Backwards compatibility alias
TYR_FINAL_ANSWER_PROMPT = TYR_FINAL_ANSWER_PROMPT_TEMPLATE


# ----------------------------------------------------------------------
# Agent Tyr
# ----------------------------------------------------------------------
class TyrAgent:
    """
    Agent Tyr (The Supreme Arbiter):
    - Synthesizes adversarial legal risks (Loki) and operational reality (Heimdall).
    - Produces a reasoned legal judgment (`final_judgment`).
    - Distills the judgment into a definitive operational/legal answer (`final_ans`).
    - Facilitates Human-in-the-Loop (HITL) review and approval.
    """

    def __init__(
        self,
        model_name: str = "claude-3-5-haiku-20241022",
        api_key: Optional[str] = None,
        judgment_prompt_template: Optional[Any] = None,
        final_answer_prompt_template: Optional[Any] = None
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.judgment_prompt_template = judgment_prompt_template or TYR_JUDGMENT_PROMPT_TEMPLATE
        self.final_answer_prompt_template = final_answer_prompt_template or TYR_FINAL_ANSWER_PROMPT_TEMPLATE

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
                print(f"[Tyr WARN] Could not initialize ChatAnthropic: {e}")
                self.use_mock = True
        else:
            self.use_mock = True

    def synthesize_judgment(
        self,
        odin_state: OdinState,
        matched_event: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Step 1: Uses TYR_JUDGMENT_PROMPT_TEMPLATE with dynamic input variables:
        - query
        - content
        - retrieved_clause_content
        - loki_interpretation
        - heimdall_review
        - event_json
        and updates odin_state.final_judgment.
        """
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
        loki_text = odin_state.loki_interpretation or "No Loki interpretation available."
        heimdall_text = odin_state.heimdall_review or "No Heimdall review available."

        # Get matched event JSON
        event_obj = matched_event or getattr(odin_state, "matched_event", None) or {}
        event_clean = {k: v for k, v in event_obj.items() if not str(k).startswith("_")}
        event_json_str = json.dumps(event_clean, indent=2, ensure_ascii=False) if event_clean else "{}"

        template_inputs = {
            "query": query,
            "content": content,
            "retrieved_clause_content": retrieved_clause_content,
            "retrived_clause_content": retrieved_clause_content,  # spelling alias
            "loki_interpretation": loki_text,
            "heimdall_review": heimdall_text,
            "event_json": event_json_str,
            "matched_event_json": event_json_str,
            "document_name": retrieved_doc.get("document_name", "Unknown"),
            "contract_id": retrieved_doc.get("contract_id", "Unknown"),
            "clause_id": retrieved_doc.get("clause_id", "Unknown")
        }

        print(f"\n[⚖️ TYR] Step 1: Synthesizing Final Judgment with Prompt Template...")
        print(f"       - Input query: '{query}'")
        print(f"       - Input loki_interpretation: {len(loki_text)} chars")
        print(f"       - Input heimdall_review: {len(heimdall_text)} chars")
        print(f"       - Input retrieved_clause_content: {len(retrieved_clause_content)} chars")
        print(f"       - Input event_json: {len(event_json_str)} chars")

        if self.use_mock or self.llm is None:
            judgment = (
                f"• VERDICT: FINDING_REVISED\n"
                f"• FACTUAL & LEGAL SYNTHESIS: Weighing Loki's legal attack regarding '{query}' against Heimdall's verified event log ({event_obj.get('event_id', 'E-05')}), the contractual claim is partially substantiated by real-world telemetry.\n"
                f"• GOVERNING EVIDENCE: Contract {retrieved_doc.get('document_name', 'B2')} ({retrieved_doc.get('clause_id', 'Operative Terms')}) and Event {event_obj.get('event_id', 'E-05')}.\n"
                f"• CONFIRMED LIABILITY / REMEDY: Liability is capped per standard terms, and service credits apply strictly for verified downtime minutes."
            )
        else:
            try:
                chain = self.judgment_prompt_template | self.llm | StrOutputParser()
                judgment = chain.invoke(template_inputs).strip()
            except Exception as e:
                judgment = (
                    f"• VERDICT: COMPROMISE\n"
                    f"• FACTUAL & LEGAL SYNTHESIS: Synthesized findings for '{query}'.\n"
                    f"• GOVERNING EVIDENCE: {retrieved_doc.get('clause_id', 'Clause')} & {event_obj.get('event_id', 'Event')}\n"
                    f"• CONFIRMED LIABILITY / REMEDY: Reconcile operational telemetry against agreed SLA schedule. (Error: {e})"
                )

        odin_state.update_final_judgment(judgment)
        print("       -> Successfully stored 'final_judgment' in OdinState!")
        return judgment

    def derive_final_answer(
        self,
        odin_state: OdinState,
        matched_event: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Step 2: Uses TYR_FINAL_ANSWER_PROMPT_TEMPLATE with dynamic input variables:
        - query
        - content
        - retrieved_clause_content
        - final_judgment
        - event_json
        and updates odin_state.final_ans.
        """
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
        judgment = odin_state.final_judgment or "No judgment available."

        event_obj = matched_event or getattr(odin_state, "matched_event", None) or {}
        event_clean = {k: v for k, v in event_obj.items() if not str(k).startswith("_")}
        event_json_str = json.dumps(event_clean, indent=2, ensure_ascii=False) if event_clean else "{}"

        template_inputs = {
            "query": query,
            "content": content,
            "retrieved_clause_content": retrieved_clause_content,
            "retrived_clause_content": retrieved_clause_content,  # spelling alias
            "final_judgment": judgment,
            "final_judgement": judgment,  # spelling alias
            "event_json": event_json_str,
            "matched_event_json": event_json_str,
            "document_name": retrieved_doc.get("document_name", "Unknown"),
            "contract_id": retrieved_doc.get("contract_id", "Unknown"),
            "clause_id": retrieved_doc.get("clause_id", "Unknown")
        }

        print(f"\n[⚖️ TYR] Step 2: Deriving Definitive Final Answer with Prompt Template...")
        print(f"       - Input final_judgment: {len(judgment)} chars")

        if self.use_mock or self.llm is None:
            ans = (
                f"• DEFINITIVE RESOLUTION: In response to '{query}', apply the negotiated SLA service credit based on verified downtime telemetry without admitting general breach liability.\n"
                f"• IMMEDIATE ACTION REQUIRED: Issue formal credit memorandum for the affected calendar month and update operational uptime records.\n"
                f"• GOVERNING CITATION: Contract {retrieved_doc.get('contract_id', 'Agreement')} ({retrieved_doc.get('clause_id', 'SLA')}) & Incident Log {event_obj.get('event_id', 'E-05')}."
            )
        else:
            try:
                chain = self.final_answer_prompt_template | self.llm | StrOutputParser()
                ans = chain.invoke(template_inputs).strip()
            except Exception as e:
                ans = (
                    f"• DEFINITIVE RESOLUTION: Resolution for query '{query}'.\n"
                    f"• IMMEDIATE ACTION REQUIRED: Reconcile operational telemetry with contract terms.\n"
                    f"• GOVERNING CITATION: {retrieved_doc.get('contract_id', 'Agreement')}. (Error: {e})"
                )

        odin_state.update_final_ans(ans)
        print("       -> Successfully stored 'final_ans' in OdinState!")
        return ans

    def request_human_approval(
        self,
        odin_state: OdinState,
        interactive: bool = True,
        auto_approve: Optional[bool] = None
    ) -> bool:
        """
        Step 3: Presents the final answer to a human reviewer and asks for approval.
        Updates odin_state.human_approval and odin_state.human_feedback.
        """
        ans = odin_state.final_ans or "(No final answer generated yet)"

        print("\n" + "=" * 80)
        print("HUMAN APPROVAL REQUIRED (HUMAN-IN-THE-LOOP)")
        print("=" * 80)
        print(f"Target Query: {odin_state.query}")
        print("\nTyr's Proposed Final Answer:")
        print("-" * 50)
        print(ans)
        print("-" * 50)

        # If auto_approve is explicitly set (e.g., testing or batch script)
        if auto_approve is not None:
            odin_state.update_human_approval(auto_approve, feedback="Set programmatically via auto_approve")
            status_text = "APPROVED" if auto_approve else "REJECTED"
            print(f"[Tyr Human Review]: Auto-{status_text} (Non-interactive mode).")
            return auto_approve

        # Interactive terminal check
        if interactive and sys.stdin and sys.stdin.isatty():
            try:
                choice = input("\nDo you approve Tyr's final answer? [Y/N]: ").strip().lower()
                if choice in ["y", "yes"]:
                    feedback = input("Optional comments/feedback (or press Enter to skip): ").strip()
                    odin_state.update_human_approval(True, feedback=feedback or None)
                    print("✅ Final answer APPROVED by human reviewer.")
                    return True
                else:
                    feedback = input("Reason for rejection or requested changes: ").strip()
                    odin_state.update_human_approval(False, feedback=feedback or "Rejected by human reviewer")
                    print("❌ Final answer REJECTED by human reviewer.")
                    return False
            except (EOFError, KeyboardInterrupt):
                print("\n[Tyr Human Review] Interaction interrupted. Setting approval to pending.")
                odin_state.update_human_approval(False, feedback="Interrupted/Pending human review")
                return False
        else:
            # Non-interactive shell fallback (mark as pending human review)
            print("[Tyr Human Review]: Running in non-interactive environment. Human approval marked as PENDING.")
            odin_state.update_human_approval(False, feedback="Pending interactive human review")
            return False

    def evaluate(
        self,
        odin_state: OdinState,
        interactive_approval: bool = True,
        auto_approve: Optional[bool] = None,
        matched_event: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Full Execution Pipeline for Agent Tyr:
        1. Synthesize final_judgment from loki_interpretation, heimdall_review, clause content, and event JSON.
        2. Derive final_ans from final_judgment, clause content, and event JSON.
        3. Request human approval and update common state.
        """
        judgment = self.synthesize_judgment(odin_state, matched_event=matched_event)
        ans = self.derive_final_answer(odin_state, matched_event=matched_event)
        approved = self.request_human_approval(
            odin_state,
            interactive=interactive_approval,
            auto_approve=auto_approve
        )

        return {
            "final_judgment": judgment,
            "final_ans": ans,
            "human_approval": approved,
            "human_feedback": odin_state.human_feedback
        }


# ----------------------------------------------------------------------
# CLI and Direct Execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent Tyr - Arbiter & Human Approval Agent")
    parser.add_argument("--query", default="SLA downtime penalty for March 3 outage",
                        help="Query to evaluate")
    parser.add_argument("--auto_approve", action="store_true",
                        help="Automatically approve without waiting for interactive input")
    args = parser.parse_args()

    print("=" * 80)
    print("PROJECT ODIN — AGENT TYR (FINAL ARBITER & DECISION ENGINE)")
    print("=" * 80)

    # Prepare sample common state with Loki and Heimdall inputs
    state = OdinState(query=args.query)
    state.update_loki_interpretation(
        "LOKI FINDING: Section 3.5 provides Harbor Retail the unilateral right to claim a 15% service credit "
        "($6,000) for March downtime. Furthermore, Section 8.1 allows termination for cause if SLA is missed twice."
    )
    state.update_heimdall_review(
        "HEIMDALL EVENT LOG: Event E-05 and E-04 confirm the March 3 outage lasted 190 minutes due to upstream "
        "StratoNode network switch failure (us-west-2). No Scheduled Maintenance was in effect."
    )

    print("\nInitial Shared OdinState:")
    print(f"- Query               : {state.query}")
    print(f"- loki_interpretation : {state.loki_interpretation[:90]}...")
    print(f"- heimdall_review     : {state.heimdall_review[:90]}...")

    # Initialize Tyr Agent
    tyr = TyrAgent()

    # Run Tyr
    result = tyr.evaluate(
        state,
        interactive_approval=True,
        auto_approve=True if args.auto_approve else None
    )

    print("\n" + "=" * 80)
    print("TYR EXECUTION SUMMARY")
    print("=" * 80)
    print("Final OdinState representation:")
    print(state)
    print("\nFinal OdinState JSON:")
    print(json.dumps(state.to_dict(), indent=2))

