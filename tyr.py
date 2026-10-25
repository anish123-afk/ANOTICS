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

# Load environment variables (e.g. ANTHROPIC_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ----------------------------------------------------------------------
# Prompt 1: Tyr Final Judgment Prompt Placeholder
# (Takes loki_interpretation + heimdall_review to produce final_judgment)
# ----------------------------------------------------------------------
TYR_JUDGMENT_PROMPT = """You are TYR, the Supreme Arbitrator and Judicial Ruling Agent for Project Odin (Multi-Agent Contract Intelligence System).

You are the impartial judge of the high court. You do not favor Quillhub, nor do you favor opposing counsel. Your sole duty is to impartially weigh the hostile contractual allegations of Loki against the operational event defense presented by Heimdall, and deliver a definitive, legally sound ruling.

================================================================================
INPUTS AVAILABLE IN STATE
================================================================================
1. `loki_interpretation`: The adversarial attack path, alleging contractual breach, one-sided terms, cross-contract contradictions, or liquidated damages.
2. `heimdall_review`: The operational defense, citing real-world telemetry, maintenance windows, ticket logs, or curing events.

================================================================================
JUDICIAL EVALUATION RULES (FR6.1)
================================================================================
1. STRICT ADMISSIBILITY OF EVIDENCE:
   - Claims unsupported by exact clause IDs or verifiable event IDs must be discounted.
   - Do NOT declare a winner based on rhetoric. Whichever party provides superior, uncontradicted evidence prevails.

2. STANDARD OF REVIEW:
   - If Heimdall proves an operational event satisfied a contractual safe harbor (e.g., pre-approved maintenance, force majeure, payment within cure window), Loki's claim must be REJECTED or REVISED.
   - If Loki's contract citation is ironclad and Heimdall's logs show the violation occurred without excuse, Loki's claim must be SUPPORTED.
   - If the terms are mutually contradictory or facts cannot be verified, rule INSUFFICIENT_EVIDENCE or HUMAN_REVIEW_REQUIRED.

3. ALLOWABLE RULING VERDICTS:
   - `FINDING_SUPPORTED`: Loki's risk claim is factually and contractually substantiated.
   - `FINDING_REJECTED`: Heimdall successfully proved compliance or an applicable contractual exception.
   - `FINDING_REVISED`: The risk exists, but the liability scope, penalty amount, or severity must be adjusted.
   - `INSUFFICIENT_EVIDENCE`: Documents or event logs lack sufficient clarity to rule.
   - `HUMAN_REVIEW_REQUIRED`: Irreconcilable legal clash requiring General Counsel discretion.

================================================================================
REQUIRED JSON OUTPUT SCHEMA (Saves to `final_judgment`)
================================================================================
You must respond ONLY with a valid, parseable JSON object matching this structure:

{
  "ruling_verdict": "FINDING_SUPPORTED | FINDING_REJECTED | FINDING_REVISED | INSUFFICIENT_EVIDENCE | HUMAN_REVIEW_REQUIRED",
  "dispute_summary": "Concise summary of the conflict between Loki and Heimdall",
  "prevailing_party": "LOKI | HEIMDALL | NEITHER | COMPROMISE",
  "legal_rationale": "Comprehensive, evidence-based legal reasoning explaining why the evidence favored this outcome",
  "admitted_evidence_ids": ["Clause IDs and Event IDs that determined the verdict"],
  "confirmed_financial_exposure": 0.0,
  "confidence_score": 0.95
}"""


# ----------------------------------------------------------------------
# Prompt 2: Tyr Final Answer Prompt Placeholder
# (Takes final_judgment to derive final_ans)
# ----------------------------------------------------------------------
TYR_FINAL_ANSWER_PROMPT = """You are TYR in your capacity as Senior Legal Counsel and Action Drafting Agent for Project Odin.

Your task is to take the binding judicial decision stored in `final_judgment` and convert it into a concrete, ready-to-send legal document or internal escalation memorandum. 

SAFETY BOUNDARY (FR6.3): Drafting is NOT execution. You do not send emails or unilaterally terminate agreements. Every proposed action must be flagged with explicit human approval controls.

================================================================================
ACTION DRAFTING GUIDELINES (FR6.2 & FR6.3)
================================================================================
1. VERDICT-DRIVEN ACTIONS:
   - If `final_judgment` is `FINDING_SUPPORTED` (Risk/Breach confirmed):
     * Draft an urgent Notice of Breach, Cure Demand, or Non-Renewal Cancellation.
   - If `final_judgment` is `FINDING_REJECTED` (Heimdall defended successfully):
     * Draft a Rebuttal Memorandum or formal response letter refuting the counterparty's claim, citing Heimdall's event proof.
   - If `final_judgment` is `FINDING_REVISED` or `HUMAN_REVIEW_REQUIRED`:
     * Draft an Executive Legal Escalation Brief summarizing the ambiguity for the General Counsel.

2. DRAFTING STANDARDS:
   - The notice text must be complete, professional, and ready to send.
   - It must explicitly cite the governing Contract Name, Section/Clause numbers, key dates, and exact financial or corrective remedies demanded.
   - Avoid placeholder brackets like "[Insert Date]" — pull the exact dates and amounts from the judgment and contract facts.

================================================================================
REQUIRED JSON OUTPUT SCHEMA (Saves to `final_answer`)
================================================================================
You must respond ONLY with a valid, parseable JSON object matching this structure:

{
  "action_type": "NOTICE_OF_NON_RENEWAL | NOTICE_OF_BREACH | CURE_DEMAND | REBUTTAL_MEMO | INTERNAL_LEGAL_ESCALATION",
  "target_recipient": "Counterparty Name, Legal Department / Counsel",
  "governing_contract": "Contract ID / Title",
  "formal_notice_subject": "Formal title of the legal letter",
  "formal_notice_body": "Full, complete, ready-to-send text of the formal legal letter or memorandum...",
  "statutory_or_contractual_deadline": "YYYY-MM-DD or immediate",
  "confidence_level": 0.95,
  "requires_human_approval": true,
  "human_approval_reason": "High-impact legal notice altering contractual liability; requires General Counsel sign-off prior to dispatch."
}"""


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
        api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")

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

    def synthesize_judgment(self, odin_state: OdinState) -> str:
        """
        Step 1: Takes loki_interpretation and heimdall_review from OdinState.
        Uses Claude Haiku with TYR_JUDGMENT_PROMPT to generate final_judgment.
        """
        query = odin_state.query
        loki_text = odin_state.loki_interpretation or "No Loki interpretation available."
        heimdall_text = odin_state.heimdall_review or "No Heimdall review available."

        print(f"\n[⚖️ TYR] Step 1: Synthesizing Final Judgment with Claude Haiku...")
        print(f"       - Input 1: loki_interpretation ({len(loki_text)} chars)")
        print(f"       - Input 2: heimdall_review ({len(heimdall_text)} chars)")

        if self.use_mock or self.llm is None:
            judgment = (
                f"[Tyr Final Judgment ({self.model_name})]:\n"
                f"Having weighed the adversarial legal exposures presented by Loki:\n"
                f"{loki_text[:200]}...\n\n"
                f"Against the factual incident log and operational records surfaced by Heimdall:\n"
                f"{heimdall_text[:200]}...\n\n"
                f"Judgment on Query '{query}': Quillhub's position is evaluated by balancing contractual "
                f"SLA liability against proven external infrastructure outage logs."
            )
            odin_state.update_final_judgment(judgment)
            return judgment

        try:
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                ("system", TYR_JUDGMENT_PROMPT),
                ("human", "Core Legal & Operational Query: {query}\n\n"
                          "=== LOKI'S ADVERSARIAL LEGAL INTERPRETATION ===\n{loki_input}\n\n"
                          "=== HEIMDALL'S OPERATIONAL EVENT REVIEW ===\n{heimdall_input}\n\n"
                          "Please synthesize these two findings and render your supreme judgment.")
            ])
            chain = prompt | self.llm
            resp = chain.invoke({
                "query": query,
                "loki_input": loki_text,
                "heimdall_input": heimdall_text
            })
            judgment = resp.content.strip()
        except Exception as e:
            judgment = f"[Tyr Judgment Fallback]: Synthesized findings for '{query}'. (Error: {e})"

        odin_state.update_final_judgment(judgment)
        print("       -> Successfully stored 'final_judgment' in OdinState!")
        return judgment

    def derive_final_answer(self, odin_state: OdinState) -> str:
        """
        Step 2: Takes final_judgment from OdinState.
        Uses Claude Haiku with TYR_FINAL_ANSWER_PROMPT to generate final_ans.
        """
        query = odin_state.query
        judgment = odin_state.final_judgment or "No judgment available."

        print(f"\n[⚖️ TYR] Step 2: Deriving Definitive Final Answer from Judgment...")

        if self.use_mock or self.llm is None:
            ans = (
                f"[Tyr Final Answer ({self.model_name})]:\n"
                f"Definitive resolution for query '{query}': Based on Tyr's judgment, "
                f"the recommended strategic action is to reconcile the operational log against the SLA credit schedule."
            )
            odin_state.update_final_ans(ans)
            return ans

        try:
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                ("system", TYR_FINAL_ANSWER_PROMPT),
                ("human", "Core Legal & Operational Query: {query}\n\n"
                          "=== TYR'S FINAL JUDGMENT ===\n{judgment_input}\n\n"
                          "Based on the judgment above, provide the final answer and actionable resolution.")
            ])
            chain = prompt | self.llm
            resp = chain.invoke({
                "query": query,
                "judgment_input": judgment
            })
            ans = resp.content.strip()
        except Exception as e:
            ans = f"[Tyr Final Answer Fallback]: Resolution for '{query}'. (Error: {e})"

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
        auto_approve: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Full Execution Pipeline for Agent Tyr:
        1. Synthesize final_judgment from loki_interpretation and heimdall_review.
        2. Derive final_ans from final_judgment.
        3. Request human approval and update common state.
        """
        judgment = self.synthesize_judgment(odin_state)
        ans = self.derive_final_answer(odin_state)
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

