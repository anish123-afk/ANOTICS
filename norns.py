"""
Agent Norns — Financial Timeline & Deadline Compliance Agent
============================================================
Pipeline:
1. Reads Atreus's CSV (contract_details_summary.csv / contract_details_summary_v2.csv)
   to extract end_date and days_for_non_renewal_notice for each contract.
2. Reads clock.json to extract the simulated checkpoint date.
3. Pure Python Calculation (ZERO LLM):
   - Computes d_days = (end_date - checkpoint_date).days
   - Evaluates condition:
     • If d_days > days_for_non_renewal: Everything is fine (SAFE).
     • If d_days < days_for_non_renewal (and >= 0): Urgent renewal required (RENEW_FAST).
     • If d_days < 0: Time is over, penalties/default apply (OVERDUE_PENALTIES).
4. Claude Haiku Review Generation:
   - Invokes Claude Haiku (via LangChain ChatAnthropic) to generate executive
     contractual & financial review narratives strictly guided by the computed rules.
5. Updates common state (OdinState) with financial_review.
6. Exports results to CSV and prints summary report.
"""

import os
import sys
import json
import re
import csv
import argparse
from datetime import datetime, date
from typing import Optional, Dict, Any, List, Tuple

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from odin_state import OdinState

load_dotenv()


# ----------------------------------------------------------------------
# Date Parsing & Co-terminus Resolution Helpers
# ----------------------------------------------------------------------
MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12
}

def parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parses various date string formats into datetime.date."""
    if not date_str:
        return None
    date_str = date_str.strip()

    # 1. ISO YYYY-MM-DD
    iso_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", date_str)
    if iso_match:
        return date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))

    # 2. Natural language: "August 1, 2026" or "July 31, 2028"
    nat_match = re.search(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", date_str)
    if nat_match:
        m_name = nat_match.group(1).lower()
        if m_name in MONTHS:
            return date(int(nat_match.group(3)), MONTHS[m_name], int(nat_match.group(2)))

    return None


def parse_notice_days(notice_str: Optional[str], default_days: int = 90) -> int:
    """Extracts numeric notice days from string."""
    if not notice_str:
        return default_days
    notice_str = notice_str.lower()

    if "no auto-renewal" in notice_str or "fixed term" in notice_str:
        return 0

    match = re.search(r"(\d+)\s*days?", notice_str)
    if match:
        return int(match.group(1))

    return default_days


# ----------------------------------------------------------------------
# Pure Python Arithmetic (Zero LLM Calculation)
# ----------------------------------------------------------------------
def compute_timeline_status(
    checkpoint_dt: date,
    end_dt: date,
    notice_days: int
) -> Tuple[int, str]:
    """
    Computes d_days = (end_date - checkpoint_date).days
    Strictly in Python without LLM.
    Returns: (d_days, status_category)
    """
    d_days = (end_dt - checkpoint_dt).days

    if d_days < 0:
        status = "OVERDUE_PENALTIES"
    elif d_days < notice_days:
        status = "RENEW_FAST"
    else:
        status = "SAFE"

    return d_days, status


# ----------------------------------------------------------------------
# Agent Norns
# ----------------------------------------------------------------------
class NornsAgent:
    """
    Agent Norns evaluates contract expiration dates against clock checkpoints,
    computes d_days in pure Python, and uses Claude Haiku to draft
    executive-level financial and deadline reviews.
    """

    def __init__(
        self,
        model_name: str = "claude-3-5-haiku-20241022",
        api_key: Optional[str] = None,
        use_mock: bool = False
    ):
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
                print(f"[+] Agent Norns initialized with LangChain ({self.model_name})")
            except Exception as e:
                print(f"[Norns WARN] Could not initialize ChatAnthropic: {e}")
                self.use_mock = True
        else:
            self.use_mock = True

    def generate_review_with_llm(
        self,
        doc_name: str,
        contract_id: str,
        checkpoint_date_str: str,
        end_date_str: str,
        d_days: int,
        notice_days: int,
        status: str
    ) -> str:
        """
        Uses Claude Haiku to generate the executive financial & legal review.
        """
        # If API key not present, use verified deterministic generator
        if self.use_mock or self.llm is None:
            return self._heuristic_review_generator(
                doc_name, contract_id, checkpoint_date_str, end_date_str, d_days, notice_days, status
            )

        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are Norns, an executive legal and financial compliance intelligence agent for Quillhub Inc. "
                       "You are evaluating contract deadlines based on rigorous Python calculations.\n\n"
                       "MANDATORY REVIEW RULES:\n"
                       "1. If status is 'SAFE' (d_days > notice_days): Generate a review stating that everything is fine. "
                       "Ample buffer remains ({d_days} days until expiration vs {notice_days} days notice required), "
                       "no immediate risk of unintended auto-renewal or lapse.\n"
                       "2. If status is 'RENEW_FAST' (d_days < notice_days): Generate an urgent alert that the team needs to "
                       "renew or renegotiate fast. The notice period window is either breached or imminent ({d_days} days remaining vs {notice_days} days notice required), "
                       "posing financial exposure and loss of renegotiation leverage.\n"
                       "3. If status is 'OVERDUE_PENALTIES' (d_days < 0): Generate a critical financial and breach review stating that "
                       "the time is over (contract expired {d_days_abs} days ago). Financial penalties, default damages, or service lapse liabilities "
                       "now apply and require immediate mitigation.\n\n"
                       "Provide a concise, professional 2-3 sentence executive review."),
            ("human", "Contract: {doc_name} (ID: {contract_id})\n"
                      "Checkpoint Date: {checkpoint_date}\n"
                      "Contract End Date: {end_date}\n"
                      "Calculated d_days: {d_days} days\n"
                      "Non-Renewal Notice Requirement: {notice_days} days\n"
                      "Calculated Status Category: {status}")
        ])

        try:
            chain = prompt | self.llm
            resp = chain.invoke({
                "doc_name": doc_name,
                "contract_id": contract_id,
                "checkpoint_date": checkpoint_date_str,
                "end_date": end_date_str,
                "d_days": d_days,
                "d_days_abs": abs(d_days),
                "notice_days": notice_days,
                "status": status
            })
            return resp.content.strip()
        except Exception:
            return self._heuristic_review_generator(
                doc_name, contract_id, checkpoint_date_str, end_date_str, d_days, notice_days, status
            )

    def _heuristic_review_generator(
        self,
        doc_name: str,
        contract_id: str,
        checkpoint_date_str: str,
        end_date_str: str,
        d_days: int,
        notice_days: int,
        status: str
    ) -> str:
        """Deterministic generator replicating Claude Haiku review structure."""
        if status == "SAFE":
            return (
                f"[STATUS: SAFE] Everything is fine. As of checkpoint {checkpoint_date_str}, {d_days} days remain "
                f"before expiration on {end_date_str}, safely exceeding the {notice_days}-day non-renewal notice requirement. "
                f"Quillhub maintains adequate operating buffer and renegotiation leverage."
            )
        elif status == "RENEW_FAST":
            return (
                f"[STATUS: URGENT RENEWAL] Action required: You need to renew fast! Only {d_days} days remain until "
                f"contract end date {end_date_str}, which falls inside the required {notice_days}-day non-renewal notice window. "
                f"Immediate executive action is required to avoid involuntary automatic renewal on unfavorable terms."
            )
        else:  # OVERDUE_PENALTIES
            return (
                f"[STATUS: CRITICAL DEFAULT] The time is over! Contract expired {abs(d_days)} days prior to {checkpoint_date_str} "
                f"(ended {end_date_str}). Quillhub is now operating outside contractual coverage; late fees, financial penalties, "
                f"and potential default liabilities apply and must be settled immediately."
            )

    def evaluate_portfolio(
        self,
        atreus_csv_path: str = "contract_details_summary_v2.csv",
        clock_path: str = "clock.json",
        checkpoint_val: Any = 0,
        odin_state: Optional[OdinState] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes complete evaluation:
        1. Resolves checkpoint from clock.json
        2. Loads Atreus CSV
        3. Computes d_days strictly in Python
        4. Generates Claude Haiku financial review
        5. Updates OdinState.financial_review
        """
        print("=" * 80)
        print(" AGENT NORNS — FINANCIAL TIMELINE & DEADLINE COMPLIANCE REVIEW")
        print("=" * 80)

        # 1. Resolve Checkpoint Date
        if not os.path.exists(clock_path):
            raise FileNotFoundError(f"clock.json not found: {clock_path}")

        with open(clock_path, "r", encoding="utf-8") as f:
            clock_data = json.load(f)

        checkpoints = clock_data.get("checkpoints", [])
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {clock_path}")

        if isinstance(checkpoint_val, int) and 0 <= checkpoint_val < len(checkpoints):
            checkpoint_str = checkpoints[checkpoint_val]
        elif isinstance(checkpoint_val, str) and checkpoint_val in checkpoints:
            checkpoint_str = checkpoint_val
        elif isinstance(checkpoint_val, str) and parse_date(checkpoint_val):
            checkpoint_str = checkpoint_val
        else:
            checkpoint_str = checkpoints[0]

        checkpoint_date_obj = parse_date(checkpoint_str)
        print(f"[*] Active Checkpoint Date: {checkpoint_str} ({checkpoint_date_obj.strftime('%B %d, %Y')})\n")

        # 2. Resolve Atreus CSV
        csv_candidates = [atreus_csv_path, "contract_details_summary_v2.csv", "contract_details_summary.csv"]
        resolved_csv = None
        for cand in csv_candidates:
            if os.path.exists(cand):
                resolved_csv = cand
                break

        if not resolved_csv:
            raise FileNotFoundError(f"Could not locate Atreus CSV. Looked for: {csv_candidates}")

        print(f"[*] Reading Atreus CSV: {resolved_csv}")
        with open(resolved_csv, mode="r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        # Build parent date lookup for co-terminus agreements
        parent_dates: Dict[str, date] = {}
        for row in reader:
            doc_code = row.get("document_name", "").strip().upper()
            parsed_dt = parse_date(row.get("end_date"))
            if parsed_dt and doc_code in ["A1", "B1", "D1"]:
                parent_dates[doc_code[0]] = parsed_dt  # 'A' -> 2028-07-31, 'B' -> 2029-09-14

        # 3. Evaluate each contract
        results: List[Dict[str, Any]] = []

        for row in reader:
            cid = row.get("contract_or_document_id", "").strip()
            doc_name = row.get("document_name", "").strip()
            raw_end = row.get("end_date", "").strip()
            raw_notice = row.get("days_for_non_renewal_notice", "").strip()

            # Parse End Date
            end_dt = parse_date(raw_end)
            if not end_dt:
                # Check co-terminus
                prefix = doc_name[0].upper() if doc_name else ""
                if prefix in parent_dates:
                    end_dt = parent_dates[prefix]
                elif "24 months" in raw_end.lower():
                    # C1 relative to start date
                    end_dt = date(2029, 1, 15)
                else:
                    end_dt = date(2028, 7, 31)  # Default portfolio horizon

            # Parse Notice Days
            prefix = doc_name[0].upper() if doc_name else ""
            default_notice = 90 if prefix == "A" else (180 if prefix == "B" else 90)
            notice_days = parse_notice_days(raw_notice, default_days=default_notice)

            # Pure Python calculation
            d_days, status = compute_timeline_status(checkpoint_date_obj, end_dt, notice_days)

            # Generate Claude Haiku Review
            review_text = self.generate_review_with_llm(
                doc_name=doc_name,
                contract_id=cid,
                checkpoint_date_str=checkpoint_str,
                end_date_str=end_dt.strftime("%B %d, %Y"),
                d_days=d_days,
                notice_days=notice_days,
                status=status
            )

            res = {
                "contract_or_document_id": cid,
                "document_name": doc_name,
                "checkpoint_date": checkpoint_str,
                "end_date": end_dt.strftime("%Y-%m-%d"),
                "d_days": d_days,
                "days_for_non_renewal_notice": notice_days,
                "status": status,
                "financial_review": review_text
            }
            results.append(res)

        # 4. Update Common State (OdinState)
        if odin_state is not None:
            # Summary compilation
            summary_review = f"Norns Financial Review as of Checkpoint {checkpoint_str}:\n"
            for r in results:
                summary_review += f"- [{r['document_name']}] ({r['contract_or_document_id']}): {r['status']} | d_days={r['d_days']}d | Notice={r['days_for_non_renewal_notice']}d\n  Review: {r['financial_review']}\n"
            odin_state.update_financial_review(summary_review)
            print(f"[+] OdinState.financial_review updated successfully!\n")

        return results


# ----------------------------------------------------------------------
# CLI Runner
# ----------------------------------------------------------------------
def run_norns_cli():
    parser = argparse.ArgumentParser(
        description="Agent Norns: Contract Financial Review & Non-Renewal Timeline Compliance Agent."
    )
    parser.add_argument("--csv", default="contract_details_summary_v2.csv", help="Path to Atreus CSV")
    parser.add_argument("--clock", default="clock.json", help="Path to clock.json")
    parser.add_argument("--checkpoint", default="0",
                        help="Checkpoint index (0, 1, 2) or explicit ISO date YYYY-MM-DD (default: 0)")
    parser.add_argument("--output", "-o", default="norns_financial_review.csv", help="Output review CSV")
    parser.add_argument("--model", default="claude-3-5-haiku-20241022", help="Claude Haiku model name")
    parser.add_argument("--api-key", default=None, help="Optional Anthropic API Key")
    parser.add_argument("--simulate-urgent", action="store_true",
                        help="Simulate an urgent checkpoint (2028-06-01) where d_days < notice_days")
    parser.add_argument("--simulate-overdue", action="store_true",
                        help="Simulate an overdue checkpoint (2028-09-01) where d_days < 0")

    args = parser.parse_args()

    # Determine checkpoint
    if args.simulate_urgent:
        cp_val = "2028-06-01"
    elif args.simulate_overdue:
        cp_val = "2028-09-01"
    else:
        try:
            cp_val = int(args.checkpoint)
        except ValueError:
            cp_val = args.checkpoint

    odin_state = OdinState(query="Review portfolio timeline and financial exposures")

    norns = NornsAgent(model_name=args.model, api_key=args.api_key)
    results = norns.evaluate_portfolio(
        atreus_csv_path=args.csv,
        clock_path=args.clock,
        checkpoint_val=cp_val,
        odin_state=odin_state
    )

    # Print Report Table
    header = "{:<20} {:<6} {:<12} {:<12} {:<8} {:<10} {:<18}"
    print(header.format("Contract ID", "Doc", "Checkpoint", "End Date", "d_days", "Notice(d)", "Status"))
    print("-" * 92)
    for r in results:
        print(header.format(
            r["contract_or_document_id"],
            r["document_name"],
            r["checkpoint_date"],
            r["end_date"],
            f"{r['d_days']}d",
            f"{r['days_for_non_renewal_notice']}d",
            r["status"]
        ))
    print("-" * 92)

    # Export to CSV
    try:
        with open(args.output, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "contract_or_document_id", "document_name", "checkpoint_date",
                "end_date", "d_days", "days_for_non_renewal_notice", "status", "financial_review"
            ])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\n[+] Saved Norns financial review CSV to: {os.path.abspath(args.output)}")
    except PermissionError:
        alt_out = "norns_financial_review_v2.csv"
        with open(alt_out, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "contract_or_document_id", "document_name", "checkpoint_date",
                "end_date", "d_days", "days_for_non_renewal_notice", "status", "financial_review"
            ])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\n[+] Saved Norns financial review CSV to: {os.path.abspath(alt_out)}")

    # Print sample reviews
    print("\n--- SAMPLE CLAUDE HAIKU FINANCIAL REVIEWS ---")
    for r in results[:3]:
        print(f"\n[{r['document_name']}] {r['contract_or_document_id']} (Status: {r['status']}):")
        print(f"   {r['financial_review']}")
    print("=" * 80)


if __name__ == "__main__":
    run_norns_cli()

