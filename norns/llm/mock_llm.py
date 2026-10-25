"""
Deterministic Mock LLM Engine for Norns.
Used for offline testing, CI/CD, and local verification when ANTHROPIC_API_KEY is not yet supplied.
Adheres strictly to the Pydantic schema and returns verifiable legal narratives from input data.
"""

from __future__ import annotations

import json
from typing import Any, Dict
from ..non_llm.schemas import CheckpointSnapshot
from .schemas import ObligationNarrative, TimelineCheckpointAnalysis


class MockNornsLLM:
    """
    Offline simulator for Norns LLM reasoning.
    Synthesizes compliant legal summaries and worst-case scenarios directly from snapshot data.
    """

    def analyze_snapshot(self, snapshot: CheckpointSnapshot) -> TimelineCheckpointAnalysis:
        date_str = snapshot.simulated_date
        alerts = []
        actions = []
        feedback = []

        overdue_count = snapshot.status_counts.get("overdue", 0)
        upcoming_count = snapshot.status_counts.get("upcoming", 0)

        exec_summary = (
            f"As of checkpoint {snapshot.checkpoint_index + 1} ({date_str}), Quillhub maintains "
            f"{snapshot.total_obligations} tracked obligations. Confirmed outstanding amount due is "
            f"${snapshot.total_outstanding:,.2f} with ${snapshot.total_penalties:,.2f} in potential late penalties. "
        )

        if overdue_count > 0:
            exec_summary += f"CRITICAL: {overdue_count} obligation(s) are currently OVERDUE and require immediate legal remedy."
        else:
            exec_summary += "All matured commitments are currently fulfilled or in good standing."

        for ob in snapshot.obligations:
            if ob.status.value in ("overdue", "upcoming", "due"):
                worst_case = ""
                legal_basis = f"Clause(s): {', '.join(ob.evidence_ids) if ob.evidence_ids else 'Contract baseline'}"

                if ob.status.value == "overdue":
                    worst_case = (
                        f"If default continues beyond contractual cure period, counterparty may declare formal breach, "
                        f"assess late fee of ${ob.potential_penalty:,.2f}, or terminate agreement for cause."
                    )
                    actions.append(f"Issue cure response / release payment for {ob.obligation_id} ({ob.contract_id})")
                elif ob.status.value == "upcoming":
                    worst_case = (
                        f"Failure to deliver formal notice prior to due date {ob.due_date} may cause automatic renewal "
                        f"or forfeiture of cancellation rights."
                    )
                    if "RENEWAL" in ob.obligation_id or "NOTICE" in ob.obligation_id:
                        actions.append(f"Prepare and serve non-renewal notice for {ob.obligation_id} before {ob.due_date}")

                alerts.append(
                    ObligationNarrative(
                        obligation_id=ob.obligation_id,
                        contract_id=ob.contract_id,
                        status=ob.status.value,
                        summary_explanation=(
                            f"{ob.description} is {ob.status.value.upper()}. "
                            f"Days remaining: {ob.days_remaining}d. Total paid to date: ${ob.total_paid:,.2f}."
                        ),
                        legal_basis=legal_basis,
                        cited_evidence_ids=ob.evidence_ids,
                        worst_case_scenario=worst_case,
                    )
                )

            if ob.status.value == "no_due_date":
                feedback.append(f"Request Thor re-extraction: Obligation {ob.obligation_id} has no determinable due date.")

        if not actions:
            actions.append("Maintain routine monitoring of upcoming contractual notice deadlines.")

        return TimelineCheckpointAnalysis(
            simulated_date=date_str,
            checkpoint_index=snapshot.checkpoint_index,
            executive_summary=exec_summary,
            total_confirmed_due=snapshot.total_outstanding,
            total_potential_penalties=snapshot.total_penalties,
            critical_alerts=alerts,
            recommended_immediate_actions=actions,
            missing_information_requests=feedback,
        )
