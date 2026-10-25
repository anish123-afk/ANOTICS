"""
Tyr Adapter Module.

Bridges Loki (Adversarial Reviewer), Heimdall (Event Monitor), and Norns (Timeline & Exposure)
to Phase 9 Tyr (Arbitration & Action Drafting Agent).

Translates live agent outputs into Tyr's expected inputs:
- Loki's RiskFinding -> LokiFindingInput
- Heimdall's ContractAlert -> HeimdallEventInput
- Norns's CheckpointSnapshot -> DeterministicContext

Strictly non-LLM, deterministic mapping. Enforces safety controls:
drafting is NOT execution; human approval is required on all proposed actions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from odin_loki import LokiReviewReport, RiskFinding
from odin_heimdall import ContractAlert
from norns.non_llm.schemas import CheckpointSnapshot
from odin_tyr import (
    Evidence,
    LokiFindingInput,
    HeimdallEventInput,
    DeterministicContext,
    TyrAgentState,
    run_tyr,
)


def map_loki_to_tyr(
    findings_or_report: Union[LokiReviewReport, List[RiskFinding]]
) -> List[LokiFindingInput]:
    """
    Map Loki's verified RiskFindings into Tyr's LokiFindingInput schema.
    Converts verbatim citations into Tyr Evidence models and preserves assumptions.
    """
    if isinstance(findings_or_report, LokiReviewReport):
        findings_list = findings_or_report.findings
    else:
        findings_list = findings_or_report or []

    tyr_findings: List[LokiFindingInput] = []

    for f in findings_list:
        tyr_evidence: List[Evidence] = []
        for cit in f.verbatim_citations:
            tyr_evidence.append(
                Evidence(
                    document_id=cit.document_id,
                    clause_id=cit.clause_reference,
                    source_text=cit.verbatim_quote,
                    page_number=cit.page_number,
                    confidence=1.0 if cit.verified else 0.8,
                )
            )

        # Map severity
        sev = f.severity.value.lower() if hasattr(f.severity, "value") else str(f.severity).lower()
        if sev not in ["low", "medium", "high", "critical"]:
            sev = "medium"

        tyr_findings.append(
            LokiFindingInput(
                risk_id=f.risk_id,
                category=f.category.value if hasattr(f.category, "value") else str(f.category),
                description=f.summary,
                severity=sev,
                likelihood="high" if sev in ["critical", "high"] else "medium",
                supporting_evidence=tyr_evidence,
                assumptions=f.assumptions_stated or [],
                unresolved_questions=[],
            )
        )

    return tyr_findings


def map_heimdall_to_tyr(
    alerts: List[Union[ContractAlert, Dict[str, Any]]]
) -> List[HeimdallEventInput]:
    """
    Map Heimdall's ContractAlert records into Tyr's HeimdallEventInput schema.
    """
    tyr_events: List[HeimdallEventInput] = []

    for a in alerts:
        if isinstance(a, dict):
            event_id = a.get("event_id", "UNKNOWN")
            c_ids = a.get("affected_contract_ids", [])
            trigger = a.get("potential_trigger", "event_notification")
            sev = a.get("severity", "medium")
            summary = a.get("summary", "")
            req_rev = a.get("requires_review", True)
        else:
            event_id = a.event_id
            c_ids = a.affected_contract_ids
            trigger = a.potential_trigger
            sev = a.severity
            summary = a.summary
            req_rev = a.requires_review

        # Create basic evidence markers
        ev_list: List[Evidence] = [
            Evidence(
                document_id=cid,
                clause_id=trigger,
                source_text=summary,
                page_number=1,
                confidence=0.9,
            )
            for cid in c_ids
        ]

        tyr_events.append(
            HeimdallEventInput(
                event_id=event_id,
                affected_contract_ids=c_ids,
                potential_trigger=trigger,
                severity=sev.lower(),
                supporting_evidence=ev_list,
                event_payload={"summary": summary},
                requires_review=req_rev,
            )
        )

    return tyr_events


def build_deterministic_context_from_norns(
    snapshot: Optional[CheckpointSnapshot],
    checkpoint_date_str: str = "2027-03-15",
    superseded_clauses: Optional[Dict[str, str]] = None,
    norns_analysis: Optional[Any] = None,
) -> DeterministicContext:
    """
    Construct Tyr's DeterministicContext from Norns's CheckpointSnapshot calculations
    and Norns LLM synthesized legal recommendations.
    """
    confirmed = 0.0
    penalties = 0.0

    if snapshot:
        confirmed = snapshot.total_outstanding
        penalties = snapshot.total_penalties

    default_superseded = {
        "A2-Section 3.2": "A3-Section 1 (Credit request deadline 15 days)",
        "A1-Section B": "A3-Section 3 (Non-renewal notice 60 days)",
    }
    if superseded_clauses:
        default_superseded.update(superseded_clauses)

    deadlines = {
        "stratonode_credit_claim": "Within 15 days of month end (A3 §1)",
        "harbor_credit_claim": "Within 45 days of month end (B1 §3.5)",
        "insurance_expiration": "June 30, 2027 at 11:59 PM",
    }
    if norns_analysis and hasattr(norns_analysis, "recommended_immediate_actions"):
        for idx, act in enumerate(norns_analysis.recommended_immediate_actions, start=1):
            deadlines[f"norns_action_{idx}"] = act

    return DeterministicContext(
        simulated_date=checkpoint_date_str,
        confirmed_exposure_usd=round(confirmed, 2),
        potential_penalty_usd=round(penalties, 2),
        unrecoverable_gap_usd=6000.0 if checkpoint_date_str >= "2027-03-15" else 0.0,
        superseded_clauses=default_superseded,
        deadlines=deadlines,
    )


def run_tyr_on_live_outputs(
    loki_report: Optional[LokiReviewReport] = None,
    heimdall_alerts: Optional[List[ContractAlert]] = None,
    norns_snapshot: Optional[CheckpointSnapshot] = None,
    checkpoint_date: str = "2027-03-15",
    norns_analysis: Optional[Any] = None,
) -> TyrAgentState:
    """
    Execute Tyr arbitration on live outputs from Loki, Heimdall, and Norns.
    """
    tyr_loki = map_loki_to_tyr(loki_report) if loki_report else []
    tyr_heimdall = map_heimdall_to_tyr(heimdall_alerts) if heimdall_alerts else []
    det_ctx = build_deterministic_context_from_norns(
        norns_snapshot,
        checkpoint_date,
        norns_analysis=norns_analysis,
    )

    return run_tyr(
        loki_findings=tyr_loki,
        heimdall_events=tyr_heimdall,
        deterministic_context=det_ctx,
    )
