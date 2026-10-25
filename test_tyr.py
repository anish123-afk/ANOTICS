"""
Unit and Integration Tests for Phase 9: Tyr — Arbitration & Action Agent.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

try:
    import pytest
except ImportError:
    pytest = None

import odin_tyr
from odin_tyr import (
    ArbitrationOutcome,
    ActionType,
    ApprovalStatus,
    Evidence,
    LokiFindingInput,
    HeimdallEventInput,
    DeterministicContext,
    arbitrate_loki_finding,
    arbitrate_heimdall_event,
    run_tyr,
)


def test_tyr_arbitrate_loki_insufficient_evidence():
    """Verify that Loki findings without supporting contract evidence are ruled INSUFFICIENT_EVIDENCE."""
    finding = LokiFindingInput(
        risk_id="RISK-NO-EVID",
        category="vague_risk",
        description="Speculative risk with zero clause citations.",
        severity="high",
        likelihood="high",
        supporting_evidence=[],
    )
    det = DeterministicContext()
    decision = arbitrate_loki_finding(finding, det)

    assert decision.outcome == ArbitrationOutcome.INSUFFICIENT_EVIDENCE
    assert "no supporting contractual evidence" in decision.rationale.lower()
    assert decision.confidence < 0.5


def test_tyr_arbitrate_loki_amendment_supersession():
    """Verify that Loki findings referencing superseded clauses are ruled FINDING_REVISED."""
    finding = LokiFindingInput(
        risk_id="RISK-SUPERSEDED",
        category="outdated_clause",
        description="Old SLA credit term from original A2.",
        severity="medium",
        likelihood="medium",
        supporting_evidence=[
            Evidence(document_id="A2", clause_id="Section 4.1", source_text="Liability is capped at 1 month fee.")
        ],
    )
    det = DeterministicContext(
        superseded_clauses={"A2-Section 4.1": "A3-Section 1 (Service Level Amendment)"}
    )
    decision = arbitrate_loki_finding(finding, det)

    assert decision.outcome == ArbitrationOutcome.FINDING_REVISED
    assert len(decision.conflicts_detected) == 1
    assert "superseded by A3-Section 1" in decision.conflicts_detected[0]
    assert decision.revised_finding is not None


def test_tyr_arbitrate_loki_critical_requires_human_review():
    """Verify that critical severity risks or unresolved legal questions require human legal review."""
    finding = LokiFindingInput(
        risk_id="RISK-CRITICAL",
        category="indemnity_exposure",
        description="Customer indemnity exposure has ambiguous territorial limits.",
        severity="critical",
        likelihood="high",
        supporting_evidence=[
            Evidence(document_id="B1", clause_id="Section 11.1", source_text="Full indemnification clause.")
        ],
        unresolved_questions=["Does jurisdiction extend to EU retail operations?"],
    )
    det = DeterministicContext()
    decision = arbitrate_loki_finding(finding, det)

    assert decision.outcome == ArbitrationOutcome.HUMAN_REVIEW_REQUIRED
    assert "human legal counsel review" in decision.rationale.lower()


def test_tyr_arbitrate_heimdall_event_rejected_no_contract():
    """Verify that Heimdall events with no affected contracts are ruled FINDING_REJECTED."""
    event = HeimdallEventInput(
        event_id="EVT-ORPHAN",
        affected_contract_ids=[],
        potential_trigger="noise_ping",
        severity="low",
    )
    det = DeterministicContext()
    decision = arbitrate_heimdall_event(event, det)

    assert decision.outcome == ArbitrationOutcome.FINDING_REJECTED
    assert "does not link to any affected contract" in decision.rationale.lower()


def test_tyr_arbitrate_heimdall_event_supported_with_evidence():
    """Verify that valid high-severity SLA outage events are ruled FINDING_SUPPORTED."""
    event = HeimdallEventInput(
        event_id="EVT-SLA-OUTAGE",
        affected_contract_ids=["A1", "A2"],
        potential_trigger="service_level_failure",
        severity="high",
        supporting_evidence=[
            Evidence(document_id="A2", clause_id="Section 3.1", source_text="Uptime availability 99.5% minimum.")
        ],
    )
    det = DeterministicContext()
    decision = arbitrate_heimdall_event(event, det)

    assert decision.outcome == ArbitrationOutcome.FINDING_SUPPORTED
    assert "supported by evidence" in decision.rationale.lower()


def test_tyr_approval_controls_safety_boundary():
    """Verify that drafting is NOT execution and all actions strictly require human approval."""
    det = DeterministicContext(
        simulated_date="2027-03-15",
        unrecoverable_gap_usd=25000.0,
        deadlines={"cure_deadline": "Within 30 days"}
    )
    event = HeimdallEventInput(
        event_id="EVT-DOWNTIME",
        affected_contract_ids=["A2"],
        potential_trigger="service_level_failure",
        severity="high",
        supporting_evidence=[
            Evidence(document_id="A2", clause_id="Section 3.1", source_text="Outage threshold met.")
        ],
    )

    state = run_tyr(heimdall_events=[event], deterministic_context=det)

    assert len(state.proposed_actions) > 0
    for action in state.proposed_actions:
        assert action.requires_human_approval is True
        assert action.approval_status == ApprovalStatus.PENDING_APPROVAL
        assert "CEO Priya Chandrasekaran or VP Legal Marcus Ellison" in action.proposed_action_body


def test_tyr_drafts_breach_and_escalation_actions():
    """Verify that Tyr drafts both an executive escalation memo and a formal breach notice."""
    det = DeterministicContext(
        simulated_date="2027-03-15",
        unrecoverable_gap_usd=50000.0,
        deadlines={"cure_deadline": "April 15, 2027"}
    )
    finding = LokiFindingInput(
        risk_id="RISK-GAP",
        category="financial_recovery_gap",
        description="Downstream SLA payout exceeds upstream recovery.",
        severity="critical",
        likelihood="high",
        supporting_evidence=[
            Evidence(document_id="B1", clause_id="Section 8.2", source_text="Liquidated damages.")
        ],
    )

    state = run_tyr(loki_findings=[finding], deterministic_context=det)

    action_types = {act.action_type for act in state.proposed_actions}
    assert ActionType.ESCALATION_MEMORANDUM in action_types
    assert ActionType.AMENDMENT_PROPOSAL in action_types


if __name__ == "__main__":
    print("=" * 70)
    print("Running Tyr Unit & Integration Tests...")
    print("=" * 70)

    test_functions = [
        test_tyr_arbitrate_loki_insufficient_evidence,
        test_tyr_arbitrate_loki_amendment_supersession,
        test_tyr_arbitrate_loki_critical_requires_human_review,
        test_tyr_arbitrate_heimdall_event_rejected_no_contract,
        test_tyr_arbitrate_heimdall_event_supported_with_evidence,
        test_tyr_approval_controls_safety_boundary,
        test_tyr_drafts_breach_and_escalation_actions,
    ]

    passed = 0
    failed = 0
    for fn in test_functions:
        test_name = fn.__name__
        try:
            fn()
            print(f"  [PASS] {test_name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_name}: {e}")
            failed += 1

    print("=" * 70)
    print(f"Test Summary: {passed} passed, {failed} failed.")
    print("=" * 70)
    if failed > 0:
        sys.exit(1)

