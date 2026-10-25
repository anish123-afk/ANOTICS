"""
Tests for Loki: Adversarial Contract-Risk Reviewer Agent (odin_loki.py).
Ensures 100% offline execution without external API dependencies.
"""

from datetime import date
from pathlib import Path
import pytest

import odin_loki
from odin_loki import (
    RiskCategory,
    FindingType,
    RiskSeverity,
    ConflictAnalyzer,
    SupersessionAnalyzer,
    FinancialGapAnalyzer,
    TimelineAnalyzer,
    EventDrivenMonitor,
    load_events,
    load_holidays,
    add_business_days,
    run_loki,
)


@pytest.fixture
def bundle_path():
    return Path("bundle")


def test_cross_contract_conflict_coexistence(bundle_path):
    """
    Test that B1 §6.1 (Retail Grocery) and C1 §4.1 (3PL) are correctly recognized
    as legitimately coexisting rather than falsely flagged as a territorial conflict.
    """
    conflicts = ConflictAnalyzer.analyze(bundle_path)
    exclusivity_risk = next((c for c in conflicts if c.risk_id == "RISK-CONFLICT-001"), None)

    assert exclusivity_risk is not None
    assert exclusivity_risk.category == RiskCategory.CROSS_CONTRACT_CONFLICT
    assert exclusivity_risk.severity == RiskSeverity.INFORMATIONAL
    assert exclusivity_risk.finding_type == FindingType.CONFIRMED_FACT
    assert "retail grocery" in exclusivity_risk.summary.lower()
    assert "third-party logistics" in exclusivity_risk.summary.lower()
    assert len(exclusivity_risk.verbatim_citations) == 2
    for cit in exclusivity_risk.verbatim_citations:
        assert cit.verified is True


def test_sla_target_misalignment(bundle_path):
    """
    Test that B1 99.9% uptime target vs A2 99.5% host commitment is flagged
    as a high-severity operational alignment risk.
    """
    conflicts = ConflictAnalyzer.analyze(bundle_path)
    sla_risk = next((c for c in conflicts if c.risk_id == "RISK-CONFLICT-002"), None)

    assert sla_risk is not None
    assert sla_risk.severity == RiskSeverity.HIGH
    assert "99.9%" in sla_risk.summary
    assert "99.5%" in sla_risk.summary
    assert len(sla_risk.verbatim_citations) == 2
    for cit in sla_risk.verbatim_citations:
        assert cit.verified is True


def test_amendment_supersession_notice_window(bundle_path):
    """
    Test that Amendment No. 1 (A3) superseding A2 (30d -> 15d claim deadline)
    and A1 (90d -> 60d non-renewal notice) is properly captured.
    """
    supersessions = SupersessionAnalyzer.analyze(bundle_path)
    assert len(supersessions) >= 2

    credit_super = next((s for s in supersessions if s.risk_id == "RISK-SUPER-001"), None)
    assert credit_super is not None
    assert credit_super.category == RiskCategory.AMENDMENT_SUPERSESSION
    assert "15 days" in credit_super.summary
    assert "30 days" in credit_super.summary

    renew_super = next((s for s in supersessions if s.risk_id == "RISK-SUPER-002"), None)
    assert renew_super is not None
    assert "60 days" in renew_super.summary
    assert "90 days" in renew_super.summary


def test_financial_recovery_gap_march(bundle_path):
    """
    Test March 2027 downtime calculation (190 min downtime = 99.57% uptime).
    Customer credit = $6,000 (below 99.9%).
    Upstream credit = $0 (above 99.5%).
    Net unrecoverable gap = $6,000.
    """
    gaps = FinancialGapAnalyzer.analyze(bundle_path)
    march_gap = next((g for g in gaps if g.risk_id == "RISK-FIN-001"), None)

    assert march_gap is not None
    assert march_gap.category == RiskCategory.FINANCIAL_RECOVERY_GAP
    assert march_gap.finding_type == FindingType.DERIVED_CALCULATION
    assert "$6,000.00" in march_gap.financial_impact
    assert "99.57%" in march_gap.summary
    assert march_gap.severity == RiskSeverity.HIGH


def test_financial_recovery_gap_may(bundle_path):
    """
    Test May 2027 downtime calculation (505 min downtime = 98.87% uptime).
    Customer credit = $6,000.
    Upstream credit = 25% base + 5% bonus = 30% of $8k = $2,400.
    Net gap = $3,600.
    """
    gaps = FinancialGapAnalyzer.analyze(bundle_path)
    may_gap = next((g for g in gaps if g.risk_id == "RISK-FIN-002"), None)

    assert may_gap is not None
    assert "$3,600.00" in may_gap.financial_impact
    assert "98.87%" in may_gap.summary
    assert "$2,400.00" in may_gap.summary


def test_notice_window_asymmetry_unconfirmed_recovery(bundle_path):
    """
    Test that upstream credit recovery is categorized as conditional/unconfirmed
    rather than declaring that the credit was lost without evidence.
    """
    gaps = FinancialGapAnalyzer.analyze(bundle_path)
    notice_risk = next((g for g in gaps if g.risk_id == "RISK-FIN-003"), None)

    assert notice_risk is not None
    assert notice_risk.finding_type == FindingType.HYPOTHESIS_OR_CONDITIONAL
    assert "45 days" in notice_risk.summary
    assert "15 days" in notice_risk.summary
    assert "conditional" in notice_risk.summary.lower()


def test_holiday_calendar_and_deemed_acceptance(bundle_path):
    """
    Test business day addition with observed holidays and D1 deemed acceptance
    for Milestone M3 (10 business days from Feb 28, 2027 = March 12, 2027).
    """
    holidays = load_holidays(bundle_path / "data" / "holidays.csv")
    assert len(holidays) > 0
    # Test Thanksgiving 2026 is in holidays
    assert date(2026, 11, 28) in holidays

    # Test adding 10 business days from 2027-02-28 (Sunday)
    deadline = add_business_days(date(2027, 2, 28), 10, holidays)
    assert deadline == date(2027, 3, 12)

    timelines = TimelineAnalyzer.analyze(bundle_path)
    m3_finding = next((t for t in timelines if t.risk_id == "RISK-TIME-001"), None)
    assert m3_finding is not None
    assert "deemed accepted" in m3_finding.summary.lower()


def test_informal_phone_call_challenge(bundle_path):
    """
    Test that Elena's phone call (E-10) is challenged as lacking contractual force
    under D1 §10.2 and §10.6.
    """
    timelines = TimelineAnalyzer.analyze(bundle_path)
    m4_finding = next((t for t in timelines if t.risk_id == "RISK-TIME-002"), None)

    assert m4_finding is not None
    assert "informal phone" in m4_finding.summary.lower()
    assert m4_finding.finding_type == FindingType.CONFIRMED_FACT


def test_payments_ledger_overdue_challenge(bundle_path):
    """
    Test that the payments ledger line 21 (April 8 overdue flag) is challenged
    as legally incorrect against B1 Section C (45 days) and Invoice E-09 (due May 17).
    """
    timelines = TimelineAnalyzer.analyze(bundle_path)
    ledger_challenge = next((t for t in timelines if t.risk_id == "RISK-TIME-003"), None)

    assert ledger_challenge is not None
    assert ledger_challenge.category == RiskCategory.EVIDENCE_CHALLENGE
    assert "May 17, 2027" in ledger_challenge.summary
    assert "45-day" in ledger_challenge.summary


def test_consecutive_sla_miss_defense(bundle_path):
    """
    Test that March (99.57%) and May (98.87%) SLA misses are recognized as non-consecutive,
    refuting any claim that Harbor has an immediate right to terminate under B1 §8.1.
    """
    timelines = TimelineAnalyzer.analyze(bundle_path)
    defense_finding = next((t for t in timelines if t.risk_id == "RISK-TIME-004"), None)

    assert defense_finding is not None
    assert "non-consecutive" in defense_finding.summary.lower()
    assert defense_finding.severity == RiskSeverity.HIGH


def test_insurance_lapse_checkpoint_progression(bundle_path):
    """
    Test that insurance lapse is evaluated conditionally based on clock checkpoint date.
    At 2027-03-15: not lapsed.
    At 2027-09-15: lapsed (June 30 expiry) and marked CRITICAL.
    """
    early_risks = EventDrivenMonitor.evaluate_checkpoint("2027-03-15", bundle_path)
    assert not any(r.risk_id == "RISK-EVENT-002" for r in early_risks)

    late_risks = EventDrivenMonitor.evaluate_checkpoint("2027-09-15", bundle_path)
    ins_risk = next((r for r in late_risks if r.risk_id == "RISK-EVENT-002"), None)
    assert ins_risk is not None
    assert ins_risk.severity == RiskSeverity.CRITICAL
    assert "June 30, 2027" in ins_risk.summary


def test_event_jsonl_parser_resilience(bundle_path):
    """
    Test that load_events parses events_drop2.jsonl without throwing an exception
    despite line 8 being malformed JSON.
    """
    events, warnings = load_events(bundle_path / "events" / "events_drop2.jsonl")
    assert len(events) >= 7
    assert len(warnings) == 1
    assert "malformed" in warnings[0].lower()


def test_run_loki_full_pipeline(bundle_path):
    """
    Test end-to-end run_loki LangGraph pipeline execution and verify report schema,
    verifications, and zero LLM calls.
    """
    report = run_loki(bundle_path=bundle_path, clock_checkpoint="2027-09-15", event_drop="Drop 2")

    assert report is not None
    assert report.total_findings >= 10
    assert report.unverified_citations_count == 0  # 100% verified citations
    assert report.metadata["llm_calls_made"] == 0
    assert report.metadata["offline_mode"] is True
    assert len(report.challenges_raised) >= 3
    assert "CRITICAL" in report.summary_by_severity
    assert "HIGH" in report.summary_by_severity
