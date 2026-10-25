"""
End-to-End Integration Test Suite for ODIN Pipeline.

Tests:
1. Repeatable Phase 1 database initialization and population without data corruption.
2. Live Thor-to-Norns adapter with successful conversion and graceful handling of missing/malformed fields.
3. Live Heimdall checkpoint temporal slicing (preventing future event leakage).
4. Master orchestrator (run_odin_pipeline) executing Phases 1–5 and Tyr cleanly.
5. Idempotent re-execution producing consistent, predictable results.
6. Tyr arbitration and action drafting safety gates (human approval enforced).
"""

from datetime import date
from pathlib import Path
import pytest
import sqlite3

from odin_thor import ContractExtractionResult, ExtractedObligation, Evidence, PaymentSchedule, PenaltyClause
from odin_thor_norns_adapter import (
    adapt_thor_to_norns,
    parse_currency_amount,
    parse_date_safe,
    parse_rate_percent,
)
from odin_heimdall import process_events, parse_event_date
from run_odin import run_odin_pipeline


# =============================================================================
# 1. Thor-to-Norns Adapter Tests
# =============================================================================

def test_thor_to_norns_adapter_successful_mapping():
    """Test mapping a real-style Thor extraction result into typed ObligationItem models."""
    sample_thor = ContractExtractionResult(
        document_id="TEST-01",
        payment_schedule=PaymentSchedule(
            fee_amount="$15,000.00",
            billing_frequency="monthly",
            late_payment_interest="1.5%",
            evidence=[Evidence(document_id="TEST-01", clause_id="Sec-4", source_text="Fee is $15,000.00 per month")],
        ),
        obligations=[
            ExtractedObligation(
                description="Deliver alpha build by 2027-04-15 with $5,000 milestone payment",
                responsible_party="Contractor Inc.",
                due_date="2027-04-15",
                evidence=[Evidence(document_id="TEST-01", clause_id="Sec-3", source_text="Due 2027-04-15")],
            ),
        ],
        penalty_clauses=[
            PenaltyClause(
                penalty_type="Late Fee",
                description="Late payments incur 1.5% penalty",
                rate_or_amount="1.5%",
            )
        ],
    )

    obligations = adapt_thor_to_norns(sample_thor)
    assert len(obligations) == 2  # 1 recurring fee + 1 milestone obligation

    fee_ob = next((o for o in obligations if o.id == "OB-TEST-01-FEE"), None)
    assert fee_ob is not None
    assert fee_ob.contract_amount == 15000.0
    assert fee_ob.penalty_rate_percent == 1.5
    assert fee_ob.due_date is None

    milestone_ob = next((o for o in obligations if o.id == "OB-TEST-01-1"), None)
    assert milestone_ob is not None
    assert milestone_ob.due_date == date(2027, 4, 15)
    assert milestone_ob.contract_amount == 5000.0
    assert milestone_ob.penalty_rate_percent == 1.5
    assert milestone_ob.responsible_party == "Contractor Inc."
    assert "TEST-01:Sec-3" in milestone_ob.evidence_ids


def test_thor_to_norns_adapter_missing_and_malformed_fields():
    """Test that missing or malformed dates and fees default cleanly without inventing data."""
    sample_sparse = ContractExtractionResult(
        document_id="SPARSE-01",
        obligations=[
            ExtractedObligation(
                description="General obligation without explicit date or amount",
                responsible_party=None,
                due_date="unspecified or ambiguous date string",
            )
        ],
    )

    obligations = adapt_thor_to_norns(sample_sparse)
    assert len(obligations) == 1
    ob = obligations[0]
    assert ob.due_date is None  # Does not invent a date
    assert ob.contract_amount == 0.0  # Does not invent an amount
    assert ob.responsible_party is None
    assert ob.penalty_rate_percent == 0.0


def test_thor_to_norns_adapter_empty_input():
    """Test that empty or None inputs return an empty list gracefully."""
    assert adapt_thor_to_norns([]) == []
    assert adapt_thor_to_norns(None) == []


def test_adapter_helper_parsers():
    """Test currency, date, and rate parser edge cases."""
    assert parse_currency_amount("$40,000.00") == 40000.0
    assert parse_currency_amount("Amount: $8,000") == 8000.0
    assert parse_currency_amount("No money mentioned") == 0.0
    assert parse_currency_amount(None) == 0.0

    assert parse_date_safe("2027-03-15") == date(2027, 3, 15)
    assert parse_date_safe("February 28, 2027") == date(2027, 2, 28)
    assert parse_date_safe("invalid-date-string") is None
    assert parse_date_safe(None) is None

    assert parse_rate_percent("1.5% per month") == 1.5
    assert parse_rate_percent("5.0%") == 5.0
    assert parse_rate_percent("no percentage") == 0.0


# =============================================================================
# 2. Heimdall Checkpoint Temporal Slicing Tests
# =============================================================================

def test_heimdall_checkpoint_temporal_filtering(tmp_path):
    """
    Test that Heimdall strictly respects checkpoint_date:
    Events dated after the checkpoint must NOT generate alerts or leak into earlier checkpoints.
    """
    db_file = tmp_path / "checkpoint_test.db"
    drop1 = Path("bundle/events/events_drop1.jsonl")
    drop2 = Path("bundle/events/events_drop2.jsonl")

    # Checkpoint 1: 2027-03-15
    res_cp1 = process_events([drop1, drop2], db_path=db_file, checkpoint_date="2027-03-15")
    # All generated alerts at Checkpoint 1 must be from events on or before March 15, 2027
    alert_event_ids_cp1 = {a["event_id"] for a in res_cp1["alerts"]}

    # Verify that later events (E-12 May outage, E-16 June insurance, E-17 June credit claim) are NOT in CP1
    assert "E-12" not in alert_event_ids_cp1
    assert "E-16" not in alert_event_ids_cp1
    assert "E-17" not in alert_event_ids_cp1

    # Checkpoint 3: 2027-09-15
    res_cp3 = process_events([drop1, drop2], db_path=db_file, checkpoint_date="2027-09-15")
    alert_event_ids_cp3 = {a["event_id"] for a in res_cp3["alerts"]}
    # In Checkpoint 3, E-16 and E-17 should be captured
    assert "E-16" in alert_event_ids_cp3
    assert "E-17" in alert_event_ids_cp3
    assert len(alert_event_ids_cp3) > len(alert_event_ids_cp1)


# =============================================================================
# 3. Master Orchestrator End-to-End Tests
# =============================================================================

def test_orchestrator_end_to_end_checkpoint_1(tmp_path):
    """Test full pipeline execution at Checkpoint 1 using a clean temporary database."""
    temp_db = tmp_path / "odin_cp1.db"

    result = run_odin_pipeline(
        checkpoint_date="2027-03-15",
        bundle_path="bundle",
        db_path=temp_db,
        include_tyr=True,
    )

    assert result["status"] == "success"
    assert result["agent_statuses"]["phase1"] == "success"
    assert result["agent_statuses"]["thor"] == "success"
    assert result["agent_statuses"]["norns"] == "success"
    assert result["agent_statuses"]["loki"] == "success"
    assert result["agent_statuses"]["heimdall"] == "success"
    assert result["agent_statuses"]["tyr"] == "success"

    # Verify Phase 1 database tables exist and are populated
    conn = sqlite3.connect(temp_db)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    assert "documents" in tables
    assert "clauses" in tables
    assert "cross_references" in tables
    assert "events_raw" in tables
    assert "contract_alerts" in tables

    # Verify row counts
    assert conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 17
    assert conn.execute("SELECT count(*) FROM clauses").fetchone()[0] >= 450
    assert conn.execute("SELECT count(*) FROM events_raw").fetchone()[0] >= 18
    conn.close()

    # Verify Tyr safety gate
    assert result["tyr"]["all_actions_require_human_approval"] is True


def test_orchestrator_idempotent_reexecution(tmp_path):
    """Test that running the orchestrator twice on the same database behaves predictably without duplicates."""
    temp_db = tmp_path / "idempotent.db"

    # First run
    run1 = run_odin_pipeline(checkpoint_date="2027-03-15", db_path=temp_db)
    # Second run
    run2 = run_odin_pipeline(checkpoint_date="2027-03-15", db_path=temp_db)

    assert run1["status"] == "success"
    assert run2["status"] == "success"
    assert run1["phase1"]["total_clauses"] == run2["phase1"]["total_clauses"]
    assert run1["norns"]["total_obligations_evaluated"] == run2["norns"]["total_obligations_evaluated"]
    assert run1["loki"]["total_findings"] == run2["loki"]["total_findings"]
    assert run1["heimdall"]["alerts_generated"] == run2["heimdall"]["alerts_generated"]


def test_orchestrator_handles_missing_manifest_explicitly(tmp_path):
    """Test that orchestrator returns explicit error instead of silent failure when manifest is missing."""
    temp_db = tmp_path / "missing.db"
    result = run_odin_pipeline(
        checkpoint_date="2027-03-15",
        bundle_path=tmp_path / "non_existent_bundle",
        db_path=temp_db,
    )

    assert result["status"] == "error"
    assert result["agent_statuses"]["phase1"] == "failed"
    assert len(result["warnings"]) > 0
    assert "Manifest not found" in result["warnings"][0]
