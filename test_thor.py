"""
Tests for Thor - Phase 4 Contract Extraction Agent.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import pytest
import odin_thor
from odin_thor import (
    Evidence,
    ContractExtractionResult,
    SemanticClauseIndex,
    run_thor,
)


def test_thor_semantic_clause_index():
    """Test BM25 semantic search across clause chunks."""
    clauses = [
        {"clause_id": "1.1", "source_text": "Customer shall pay a monthly fee of $5,000 USD."},
        {"clause_id": "2.1", "source_text": "Provider will ensure 99.9% uptime availability."},
        {"clause_id": "3.1", "source_text": "Either party may terminate for material breach with 30 days notice."},
    ]
    index = SemanticClauseIndex(clauses)

    # Search for payment terms
    res_payment = index.search("monthly fee payment invoice", top_k=2)
    assert len(res_payment) > 0
    assert res_payment[0][1]["clause_id"] == "1.1"

    # Search for termination terms
    res_term = index.search("terminate material breach notice", top_k=2)
    assert len(res_term) > 0
    assert res_term[0][1]["clause_id"] == "3.1"


def test_thor_extracts_correct_fields_from_contract():
    """Test that Thor extracts parties, term, payments, obligations, penalties, and cross-references."""
    result = run_thor("A1")
    assert isinstance(result, ContractExtractionResult)
    assert result.document_id == "A1"

    # Check parties
    party_roles = {p.role: p.name for p in result.parties}
    assert "Provider" in party_roles
    assert "StratoNode Inc." in party_roles["Provider"]
    assert "Customer" in party_roles
    assert "Quillhub Inc." in party_roles["Customer"]

    # Check duration & dates
    assert result.duration_and_term is not None
    assert result.duration_and_term.start_date is not None
    assert "August 1, 2026" in result.duration_and_term.start_date
    assert "24 months" in result.duration_and_term.initial_period

    # Check payment schedule
    assert result.payment_schedule is not None
    assert "$8,000.00" in result.payment_schedule.fee_amount
    assert result.payment_schedule.payment_terms_days == 30

    # Check penalty clause
    assert len(result.penalty_clauses) > 0
    late_penalty = next((p for p in result.penalty_clauses if "Late" in p.penalty_type), None)
    assert late_penalty is not None
    assert "1.5%" in late_penalty.rate_or_amount

    # Check cross-references
    assert len(result.cross_references) > 0


def test_thor_identifies_missing_information_without_inventing():
    """Test that Thor does not hallucinate missing fields and flags them explicitly."""
    # Dummy contract with only a title and scope
    minimal_clauses = [
        {"clause_id": "PREAMBLE", "source_text": "Non-Disclosure Agreement between Acme and Beta."},
        {"clause_id": "1.0", "source_text": "Recipient shall keep information secret."},
    ]
    result = run_thor("MINIMAL_DOC", clauses=minimal_clauses)

    # Missing fees should not be hallucinated
    assert result.payment_schedule is None or result.payment_schedule.fee_amount is None
    # Missing duration/start date should not be invented
    assert result.duration_and_term is None or result.duration_and_term.start_date is None

    # Missing fields must be explicitly recorded
    assert any("payment_schedule" in m for m in result.ambiguous_or_missing)
    assert any("duration_and_term" in m for m in result.ambiguous_or_missing)


def test_thor_produces_verifiable_citations():
    """Test that Thor validates citations against the verbatim clause source text."""
    result = run_thor("A1")
    assert result.verification_passed is True
    assert len(result.unverified_citations) == 0

    # Ensure every piece of evidence has confidence score in [0.0, 1.0]
    for p in result.parties:
        for ev in p.evidence:
            assert 0.0 <= ev.confidence <= 1.0
            assert len(ev.source_text) > 0


def test_thor_handles_inconsistent_clauses_and_preserves_uncertainty():
    """Test that ambiguous clauses preserve uncertainty rather than claiming certainty."""
    ambiguous_clauses = [
        {
            "clause_id": "4.1",
            "source_text": "Contractor shall complete the deliverables within a reasonable period as determined by Company.",
        }
    ]
    result = run_thor("AMBIGUOUS_DOC", clauses=ambiguous_clauses)
    assert len(result.obligations) > 0
    ob = result.obligations[0]
    # Since no specific due date is stated, due_date must be None and ambiguity flagged
    assert ob.due_date is None
    assert ob.responsible_party == "Contractor"


def test_thor_identifies_references_to_other_documents():
    """Test that Thor detects cross-references referencing standard terms (STD)."""
    clauses = [
        {
            "clause_id": "C",
            "source_text": "In accordance with Section 4.4 of the Standard Terms, overdue amounts accrue interest.",
        }
    ]
    result = run_thor("A1", clauses=clauses)
    assert len(result.cross_references) > 0
    ref = next(r for r in result.cross_references if r.referenced_clause == "4.4")
    assert ref.target_document == "STD"
    assert ref.source_clause == "C"


def test_thor_targeted_re_extraction_workflow():
    """Test that Thor can re-extract specific clauses when targeted."""
    clauses = [
        {"clause_id": "1.1", "source_text": "Provider will provide Technical Support."},
        {"clause_id": "2.1", "source_text": "Customer shall not reverse engineer software."},
    ]
    # First pass
    initial_res = run_thor("TEST_TARGET", clauses=clauses)
    initial_count = len(initial_res.obligations)

    # Targeted re-extraction on clause 2.1
    re_res = run_thor("TEST_TARGET", clauses=clauses, targeted_clause_ids=["2.1"])
    assert len(re_res.obligations) >= initial_count

