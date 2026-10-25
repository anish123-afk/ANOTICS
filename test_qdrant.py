"""
Tests for Qdrant Vector Database Integration.
"""

from pathlib import Path
import pytest
import odin_qdrant


def test_embedding_generation():
    """Test generating continuous semantic vector embeddings."""
    text = "The monthly subscription fee is $8,000 USD payable within 30 days."
    vec = odin_qdrant.get_text_embedding(text)
    assert isinstance(vec, list)
    assert len(vec) == odin_qdrant.VECTOR_DIM
    assert any(v != 0.0 for v in vec)

    # Empty text returns zero vector
    empty_vec = odin_qdrant.get_text_embedding("")
    assert len(empty_vec) == odin_qdrant.VECTOR_DIM
    assert all(v == 0.0 for v in empty_vec)


def test_qdrant_in_memory_upsert_and_vector_search():
    """Test Qdrant collection creation, clause upsert, and semantic vector similarity search."""
    client = odin_qdrant.get_qdrant_client(":memory:")
    odin_qdrant.init_qdrant_collections(client=client)

    sample_clauses = [
        {
            "document_id": "A1",
            "clause_id": "C",
            "heading": "Section C — Fees and Payment",
            "source_text": "Monthly Fee: $8,000.00 USD. Invoicing on first business day.",
            "page_number": 1,
            "source_file": "bundle/contracts/A1.txt",
            "source_hash": "dummyhashA1",
        },
        {
            "document_id": "A2",
            "clause_id": "2.1",
            "heading": "SECTION 2 — UPTIME COMMITMENT",
            "source_text": "Provider will make the Cloud Service available at least 99.5% Uptime.",
            "page_number": 1,
            "source_file": "bundle/contracts/A2.txt",
            "source_hash": "dummyhashA2",
        },
    ]

    count = odin_qdrant.upsert_clauses_to_qdrant(sample_clauses, client=client)
    assert count == 2

    # Query for payment terms using vector search
    results_payment = odin_qdrant.search_clauses_in_qdrant(
        query="monthly fee invoices payment amount",
        client=client,
        top_k=2,
    )
    assert len(results_payment) > 0
    top_hit = results_payment[0]["clause"]
    assert top_hit["document_id"] == "A1"
    assert top_hit["clause_id"] == "C"

    # Query for uptime commitment
    results_uptime = odin_qdrant.search_clauses_in_qdrant(
        query="uptime commitment availability service level percentage",
        client=client,
        top_k=2,
    )
    assert len(results_uptime) > 0
    top_hit_uptime = results_uptime[0]["clause"]
    assert top_hit_uptime["document_id"] == "A2"
    assert top_hit_uptime["clause_id"] == "2.1"


def test_qdrant_direct_clause_lookup():
    """Test retrieving exact clause payloads from Qdrant by document_id and clause_id."""
    client = odin_qdrant.get_qdrant_client(":memory:")
    odin_qdrant.init_qdrant_collections(client=client)

    clause = {
        "document_id": "B1",
        "clause_id": "D",
        "heading": "Section D — Technical Support",
        "source_text": "Support tier is Business Support 24x7.",
        "page_number": 2,
        "source_file": "bundle/contracts/B1.txt",
        "source_hash": "hashB1",
    }
    odin_qdrant.upsert_clauses_to_qdrant([clause], client=client)

    retrieved = odin_qdrant.get_clause_from_qdrant("B1", "D", client=client)
    assert retrieved is not None
    assert retrieved["document_id"] == "B1"
    assert retrieved["clause_id"] == "D"
    assert "Business Support" in retrieved["source_text"]

