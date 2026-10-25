"""
Tests for Odin Phase 1: Document Ingestion and Citation System.
"""

import json
from pathlib import Path
import sqlite3
import pytest

import fitz
import odin_phase1


@pytest.fixture
def test_db(tmp_path):
    """Provides a temporary test database path."""
    return tmp_path / "test_odin.db"


def test_sha256_hashing(tmp_path):
    """Test SHA-256 calculation for file integrity."""
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Hello Odin Contract Intelligence", encoding="utf-8")
    
    hash_val = odin_phase1.compute_sha256(test_file)
    assert isinstance(hash_val, str)
    assert len(hash_val) == 64
    assert hash_val == odin_phase1.compute_sha256(test_file)


def test_txt_and_pdf_extraction(tmp_path):
    """Test text and page extraction from TXT and PDF files."""
    txt_file = tmp_path / "contract.txt"
    txt_file.write_text("1.1 Term. This agreement shall run for one year.", encoding="utf-8")
    
    txt_res = odin_phase1.extract_text_from_file(txt_file)
    assert txt_res["status"] == "success"
    assert txt_res["page_count"] == 1
    assert "1.1 Term" in txt_res["pages"][0]["text"]
    assert len(txt_res["source_hash"]) == 64

    pdf_file = tmp_path / "contract.pdf"
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((50, 72), "SECTION 1 - SCOPE\n1.1 Services provided.")
    page2 = doc.new_page()
    page2.insert_text((50, 72), "SECTION 2 - PAYMENT\n2.1 Invoices are due in 30 days.")
    doc.save(pdf_file)
    doc.close()

    pdf_res = odin_phase1.extract_text_from_file(pdf_file)
    assert pdf_res["status"] == "success"
    assert pdf_res["page_count"] == 2
    assert "SECTION 1" in pdf_res["pages"][0]["text"]
    assert "SECTION 2" in pdf_res["pages"][1]["text"]


def test_clause_segmentation():
    """Test splitting text into clauses and preserving unassigned text."""
    pages = [
        {
            "page_number": 1,
            "text": "PREAMBLE\nThis agreement is made between Parties.\n\n"
                    "1.1 Definitions\nCustomer means the company.\n\n"
                    "1.2 Term\nTerm begins on start date.\n"
        }
    ]
    clauses = odin_phase1.split_text_into_clauses(
        pages=pages,
        document_id="TEST_DOC",
        source_file="contracts/test.txt",
        source_hash="dummy_hash",
    )

    assert len(clauses) == 3
    assert clauses[0]["clause_id"] == "PREAMBLE"
    assert "This agreement is made" in clauses[0]["source_text"]
    assert clauses[1]["clause_id"] == "1.1"
    assert "Customer means" in clauses[1]["source_text"]
    assert clauses[2]["clause_id"] == "1.2"
    assert "Term begins" in clauses[2]["source_text"]


def test_ocr_detection_on_empty_scanned_pdf(tmp_path):
    """Test detection of PDFs requiring OCR when no text is extractable."""
    empty_pdf = tmp_path / "scanned_doc.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(empty_pdf)
    doc.close()

    res = odin_phase1.extract_text_from_file(empty_pdf)
    assert res["status"] == "ocr_needed"
    assert "OCR" in res["error_message"]


def test_missing_files_and_failures(tmp_path, test_db):
    """Test graceful error handling for missing files."""
    missing_file = tmp_path / "does_not_exist.pdf"
    res = odin_phase1.extract_text_from_file(missing_file)
    assert res["status"] == "error"
    assert "File not found" in res["error_message"]

    valid_txt = tmp_path / "valid.txt"
    valid_txt.write_text("SECTION 1\n1.1 All good.", encoding="utf-8")
    
    manifest_file = tmp_path / "manifest.json"
    manifest_data = {
        "files": [
            {"path": "missing.txt", "title": "Missing Doc"},
            {"path": "valid.txt", "title": "Valid Doc"},
        ]
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    report = odin_phase1.ingest_documents(manifest_file, test_db, include_pdf_twins=False)
    assert report["processed_documents"] == 1
    assert report["failed_documents"] == 1


def test_cross_reference_detection():
    """Test detecting and resolving cross references."""
    clauses = [
        {
            "document_id": "A1",
            "clause_id": "4.1",
            "source_text": "Customer shall pay fees pursuant to Section 4.2 and Section 99.0.",
            "source_file": "A1.txt",
        }
    ]
    known_clauses = {
        "A1": {"4.1", "4.2"}
    }
    refs = odin_phase1.detect_cross_references(clauses, all_clause_ids=known_clauses)
    assert len(refs) == 2
    
    ref_4_2 = next(r for r in refs if r["referenced_clause_id"] == "4.2")
    assert ref_4_2["resolved"] == 1
    assert ref_4_2["target_document_id"] == "A1"

    ref_99 = next(r for r in refs if r["referenced_clause_id"] == "99.0")
    assert ref_99["resolved"] == 0
    assert ref_99["target_document_id"] is None


def test_citation_retrieval_and_verification(test_db):
    """Test retrieving and verifying clause citations against real bundle files."""
    manifest_path = Path("bundle/manifest.json")
    assert manifest_path.exists(), "Manifest file should exist"

    report = odin_phase1.ingest_documents(manifest_path, test_db)
    assert report["processed_documents"] > 0

    citation_txt = odin_phase1.get_clause_citation("STD", "1.1", test_db, source_file_type="txt")
    assert citation_txt is not None
    assert citation_txt["clause_id"] == "1.1"
    assert "Access and Use" in citation_txt["source_text"]
    assert citation_txt["source_hash"] != ""

    verified, msg = odin_phase1.verify_citation(citation_txt)
    assert verified is True
    assert "verified successfully" in msg.lower()

    citation_pdf = odin_phase1.get_clause_citation("STD", "1.1", test_db, source_file_type="pdf")
    assert citation_pdf is not None
    assert citation_pdf["page_number"] == 1
    verified_pdf, msg_pdf = odin_phase1.verify_citation(citation_pdf)
    assert verified_pdf is True

