"""
Odin Phase 1: Document Ingestion and Citation System.

Provides simple, clean functions to:
1. Ingest contracts (.pdf and .txt) defined in manifest.json.
2. Compute SHA-256 hashes and extract text with PyMuPDF preserving page numbers.
3. Handle missing files, extraction failures, and detect scanned PDFs needing OCR.
4. Segment text into clauses without discarding preamble/unassigned text.
5. Record citation metadata (document_id, clause_id, source_text, page_number, source_hash).
6. Detect cross-references (e.g. Section 4.2, Clause 7.1) and resolve them against known clauses.
7. Persist all records idempotently in SQLite.
8. Retrieve and verify citations against source files.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger("odin.ingestion")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Regular expression to match section/clause headings
CLAUSE_HEADING_REGEX = re.compile(
    r"^(?:(?:SECTION|Section|CLAUSE|Clause|ARTICLE|Article)\s+([0-9A-Z]+(?:\.[0-9A-Z]+)*)|([0-9]+\.[0-9]+(?:\.[0-9]+)*))(?:\s*[\:\—\-\–\.]|\s+|$)",
    re.IGNORECASE,
)

# Regular expression to match named structural blocks (e.g. Cover Page blocks, Exhibits, Schedules)
NAMED_BLOCK_REGEX = re.compile(
    r"^(PROVIDER\s+INFORMATION|CUSTOMER\s+INFORMATION|KEY\s+TERMS|SIGNATURES|EXHIBIT\s+[A-Z0-9]+|SCHEDULE\s+[A-Z0-9]+)\b",
    re.IGNORECASE,
)


def detect_structural_heading(line: str, next_line: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """
    Structure-aware detector for contract document boundaries:
    1. Standard numbered/lettered clauses (e.g. 'SECTION 1', 'Section A', '1.1 Access and Use')
    2. Structural contract sections with underline formatting (e.g. 'PROVIDER INFORMATION', 'KEY TERMS')
    Returns:
        (clause_id, heading_text) or None
    """
    line_clean = line.strip()
    if not line_clean:
        return None

    # Check numbered/lettered clause regex
    match = CLAUSE_HEADING_REGEX.match(line_clean)
    if match:
        clause_id = match.group(1) or match.group(2)
        return clause_id, line_clean

    # Check structural blocks followed by Markdown/ASCII underlines (--- or ===)
    if next_line:
        nl_clean = next_line.strip()
        if len(nl_clean) >= 3 and set(nl_clean) <= {"-", "=", "_"}:
            m_block = NAMED_BLOCK_REGEX.match(line_clean)
            if m_block:
                return m_block.group(1).upper(), line_clean

    return None

# Regular expression to match cross references inside clause text
CROSS_REF_REGEX = re.compile(
    r"\b(?:Section|Clause|§)\s+([0-9A-Z]+(?:\.[0-9A-Z]+)*)\b",
    re.IGNORECASE,
)


def compute_sha256(file_path: Path | str) -> str:
    """Calculate the SHA-256 hash of a file."""
    path = Path(file_path)
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def init_database(db_path: Path | str = "odin.db") -> sqlite3.Connection:
    """
    Initialize SQLite database tables for documents, clauses, and cross-references.
    Ensures idempotent setup with appropriate constraints.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT NOT NULL,
                file_path TEXT NOT NULL UNIQUE,
                file_type TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                page_count INTEGER NOT NULL,
                ocr_needed INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                error_message TEXT,
                PRIMARY KEY (file_path)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS clauses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                clause_id TEXT NOT NULL,
                heading TEXT,
                source_text TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                source_file TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                UNIQUE(document_id, clause_id, page_number, source_file)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS cross_references (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                source_clause_id TEXT NOT NULL,
                referenced_clause_id TEXT NOT NULL,
                raw_match TEXT NOT NULL,
                resolved INTEGER NOT NULL,
                target_document_id TEXT,
                source_file TEXT NOT NULL,
                UNIQUE(document_id, source_clause_id, referenced_clause_id, raw_match, source_file)
            )
        """)
    return conn


def extract_text_from_file(file_path: Path | str) -> Dict[str, Any]:
    """
    Extract text and metadata from PDF or TXT files.
    Returns:
        {
            "status": "success" | "ocr_needed" | "error",
            "file_type": "pdf" | "txt",
            "source_hash": str,
            "page_count": int,
            "pages": List[Dict[str, Any]]  # [{"page_number": int, "text": str}],
            "error_message": Optional[str]
        }
    """
    path = Path(file_path)
    if not path.is_file():
        return {
            "status": "error",
            "file_type": path.suffix.lower().replace(".", ""),
            "source_hash": "",
            "page_count": 0,
            "pages": [],
            "error_message": f"File not found: {path}",
        }

    suffix = path.suffix.lower()
    source_hash = compute_sha256(path)

    if suffix == ".txt":
        try:
            content = path.read_text(encoding="utf-8")
            return {
                "status": "success",
                "file_type": "txt",
                "source_hash": source_hash,
                "page_count": 1,
                "pages": [{"page_number": 1, "text": content}],
                "error_message": None,
            }
        except Exception as e:
            logger.error("Failed to read TXT file %s: %s", path, e)
            return {
                "status": "error",
                "file_type": "txt",
                "source_hash": source_hash,
                "page_count": 0,
                "pages": [],
                "error_message": str(e),
            }

    elif suffix == ".pdf":
        try:
            doc = fitz.open(path)
            pages = []
            total_text_length = 0
            for idx, page in enumerate(doc, start=1):
                text = page.get_text()
                total_text_length += len(text.strip())
                pages.append({"page_number": idx, "text": text})
            # Check if PDF contains no extractable text (e.g. scanned image PDF)
            if total_text_length == 0:
                logger.warning("PDF contains no extractable text, marks OCR required: %s", path)
                return {
                    "status": "ocr_needed",
                    "file_type": "pdf",
                    "source_hash": source_hash,
                    "page_count": len(doc),
                    "pages": pages,
                    "error_message": "Document contains no extractable text and requires OCR.",
                }

            return {
                "status": "success",
                "file_type": "pdf",
                "source_hash": source_hash,
                "page_count": len(doc),
                "pages": pages,
                "error_message": None,
            }
        except Exception as e:
            logger.error("Failed to parse PDF file %s: %s", path, e)
            return {
                "status": "error",
                "file_type": "pdf",
                "source_hash": source_hash,
                "page_count": 0,
                "pages": [],
                "error_message": str(e),
            }

    else:
        return {
            "status": "error",
            "file_type": suffix.replace(".", ""),
            "source_hash": source_hash,
            "page_count": 0,
            "pages": [],
            "error_message": f"Unsupported file extension: {suffix}",
        }


def split_text_into_clauses(
    pages: List[Dict[str, Any]],
    document_id: str,
    source_file: str,
    source_hash: str,
) -> List[Dict[str, Any]]:
    """
    Structure-aware contract clause splitter.
    Segments document text into hierarchical clauses and structural blocks using:
    - Clause and section numbering ('SECTION 1', 'Section A', '1.1 Access and Use')
    - Formal agreement divisions ('PROVIDER INFORMATION', 'CUSTOMER INFORMATION', 'KEY TERMS')
    - Preserves all preambles, recitals, and boundary lines without discarding any text.
    """
    clauses: List[Dict[str, Any]] = []

    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"]
        lines = page_text.splitlines()

        current_clause_id: Optional[str] = None
        current_heading: Optional[str] = None
        current_lines: List[str] = []

        total_lines = len(lines)
        for idx, line in enumerate(lines):
            next_line = lines[idx + 1] if idx + 1 < total_lines else None
            header_match = detect_structural_heading(line, next_line)

            if header_match:
                clause_id_val, heading_val = header_match

                # Flush existing clause buffer
                if current_lines:
                    text_content = "\n".join(current_lines).strip()
                    if text_content:
                        clauses.append({
                            "document_id": document_id,
                            "clause_id": current_clause_id or "PREAMBLE",
                            "heading": current_heading,
                            "source_text": text_content,
                            "page_number": page_number,
                            "source_file": source_file,
                            "source_hash": source_hash,
                        })
                    current_lines = []

                current_clause_id = clause_id_val
                current_heading = heading_val
                current_lines.append(line)
            else:
                current_lines.append(line)

        # Flush any remaining buffer on this page
        if current_lines:
            text_content = "\n".join(current_lines).strip()
            if text_content:
                clauses.append({
                    "document_id": document_id,
                    "clause_id": current_clause_id or "PREAMBLE",
                    "heading": current_heading,
                    "source_text": text_content,
                    "page_number": page_number,
                    "source_file": source_file,
                    "source_hash": source_hash,
                })

    return clauses


def detect_cross_references(
    clauses: List[Dict[str, Any]],
    all_clause_ids: Optional[Dict[str, set[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Detect cross-references (e.g. 'Section 4.2', 'Clause 7.1') in clause text.
    Marks references as resolved if the referenced clause exists in known clauses.
    """
    all_clause_ids = all_clause_ids or {}
    refs: List[Dict[str, Any]] = []

    for clause in clauses:
        doc_id = clause["document_id"]
        source_clause = clause["clause_id"]
        text = clause["source_text"]

        for match in CROSS_REF_REGEX.finditer(text):
            target_id = match.group(1)
            raw_match = match.group(0)

            # Avoid trivial self-references if heading is repeated
            if target_id == source_clause and match.start() < 20:
                continue

            # Check resolution within current document or known documents
            is_resolved = False
            target_doc = None

            if doc_id in all_clause_ids and target_id in all_clause_ids[doc_id]:
                is_resolved = True
                target_doc = doc_id
            else:
                # Check if it exists in any other document (e.g., STD standard terms)
                for other_doc, other_clauses in all_clause_ids.items():
                    if target_id in other_clauses:
                        is_resolved = True
                        target_doc = other_doc
                        break

            refs.append({
                "document_id": doc_id,
                "source_clause_id": source_clause,
                "referenced_clause_id": target_id,
                "raw_match": raw_match,
                "resolved": 1 if is_resolved else 0,
                "target_document_id": target_doc,
                "source_file": clause["source_file"],
            })

    return refs


def ingest_documents(
    manifest_path: Path | str,
    db_path: Path | str = "odin.db",
    include_pdf_twins: bool = True,
) -> Dict[str, Any]:
    """
    Execute Phase 1 pipeline reading manifest.json and ingesting documents into SQLite.
    Continues processing even if an individual document encounters errors.
    """
    mpath = Path(manifest_path)
    bundle_dir = mpath.parent

    if not mpath.exists():
        raise FileNotFoundError(f"Manifest not found: {mpath}")

    with mpath.open("r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    conn = init_database(db_path)
    report = {
        "processed_documents": 0,
        "failed_documents": 0,
        "ocr_required": 0,
        "total_clauses": 0,
        "total_cross_references": 0,
        "details": [],
    }

    files_to_process: List[Tuple[str, Path]] = []
    for entry in manifest_data.get("files", []):
        rel_path = entry.get("path", "")
        p = bundle_dir / rel_path
        if p.suffix.lower() in [".txt", ".pdf"]:
            doc_id = p.stem
            files_to_process.append((doc_id, p))
            if include_pdf_twins and p.suffix.lower() == ".txt" and "contracts" in str(p):
                pdf_twin = p.with_suffix(".pdf")
                if pdf_twin.exists():
                    files_to_process.append((doc_id, pdf_twin))

    unique_files = {}
    for doc_id, file_path in files_to_process:
        unique_files[str(file_path.resolve())] = (doc_id, file_path)

    extracted_docs = []
    known_clause_map: Dict[str, set[str]] = {}

    for doc_id, file_path in unique_files.values():
        extract_result = extract_text_from_file(file_path)
        rel_file_str = str(file_path).replace("\\", "/")

        if extract_result["status"] == "error":
            logger.error("Extraction failed for %s: %s", file_path, extract_result["error_message"])
            report["failed_documents"] += 1
            report["details"].append({
                "document_id": doc_id,
                "file": rel_file_str,
                "status": "error",
                "message": extract_result["error_message"],
            })
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO documents 
                    (document_id, file_path, file_type, source_hash, page_count, ocr_needed, status, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (doc_id, rel_file_str, extract_result["file_type"], "", 0, 0, "error", extract_result["error_message"]),
                )
            continue

        if extract_result["status"] == "ocr_needed":
            report["ocr_required"] += 1
            report["details"].append({
                "document_id": doc_id,
                "file": rel_file_str,
                "status": "ocr_needed",
                "message": extract_result["error_message"],
            })
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO documents 
                    (document_id, file_path, file_type, source_hash, page_count, ocr_needed, status, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (doc_id, rel_file_str, extract_result["file_type"], extract_result["source_hash"],
                     extract_result["page_count"], 1, "ocr_needed", extract_result["error_message"]),
                )
            continue

        clauses = split_text_into_clauses(
            pages=extract_result["pages"],
            document_id=doc_id,
            source_file=rel_file_str,
            source_hash=extract_result["source_hash"],
        )

        if doc_id not in known_clause_map:
            known_clause_map[doc_id] = set()
        for cl in clauses:
            known_clause_map[doc_id].add(cl["clause_id"])

        extracted_docs.append({
            "doc_id": doc_id,
            "file_path": file_path,
            "rel_file_str": rel_file_str,
            "extract_result": extract_result,
            "clauses": clauses,
        })

    for item in extracted_docs:
        doc_id = item["doc_id"]
        rel_file_str = item["rel_file_str"]
        extract_result = item["extract_result"]
        clauses = item["clauses"]

        refs = detect_cross_references(clauses, all_clause_ids=known_clause_map)

        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents 
                (document_id, file_path, file_type, source_hash, page_count, ocr_needed, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    rel_file_str,
                    extract_result["file_type"],
                    extract_result["source_hash"],
                    extract_result["page_count"],
                    0,
                    "success",
                    None,
                ),
            )

            for cl in clauses:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO clauses
                    (document_id, clause_id, heading, source_text, page_number, source_file, source_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cl["document_id"],
                        cl["clause_id"],
                        cl["heading"],
                        cl["source_text"],
                        cl["page_number"],
                        cl["source_file"],
                        cl["source_hash"],
                    ),
                )

            for r in refs:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cross_references
                    (document_id, source_clause_id, referenced_clause_id, raw_match, resolved, target_document_id, source_file)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        r["document_id"],
                        r["source_clause_id"],
                        r["referenced_clause_id"],
                        r["raw_match"],
                        r["resolved"],
                        r["target_document_id"],
                        r["source_file"],
                    ),
                )

        report["processed_documents"] += 1
        report["total_clauses"] += len(clauses)
        report["total_cross_references"] += len(refs)
        report["details"].append({
            "document_id": doc_id,
            "file": rel_file_str,
            "status": "success",
            "page_count": extract_result["page_count"],
            "clauses_count": len(clauses),
            "cross_refs_count": len(refs),
        })

    conn.close()
    return report


def get_clause_citation(
    document_id: str,
    clause_id: str,
    db_path: Path | str = "odin.db",
    source_file_type: Optional[str] = "txt",
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a clause and citation record by document_id and clause_id.
    Optionally filter by source_file_type ('txt' or 'pdf').
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if source_file_type:
        cursor.execute(
            """
            SELECT document_id, clause_id, heading, source_text, page_number, source_file, source_hash
            FROM clauses
            WHERE document_id = ? AND clause_id = ? AND source_file LIKE ?
            ORDER BY page_number ASC
            LIMIT 1
            """,
            (document_id, clause_id, f"%.{source_file_type}"),
        )
    else:
        cursor.execute(
            """
            SELECT document_id, clause_id, heading, source_text, page_number, source_file, source_hash
            FROM clauses
            WHERE document_id = ? AND clause_id = ?
            ORDER BY page_number ASC
            LIMIT 1
            """,
            (document_id, clause_id),
        )

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "document_id": row["document_id"],
        "clause_id": row["clause_id"],
        "heading": row["heading"],
        "source_text": row["source_text"],
        "page_number": row["page_number"],
        "source_file": row["source_file"],
        "source_hash": row["source_hash"],
    }


def verify_citation(citation: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Verify that a clause citation's evidence genuinely exists in the stored source file:
    1. Check that source_file exists.
    2. Check that current file hash matches citation['source_hash'].
    3. Check that citation['source_text'] is present in the source file on the stated page.
    """
    file_path = Path(citation.get("source_file", ""))
    if not file_path.exists():
        return False, f"Source file does not exist: {file_path}"

    current_hash = compute_sha256(file_path)
    if current_hash != citation.get("source_hash"):
        return False, f"Hash mismatch: expected {citation.get('source_hash')} but found {current_hash}"

    page_number = citation.get("page_number", 1)
    source_text = citation.get("source_text", "").strip()

    if file_path.suffix.lower() == ".txt":
        full_text = file_path.read_text(encoding="utf-8")
        if source_text in full_text:
            return True, "Citation verified successfully against canonical text file."
        else:
            return False, "Citation text not found in source text file."

    elif file_path.suffix.lower() == ".pdf":
        try:
            doc = fitz.open(file_path)
            if page_number < 1 or page_number > len(doc):
                return False, f"Page number {page_number} out of range (total pages: {len(doc)})"
            page_text = doc[page_number - 1].get_text()
            normalized_source = re.sub(r"\s+", " ", source_text)
            normalized_page = re.sub(r"\s+", " ", page_text)
            if normalized_source in normalized_page:
                return True, "Citation verified successfully against PDF page."
            else:
                return False, f"Citation text not found on page {page_number} of PDF."
        except Exception as e:
            return False, f"Error verifying PDF page: {e}"

    return False, "Unsupported file format."

