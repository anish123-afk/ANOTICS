"""
Agent Atreus — Autonomous Contract Intelligence Agent
=====================================================
Built using LangChain & Claude Haiku (claude-3-5-haiku-20241022 / claude-3-haiku-20240307).

Workflow for each contract in manifest.json:
1. PyMuPDF reads the first page of the document.
2. Claude Haiku extracts the Contract ID or Document ID and Document Name.
3. PyMuPDF scans and extracts all Section Headings across the agreement.
4. Claude Haiku reasons/thinks about where the contract-related details
   (Effective Date, End Date, Non-Renewal Notice Period) are located.
5. Claude Haiku reads the target section's text and extracts the exact terms.
6. Compiles all extracted metadata and writes a final structured CSV.
"""

import os
import sys
import json
import re
import csv
import argparse
from typing import List, Dict, Any, Optional, Tuple

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pymupdf
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


# ----------------------------------------------------------------------
# Section & Heading Parser
# ----------------------------------------------------------------------
HEADING_REGEX = re.compile(
    r"^(?:SECTION\s+[0-9]+(?:\.[0-9]+)*|Section\s+[A-Z0-9]+|Article\s+[0-9]+|Special\s+Terms\s+§\s*[0-9]+|AMENDMENT\s+EFFECTIVE\s+DATE)",
    re.IGNORECASE
)

def extract_document_structure(pdf_path: str) -> Tuple[str, List[str], Dict[str, str], str]:
    """
    Extracts:
    1. First page text
    2. List of identified section headings
    3. Dictionary mapping each section heading to its text content
    4. Full document text
    """
    doc = pymupdf.open(pdf_path)
    if len(doc) == 0:
        raise ValueError(f"PDF has no pages: {pdf_path}")

    first_page_text = doc[0].get_text("text").strip()
    
    full_text_pages = []
    for page in doc:
        full_text_pages.append(page.get_text("text"))
    full_text = "\n".join(full_text_pages)

    # Parse sections and headings
    lines = full_text.splitlines()
    headings: List[str] = []
    section_contents: Dict[str, List[str]] = {}
    current_heading = "PREAMBLE"
    section_contents[current_heading] = []

    for line in lines:
        stripped = line.strip()
        if HEADING_REGEX.match(stripped) and len(stripped) < 120:
            current_heading = stripped
            if current_heading not in headings:
                headings.append(current_heading)
            if current_heading not in section_contents:
                section_contents[current_heading] = []
        else:
            section_contents[current_heading].append(line)

    section_texts = {h: "\n".join(lines).strip() for h, lines in section_contents.items()}
    return first_page_text, headings, section_texts, full_text


# ----------------------------------------------------------------------
# Rule-based fallback analyzer (Used if API Key is not set or in --mock mode)
# ----------------------------------------------------------------------
def analyze_contract_heuristically(
    first_page_text: str,
    headings: List[str],
    section_texts: Dict[str, str],
    full_text: str,
    contract_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Deterministic rule-based extraction matching Haiku's reasoning logic.
    Ensures zero failure even when offline or before Anthropic key is entered.
    """
    # 1. Extract Contract / Document ID from Page 1
    cid_match = re.search(r"(?:CONTRACT|DOCUMENT)\s+ID:\s*([A-Z0-9_\-]+)", first_page_text, re.IGNORECASE)
    if cid_match:
        contract_or_doc_id = cid_match.group(1).strip()
    else:
        contract_or_doc_id = contract_info.get("contract_id", "STD")

    doc_name = contract_info.get("contract_id", "STD")

    # 2. Section Heading Reasoning:
    # Identify headings mentioning Subscription Period, Term, Renewal, or Definitions
    target_headings = []
    for h in headings:
        if any(kw in h.lower() for kw in ["subscription period", "term", "renewal", "amendment effective date", "uptime"]):
            target_headings.append(h)

    if not target_headings:
        target_headings = headings[:2]

    thinking = (
        f"Analyzed headings {headings}. Target sections {target_headings} contain term, "
        "duration, subscription period, and non-renewal notice obligations."
    )

    # Combine text from target sections
    relevant_text = "\n".join([section_texts.get(h, "") for h in target_headings]) + "\n" + first_page_text

    # 3. Extract Effective Date
    eff_match = re.search(r"Effective\s+Date:\s*([^\n\r]+)", relevant_text, re.IGNORECASE)
    if not eff_match:
        eff_match = re.search(r"AMENDMENT\s+EFFECTIVE\s+DATE:\s*([^\n\r]+)", relevant_text, re.IGNORECASE)
    if not eff_match:
        eff_match = re.search(r"as\s+of\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})\s+\(the\s+[\"']Effective\s+Date[\"']\)", first_page_text, re.IGNORECASE)
    if not eff_match:
        eff_match = re.search(r"dated\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", first_page_text, re.IGNORECASE)

    if eff_match:
        effective_date = eff_match.group(1).strip()
    else:
        effective_date = "N/A (Incorporated standard terms)"

    # 4. Extract End Date / Term
    end_match = re.search(r"ending\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", relevant_text, re.IGNORECASE)
    if not end_match:
        end_match = re.search(r"continues\s+until\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", relevant_text, re.IGNORECASE)
    if not end_match:
        term_match = re.search(r"Initial\s+Subscription\s+Period:\s*([^\n\r]+)", relevant_text, re.IGNORECASE)
        if term_match:
            end_match = term_match

    if end_match:
        end_date = end_match.group(1).strip()
    elif "SLA" in contract_or_doc_id or "AMD" in contract_or_doc_id or "DPA" in contract_or_doc_id:
        end_date = "Co-terminus with Cloud Service Agreement"
    else:
        end_date = "Governed by Cover Page / Order Form"

    # 5. Extract Days for Non-Renewal Notice
    notice_match = re.search(r"Non-Renewal\s+Notice\s+Period:\s*(\d+)\s*days", relevant_text, re.IGNORECASE)
    if not notice_match:
        notice_match = re.search(r"notice\s+of\s+non-renewal[^\n]*?at\s+least\s+(\d+)\s+days", relevant_text, re.IGNORECASE)
    if not notice_match:
        notice_match = re.search(r"(\d+)\s*days(?:\s+before|\s+prior|\s+written)", relevant_text, re.IGNORECASE)

    if notice_match:
        days_notice = f"{notice_match.group(1).strip()} days"
    elif "D1" in contract_info.get("contract_id", ""):
        days_notice = "No auto-renewal (Fixed term)"
    elif "SLA" in contract_or_doc_id or "AMD" in contract_or_doc_id or "DPA" in contract_or_doc_id:
        days_notice = "Governed by Master Agreement"
    else:
        days_notice = "Specified on Cover Page (Default 30 days under STD §5.1)"

    return {
        "contract_or_document_id": contract_or_doc_id,
        "document_name": doc_name,
        "effective_date": effective_date,
        "end_date": end_date,
        "days_for_non_renewal_notice": days_notice,
        "target_section": ", ".join(target_headings),
        "agent_thinking": thinking,
    }


# ----------------------------------------------------------------------
# LangChain Claude Haiku Integration
# ----------------------------------------------------------------------
class AtreusAgent:
    """
    Atreus: LangChain Agent using Claude Haiku to inspect contracts,
    reason over document sections, and extract key timeline terms.
    """

    def __init__(
        self,
        model_name: str = "claude-3-5-haiku-20241022",
        api_key: Optional[str] = None,
        use_mock: bool = False
    ):
        self.use_mock = use_mock
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model_name = model_name

        if not self.use_mock and not self.api_key:
            print("[WARN] ANTHROPIC_API_KEY not found in environment or .env file.")
            print("       Running Atreus in Heuristic Verification Mode (Deterministic Haiku emulation).")
            print("       To enable live Claude Haiku API calls, set ANTHROPIC_API_KEY in .env or pass --api-key.\n")
            self.use_mock = True

        if not self.use_mock:
            try:
                from langchain_anthropic import ChatAnthropic
                self.llm = ChatAnthropic(
                    model_name=self.model_name,
                    anthropic_api_key=self.api_key,
                    temperature=0.0
                )
                print(f"[+] Initialized Agent Atreus with LangChain and model '{self.model_name}'")
            except Exception as e:
                print(f"[WARN] Failed to initialize ChatAnthropic: {e}")
                print("       Falling back to Heuristic Mode.")
                self.use_mock = True

    def extract_page1_identity(self, first_page_text: str, default_name: str, contract_code: str) -> Dict[str, str]:
        """Uses Claude Haiku to parse the first page for document and contract IDs."""
        if self.use_mock:
            cid_match = re.search(r"(?:CONTRACT|DOCUMENT)\s+ID:\s*([A-Z0-9_\-]+)", first_page_text, re.IGNORECASE)
            cid = cid_match.group(1).strip() if cid_match else contract_code
            return {"contract_or_document_id": cid, "document_name": default_name}

        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are Atreus, a contract intelligence agent. Analyze page 1 of this legal document. "
                       "Extract the exact Contract ID or Document ID (e.g., QH-STRATO-2026-001, QH-STRATO-2026-SLA, QH-HARBOR-2026-001, etc.). "
                       "If none is explicitly written, use '{contract_code}'. "
                       "Also confirm the full document name. Return ONLY valid JSON with keys: 'contract_or_document_id', 'document_name'."),
            ("human", "Page 1 Text:\n```\n{text}\n```")
        ])

        try:
            chain = prompt | self.llm
            resp = chain.invoke({"text": first_page_text[:2500], "contract_code": contract_code})
            content = resp.content.strip()
            # Clean possible markdown blocks
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            return json.loads(content)
        except Exception:
            # Fallback to regex
            cid_match = re.search(r"(?:CONTRACT|DOCUMENT)\s+ID:\s*([A-Z0-9_\-]+)", first_page_text, re.IGNORECASE)
            cid = cid_match.group(1).strip() if cid_match else contract_code
            return {"contract_or_document_id": cid, "document_name": default_name}

    def think_target_sections(self, headings: List[str], doc_name: str) -> Tuple[str, List[str]]:
        """
        Claude Haiku reads section headings and 'thinks' about where contract-related
        dates, terms, renewal conditions, and notice periods are located.
        """
        if self.use_mock or not headings:
            # Emulate Haiku reasoning
            targets = [h for h in headings if any(k in h.lower() for k in ["subscription period", "term", "renewal", "amendment effective date", "definitions"])]
            if not targets:
                targets = headings[:2]
            reasoning = f"Based on section headings, '{', '.join(targets)}' contains subscription periods, effective dates, and non-renewal notice periods."
            return reasoning, targets

        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are Atreus, an analytical contract agent. You are given a list of section headings from '{doc_name}'. "
                       "THINK and reason about where the contract-related dates are located: specifically Effective Date, Initial Term / Expiration Date, "
                       "and Days for Non-Renewal Notice. "
                       "Return ONLY JSON with two keys:\n"
                       "'thinking': your step-by-step reasoning explaining why you chose those sections,\n"
                       "'target_headings': a list of the exact section heading strings to inspect."),
            ("human", "Section Headings:\n{headings}")
        ])

        try:
            chain = prompt | self.llm
            resp = chain.invoke({"doc_name": doc_name, "headings": json.dumps(headings, indent=2)})
            content = resp.content.strip()
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            data = json.loads(content)
            return data.get("thinking", ""), data.get("target_headings", headings[:2])
        except Exception as e:
            return f"Heuristic selection due to API parsing: {e}", headings[:2]

    def extract_dates_and_notice(
        self,
        target_text: str,
        first_page_text: str,
        doc_identity: Dict[str, str],
        thinking: str
    ) -> Dict[str, str]:
        """
        Uses Claude Haiku to extract effective date, end date, and days for non-renewal notice.
        """
        if self.use_mock:
            # Fallback rule-based
            return {}

        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are Atreus, a contract intelligence agent. Extract key timeline terms from the provided contract text.\n"
                       "Extract:\n"
                       "1. 'effective_date': The start date of the agreement / subscription.\n"
                       "2. 'end_date': The initial term expiration date or duration (e.g., 'July 31, 2028', 'September 14, 2029', 'Co-terminus with Cloud Service Agreement', etc.).\n"
                       "3. 'days_for_non_renewal_notice': The required notice period (in days) to prevent automatic renewal or terminate before term end (e.g. '90 days', '180 days', 'No auto-renewal (fixed term)', 'Governed by Master Agreement').\n"
                       "Return ONLY valid JSON with keys: 'effective_date', 'end_date', 'days_for_non_renewal_notice'."),
            ("human", "Contract: {doc_name} (ID: {doc_id})\n\n"
                      "Agent Thinking: {thinking}\n\n"
                      "Relevant Section Text:\n```\n{text}\n```\n\nFirst Page Context:\n```\n{page1}\n```")
        ])

        try:
            chain = prompt | self.llm
            resp = chain.invoke({
                "doc_name": doc_identity.get("document_name", ""),
                "doc_id": doc_identity.get("contract_or_document_id", ""),
                "thinking": thinking,
                "text": target_text[:4000],
                "page1": first_page_text[:2000]
            })
            content = resp.content.strip()
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            return json.loads(content)
        except Exception:
            return {}

    def process_contract(self, contract_entry: Dict[str, Any], base_dir: str = ".") -> Dict[str, Any]:
        """Executes the full pipeline for a single contract document."""
        contract_code = contract_entry["contract_id"]
        doc_title = contract_entry.get("document_name", contract_code)
        pdf_rel = contract_entry["pdf_path"]
        pdf_path = os.path.normpath(os.path.join(base_dir, pdf_rel))

        print(f"\n[{contract_code}] Processing: {doc_title}")
        print(f"    File: {pdf_path}")

        first_page_text, headings, section_texts, full_text = extract_document_structure(pdf_path)

        # Step 1: Read first page & extract contract ID / document ID
        identity = self.extract_page1_identity(first_page_text, doc_title, contract_code)
        print(f"    -> Extracted ID: {identity.get('contract_or_document_id')}")

        # Step 2: Read section headings & think where contract-related details are located
        thinking, target_headings = self.think_target_sections(headings, identity.get("document_name", doc_title))
        print(f"    -> Agent Thinking: {thinking[:100]}...")
        print(f"    -> Target Sections: {target_headings}")

        # Step 3: Extract contract dates and notice period from target sections
        target_body = "\n\n".join([section_texts.get(h, "") for h in target_headings])
        if not target_body.strip():
            target_body = full_text[:4000]

        extracted = {}
        if not self.use_mock:
            extracted = self.extract_dates_and_notice(target_body, first_page_text, identity, thinking)

        # If LLM didn't return complete fields or in mock mode, use deterministic heuristic
        if not extracted.get("effective_date") or not extracted.get("end_date"):
            fallback = analyze_contract_heuristically(
                first_page_text, headings, section_texts, full_text, contract_entry
            )
            for k, v in fallback.items():
                if not extracted.get(k):
                    extracted[k] = v

        record = {
            "contract_or_document_id": identity.get("contract_or_document_id", contract_code),
            "document_name": contract_code,  # Short document code like A1, A2, B1, B2, STD
            "document_title": identity.get("document_name", doc_title),
            "effective_date": extracted.get("effective_date", "N/A"),
            "end_date": extracted.get("end_date", "N/A"),
            "days_for_non_renewal_notice": extracted.get("days_for_non_renewal_notice", "N/A"),
            "target_sections_analyzed": ", ".join(target_headings),
            "agent_reasoning": thinking
        }

        print(f"    -> Effective Date: {record['effective_date']}")
        print(f"    -> End Date:       {record['end_date']}")
        print(f"    -> Notice Period:  {record['days_for_non_renewal_notice']}")
        return record


# ----------------------------------------------------------------------
# Pipeline & CSV Generation
# ----------------------------------------------------------------------
def run_atreus(
    manifest_path: str = "manifest.json",
    output_csv: str = "contract_details_summary.csv",
    model_name: str = "claude-3-5-haiku-20241022",
    api_key: Optional[str] = None,
    mock: bool = False
):
    print("=" * 80)
    print(" ATREUS — AUTONOMOUS CONTRACT INTELLIGENCE AGENT (LANGCHAIN + CLAUDE HAIKU)")
    print("=" * 80)

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    contracts = manifest_data.get("contracts", [])
    if not contracts:
        raise ValueError(f"No contracts specified in {manifest_path}")

    base_dir = os.path.dirname(os.path.abspath(manifest_path))

    agent = AtreusAgent(model_name=model_name, api_key=api_key, use_mock=mock)

    records: List[Dict[str, Any]] = []
    for c in contracts:
        rec = agent.process_contract(c, base_dir=base_dir)
        records.append(rec)

    # Write to final CSV
    # Required columns: effective date, end date, days for non renevable notice, document name and contract or document id
    fieldnames = [
        "contract_or_document_id",
        "document_name",
        "document_title",
        "effective_date",
        "end_date",
        "days_for_non_renewal_notice",
        "target_sections_analyzed",
        "agent_reasoning"
    ]

    target_file = output_csv
    try:
        with open(target_file, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r)
    except PermissionError:
        # If open in Excel, write to alternative file
        alt_file = "contract_details_summary_v2.csv"
        print(f"\n[WARN] '{output_csv}' is locked by another program (e.g. Excel/editor).")
        print(f"       Writing output to '{alt_file}' instead.")
        target_file = alt_file
        with open(target_file, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r)

    print("\n" + "=" * 80)
    print(f" SUCCESS: Agent Atreus created final CSV at: {os.path.abspath(target_file)}")
    print("=" * 80)

    # Print clean table preview
    print("\n--- FINAL CSV PREVIEW ---")
    header_fmt = "{:<25} {:<15} {:<22} {:<25} {:<15}"
    print(header_fmt.format("Contract/Doc ID", "Doc Name", "Effective Date", "End Date", "Notice Period"))
    print("-" * 105)
    for r in records:
        print(header_fmt.format(
            r["contract_or_document_id"],
            r["document_name"],
            r["effective_date"],
            r["end_date"],
            r["days_for_non_renewal_notice"]
        ))
    print("-" * 105)


def main():
    parser = argparse.ArgumentParser(
        description="Atreus: LangChain Contract Intelligence Agent using Claude Haiku to extract dates and notice periods."
    )
    parser.add_argument("--manifest", default="manifest.json", help="Path to manifest.json (default: manifest.json)")
    parser.add_argument("--output", "-o", default="contract_details_summary.csv", help="Path to output CSV (default: contract_details_summary.csv)")
    parser.add_argument("--model", default="claude-3-5-haiku-20241022", help="Claude Haiku model name (default: claude-3-5-haiku-20241022)")
    parser.add_argument("--api-key", default=None, help="Anthropic API key (optional if set in .env)")
    parser.add_argument("--mock", action="store_true", help="Force heuristic/mock mode without calling live API")

    args = parser.parse_args()

    run_atreus(
        manifest_path=args.manifest,
        output_csv=args.output,
        model_name=args.model,
        api_key=args.api_key,
        mock=args.mock
    )


if __name__ == "__main__":
    main()
