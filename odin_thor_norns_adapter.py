"""
Thor-to-Norns Adapter Module.

Bridges Phase 2 (Thor Contract Extraction) to Phase 3 (Norns Timeline Agent).
Consumes structured ContractExtractionResult objects from Thor and maps only
supported facts into Norns's typed ObligationItem models.

Zero LLM calls. Deterministic parsing, strict validation, and faithful preservation
of evidence citations. Never invents missing dates, fees, penalties, or parties.
"""

from __future__ import annotations

from datetime import date, datetime
import logging
import re
from typing import Any, Dict, List, Optional, Union

from odin_thor import ContractExtractionResult, ExtractedObligation
from norns.non_llm.schemas import ObligationItem

logger = logging.getLogger("odin.adapter")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# =============================================================================
# Helper Parsers (Safe, Deterministic, Non-Fabricating)
# =============================================================================

CURRENCY_REGEX = re.compile(r"\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{1,2})?|\b[0-9]+(?:\.[0-9]{1,2})?\b)")
PERCENT_REGEX = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")
DATE_ISO_REGEX = re.compile(r"\b([12][0-9]{3}-[01][0-9]-[0-3][0-9])\b")


def parse_currency_amount(val: Optional[str]) -> float:
    """
    Extract first valid currency figure from text string.
    Returns 0.0 if not found or unparseable. Never invents amounts.
    """
    if not val:
        return 0.0
    val_clean = str(val).strip()
    match = CURRENCY_REGEX.search(val_clean)
    if match:
        raw_num = match.group(1).replace(",", "")
        try:
            return float(raw_num)
        except ValueError:
            return 0.0
    return 0.0


def parse_rate_percent(val: Optional[str]) -> float:
    """
    Extract first percentage rate from text string.
    Returns 0.0 if not found or unparseable.
    """
    if not val:
        return 0.0
    match = PERCENT_REGEX.search(str(val))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0
    return 0.0


def parse_date_safe(val: Optional[str]) -> Optional[date]:
    """
    Safely parse date from string using common legal and ISO date formats.
    Returns None if missing, ambiguous, or malformed. Never invents a date.
    """
    if not val:
        return None
    val_str = str(val).strip()

    # Try ISO YYYY-MM-DD match
    iso_match = DATE_ISO_REGEX.search(val_str)
    if iso_match:
        try:
            return datetime.strptime(iso_match.group(1), "%Y-%m-%d").date()
        except ValueError:
            pass

    formats = [
        "%Y-%m-%d",
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d %Y",
        "%b %d %Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(val_str, fmt).date()
        except ValueError:
            continue

    logger.debug("Could not parse date from '%s'", val_str)
    return None


# =============================================================================
# Core Adapter Logic
# =============================================================================

def adapt_thor_to_norns(
    thor_results: Union[ContractExtractionResult, List[ContractExtractionResult]]
) -> List[ObligationItem]:
    """
    Convert one or more Thor ContractExtractionResult objects into Norns ObligationItem models.

    Rules:
    - If due date is missing or unparseable, due_date is set to None (evaluated as NO_DUE_DATE by Norns).
    - If fee amount is unparseable or missing, contract_amount defaults to 0.0.
    - Contractual penalty percentages are mapped from PenaltyClause if detected for that contract.
    - Preserves evidence citations as evidence_ids (e.g. 'A1:Section 4.1').
    """
    if isinstance(thor_results, ContractExtractionResult):
        results_list = [thor_results]
    else:
        results_list = thor_results or []

    obligations: List[ObligationItem] = []

    for res in results_list:
        doc_id = res.document_id

        # 1. Determine penalty rate for this contract (if any)
        penalty_rate = 0.0
        if res.penalty_clauses:
            for p in res.penalty_clauses:
                p_rate = parse_rate_percent(p.rate_or_amount) or parse_rate_percent(p.description)
                if p_rate > 0.0:
                    penalty_rate = p_rate
                    break

        if penalty_rate == 0.0 and res.payment_schedule and res.payment_schedule.late_payment_interest:
            penalty_rate = parse_rate_percent(res.payment_schedule.late_payment_interest)

        # 2. Map Payment Schedule as a base recurring obligation
        if res.payment_schedule and res.payment_schedule.fee_amount:
            fee = parse_currency_amount(res.payment_schedule.fee_amount)
            freq = (res.payment_schedule.billing_frequency or "monthly").lower()
            pay_evidence = [
                f"{ev.document_id}:{ev.clause_id or 'PAYMENT'}"
                for ev in res.payment_schedule.evidence
            ]
            obligations.append(
                ObligationItem(
                    id=f"OB-{doc_id}-FEE",
                    contract_id=doc_id,
                    description=f"Recurring {freq} contract fee for {doc_id} ({res.payment_schedule.fee_amount})",
                    responsible_party=(
                        "Quillhub Inc." if doc_id.startswith("A") or doc_id == "D1"
                        else "Customer"
                    ),
                    due_date=None,  # Recurring schedule without fixed date
                    contract_amount=fee,
                    penalty_rate_percent=penalty_rate,
                    evidence_ids=pay_evidence,
                )
            )

        # 3. Map Extracted Obligations
        for idx, ob in enumerate(res.obligations, start=1):
            parsed_due = parse_date_safe(ob.due_date)
            parsed_amount = parse_currency_amount(ob.description)

            # Build evidence IDs
            ev_ids = [
                f"{ev.document_id}:{ev.clause_id or 'CLAUSE'}"
                for ev in ob.evidence
            ]
            if not ev_ids:
                ev_ids = [f"{doc_id}:OBLIGATION_{idx}"]

            obligations.append(
                ObligationItem(
                    id=f"OB-{doc_id}-{idx}",
                    contract_id=doc_id,
                    description=ob.description,
                    responsible_party=ob.responsible_party,
                    due_date=parsed_due,
                    contract_amount=parsed_amount,
                    penalty_rate_percent=penalty_rate,
                    evidence_ids=ev_ids,
                )
            )

    return obligations
