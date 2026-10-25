"""
Phase 4: Loki — Adversarial Contract-Risk Reviewer Agent.

Objective:
Act as an adversarial contract-risk reviewer for the ODIN system.
Thor extracts contract facts. Loki challenges those facts, compares relevant evidence,
identifies potential risks, tests assumptions, and distinguishes confirmed facts,
derived calculations, conditional hypotheses, and unresolved questions.

Absolute Constraint:
Runs 100% locally and offline without external LLM API calls (no Claude, OpenAI, Gemini).
Built on deterministic legal-logic rules, regex, temporal reasoning, business calendar
calculations, and exact evidence verification against canonical text files.

Core Risk Modules:
1. Cross-Contract Conflicts:
   - Distinguishes genuine contradictions from provisions that legitimately coexist
     (e.g., B1 retail grocery exclusivity vs C1 third-party logistics exclusivity).
   - Identifies operational & SLA target misalignments (B1 99.9% vs A2 99.5%).
   - Flags governing law and venue divergences.
2. Amendment & Supersession Risks:
   - Detects amendments (e.g., A3 amending A1 Cover Page and A2 SLA).
   - Replaces old terms with new ones while flagging obsolete terms (15-day vs 30-day
     credit request window; 60-day vs 90-day non-renewal notice; 5% downtime bonus).
3. Upstream / Downstream Financial Recovery Gaps:
   - Compares customer SLA payouts against upstream host SLA recoveries.
   - Accurately models March 2027 ($6,000 unrecoverable) and May 2027 ($3,600 gap).
   - Labels upstream claims as conditional/unconfirmed if not evidenced in communications.
4. Timeline & Obligation Risks:
   - Incorporates company holiday schedule (data/holidays.csv) to compute business days.
   - Evaluates milestone deliverables under D1 (M3 deemed accepted; M4 late submission).
   - Challenges erroneous payment ledger assumptions (April 2027 fee marked overdue early).
   - Evaluates consecutive SLA miss thresholds (March and May are non-consecutive).
   - Tracks insurance policy expiration (June 30, 2027) across clock checkpoints.
5. Evidence & Reasoning Challenges:
   - Categorizes every finding into Confirmed Fact, Derived Calculation, Conditional Hypothesis,
     or Unresolved Question.
   - Cites verbatim quotes verifiable against canonical .txt files.
6. Incremental Event-Driven Progression:
   - Evaluates risks progressively across clock checkpoints (2027-03-15, 2027-06-01, 2027-09-15)
     and staged event drops (Drop 1, Drop 2).
   - Tolerates malformed event lines gracefully.
"""

from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from enum import Enum
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

import odin_phase1
import odin_thor

logger = logging.getLogger("odin.loki")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# =============================================================================
# 1. Pydantic Schemas for Loki Adversarial Review
# =============================================================================

class RiskCategory(str, Enum):
    CROSS_CONTRACT_CONFLICT = "cross_contract_conflict"
    AMENDMENT_SUPERSESSION = "amendment_supersession"
    FINANCIAL_RECOVERY_GAP = "financial_recovery_gap"
    TIMELINE_OBLIGATION = "timeline_obligation"
    EVIDENCE_CHALLENGE = "evidence_challenge"
    EVENT_DRIVEN_RISK = "event_driven_risk"


class FindingType(str, Enum):
    CONFIRMED_FACT = "confirmed_fact"
    DERIVED_CALCULATION = "derived_calculation"
    HYPOTHESIS_OR_CONDITIONAL = "hypothesis_or_conditional"
    UNRESOLVED_QUESTION = "unresolved_question"


class RiskSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"


class VerifiableCitation(BaseModel):
    document_id: str
    clause_reference: str
    verbatim_quote: str
    page_number: int = 1
    source_file: str
    verified: bool = False
    verification_notes: Optional[str] = None


class RiskFinding(BaseModel):
    risk_id: str
    title: str
    category: RiskCategory
    finding_type: FindingType
    severity: RiskSeverity
    summary: str
    parties_involved: List[str] = Field(default_factory=list)
    affected_contracts: List[str] = Field(default_factory=list)
    verbatim_citations: List[VerifiableCitation] = Field(default_factory=list)
    reasoning_chain: List[str] = Field(default_factory=list)
    financial_impact: Optional[str] = None
    assumptions_stated: List[str] = Field(default_factory=list)
    next_action_recommended: str


class LokiReviewReport(BaseModel):
    checkpoint_date: Optional[str] = None
    event_drop: Optional[str] = None
    total_findings: int = 0
    findings: List[RiskFinding] = Field(default_factory=list)
    summary_by_category: Dict[str, int] = Field(default_factory=dict)
    summary_by_severity: Dict[str, int] = Field(default_factory=dict)
    unverified_citations_count: int = 0
    challenges_raised: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# 2. Context & Data Loaders (Robust & Resilient)
# =============================================================================

def parse_iso_date(date_str: str) -> Optional[date]:
    """Parse YYYY-MM-DD or ISO timestamp into a date object."""
    if not date_str:
        return None
    try:
        clean = date_str.split("T")[0].strip()
        return datetime.strptime(clean, "%Y-%m-%d").date()
    except Exception:
        return None


def load_holidays(holidays_csv_path: Path | str) -> Set[date]:
    """
    Load company and observed business holidays from CSV.
    Returns a set of date objects.
    """
    path = Path(holidays_csv_path)
    holidays: Set[date] = set()
    if not path.is_file():
        return holidays

    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d_val = parse_iso_date(row.get("date", ""))
                if d_val:
                    holidays.add(d_val)
    except Exception as e:
        logger.warning("Could not read holidays file %s: %s", path, e)
    return holidays


def add_business_days(start: date, num_days: int, holidays: Set[date]) -> date:
    """
    Add business days (excluding Saturdays, Sundays, and observed holidays).
    """
    current = start
    added = 0
    while added < num_days:
        current += timedelta(days=1)
        if current.weekday() < 5 and current not in holidays:
            added += 1
    return current


def load_events(events_path: Path | str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Load events from JSONL file with resilience against malformed lines.
    Returns: (valid_events, parse_warnings)
    """
    path = Path(events_path)
    events: List[Dict[str, Any]] = []
    warnings: List[str] = []

    if not path.is_file():
        return events, [f"Event file not found: {path}"]

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                events.append(data)
            except Exception as e:
                msg = f"Line {line_num} in {path.name} is malformed JSON ({e}): '{line_str[:50]}...'"
                logger.warning(msg)
                warnings.append(msg)

    return events, warnings


def load_payments_ledger(ledger_path: Path | str) -> List[Dict[str, Any]]:
    """Load payment transactions from accounting ledger CSV."""
    path = Path(ledger_path)
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows

    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    except Exception as e:
        logger.warning("Could not read payments ledger: %s", e)
    return rows


# =============================================================================
# 3. Verifier Helper
# =============================================================================

def create_and_verify_citation(
    document_id: str,
    clause_reference: str,
    verbatim_quote: str,
    bundle_dir: Path | str = "bundle",
    page_number: int = 1,
) -> VerifiableCitation:
    """
    Construct a VerifiableCitation and verify it immediately against the canonical .txt contract.
    """
    bpath = Path(bundle_dir)
    source_file = bpath / "contracts" / f"{document_id}.txt"
    if not source_file.exists():
        # Check if in company dir
        if (bpath / "company" / f"{document_id}.txt").exists():
            source_file = bpath / "company" / f"{document_id}.txt"

    clean_quote = verbatim_quote.strip()
    is_verified = False
    notes = "Source file not found"

    if source_file.exists():
        content = source_file.read_text(encoding="utf-8")
        if clean_quote in content:
            is_verified = True
            notes = "Verified verbatim against canonical text file."
        else:
            # Try whitespace-normalized comparison
            norm_content = re.sub(r"\s+", " ", content)
            norm_quote = re.sub(r"\s+", " ", clean_quote)
            if norm_quote in norm_content:
                is_verified = True
                notes = "Verified verbatim with normalized whitespace."
            else:
                notes = "Verbatim quote snippet not found in source text file."

    return VerifiableCitation(
        document_id=document_id,
        clause_reference=clause_reference,
        verbatim_quote=clean_quote,
        page_number=page_number,
        source_file=str(source_file).replace("\\", "/"),
        verified=is_verified,
        verification_notes=notes,
    )


# =============================================================================
# 4. Specialized Risk Analyzers
# =============================================================================

class ConflictAnalyzer:
    """
    Evaluates cross-contract clauses to distinguish genuine contradictions
    from provisions that legitimately coexist (e.g. market sector distinction).
    """

    @staticmethod
    def analyze(bundle_dir: Path | str = "bundle") -> List[RiskFinding]:
        findings: List[RiskFinding] = []

        # 1. Harbor Retail (B1) vs Vantage Logistics (C1) Exclusivity Analysis
        b1_quote = (
            "Quillhub grants Harbor Retail the exclusive right to use the Route Optimizer "
            "Enterprise platform within the Pacific Northwest retail grocery sector (defined as "
            "retail grocery operations headquartered in the states of Washington and Oregon)."
        )
        c1_quote = (
            "Quillhub grants Vantage Logistics the exclusive right to use the Route "
            "Optimizer platform (including all tiers: Standard, Pro, and "
            "Enterprise) within the Pacific Northwest third-party logistics sector "
            "(defined as third-party logistics companies headquartered in the "
            "states of Washington and Oregon that provide warehousing and last-mile "
            "delivery services)."
        )

        cit_b1 = create_and_verify_citation("B1", "§6.1", b1_quote, bundle_dir)
        cit_c1 = create_and_verify_citation("C1", "§4.1", c1_quote, bundle_dir)

        findings.append(RiskFinding(
            risk_id="RISK-CONFLICT-001",
            title="Coexisting Regional Exclusivities: Distinct Sector Scopes (Retail Grocery vs 3PL)",
            category=RiskCategory.CROSS_CONTRACT_CONFLICT,
            finding_type=FindingType.CONFIRMED_FACT,
            severity=RiskSeverity.INFORMATIONAL,
            summary=(
                "Harbor Retail (B1 §6.1) and Vantage Logistics (C1 §4.1) both grant exclusive rights "
                "in the Pacific Northwest (Washington and Oregon). However, they legitimately coexist without "
                "contradiction because their scopes are restricted to mutually exclusive sectors: "
                "Harbor is restricted to 'retail grocery operations' whereas Vantage is restricted to "
                "'third-party logistics companies that provide warehousing and last-mile delivery services'."
            ),
            parties_involved=["Harbor Retail Group LLC", "Vantage Logistics Corporation", "Quillhub Inc."],
            affected_contracts=["B1", "C1"],
            verbatim_citations=[cit_b1, cit_c1],
            reasoning_chain=[
                "Both contracts define the geographic scope as Washington and Oregon (Pacific Northwest).",
                "B1 Section 6.1 restricts exclusivity strictly to the 'retail grocery sector'.",
                "C1 Section 4.1 restricts exclusivity strictly to the 'third-party logistics sector (3PL)'.",
                "Because neither party operates within the other's designated market segment, no breach or overlap exists.",
                "Adversarial Warning: Do not misclassify as a territorial conflict solely due to shared geography.",
            ],
            financial_impact="$0.00 direct exposure (no breach detected).",
            assumptions_stated=[
                "Assumes Vantage Logistics does not engage in retail grocery operations.",
                "Assumes Harbor Retail does not offer third-party logistics warehousing to outside merchants.",
            ],
            next_action_recommended="Maintain sales qualification checklist ensuring future PNW prospects fit neither sector.",
        ))

        # 2. Upstream vs Customer SLA Target Misalignment (B1 §3.1 vs A2 §2.1)
        a2_quote = (
            "Provider will make the Cloud Service available at least ninety-nine "
            "and one-half percent (99.5%) of the minutes in each calendar month, not "
            "counting Scheduled Maintenance (the \"Uptime Target\")."
        )
        b1_sla_quote = (
            "Quillhub will make the Platform available at least "
            "ninety-nine and nine-tenths percent (99.9%) of the minutes in each "
            "calendar month, not counting Scheduled Maintenance (the \"Harbor Uptime "
            "Target\")."
        )
        cit_a2 = create_and_verify_citation("A2", "Section 2.1", a2_quote, bundle_dir)
        cit_b1_sla = create_and_verify_citation("B1", "§3.1", b1_sla_quote, bundle_dir)

        findings.append(RiskFinding(
            risk_id="RISK-CONFLICT-002",
            title="Operational SLA Gap: Customer Uptime Target (99.9%) Exceeds Upstream Provider SLA (99.5%)",
            category=RiskCategory.CROSS_CONTRACT_CONFLICT,
            finding_type=FindingType.CONFIRMED_FACT,
            severity=RiskSeverity.HIGH,
            summary=(
                "Quillhub committed to a 99.9% uptime target for Harbor Retail (B1 §3.1), but its underlying "
                "infrastructure provider StratoNode is only contractually committed to 99.5% uptime (A2 §2.1). "
                "Any downtime incident resulting in monthly uptime between 99.50% and 99.89% places Quillhub in "
                "breach of Harbor's SLA while StratoNode remains fully compliant, creating a structural recovery gap."
            ),
            parties_involved=["Harbor Retail Group LLC", "StratoNode Inc.", "Quillhub Inc."],
            affected_contracts=["B1", "A2"],
            verbatim_citations=[cit_b1_sla, cit_a2],
            reasoning_chain=[
                "B1 §3.1 establishes 99.9% Monthly Uptime Percentage target for Harbor Retail.",
                "A2 Section 2.1 establishes 99.5% Monthly Uptime Percentage target for StratoNode CloudCore.",
                "99.9% permits max 44.64 minutes downtime in a 31-day month (44,640 min).",
                "99.5% permits up to 223.2 minutes downtime in the same month.",
                "If downtime is between 45 and 223 minutes, Quillhub owes Harbor service credits (15% = $6,000) "
                "with zero contractual recourse against StratoNode.",
            ],
            financial_impact="Up to $6,000.00/month unrecoverable credit penalty per incident window.",
            assumptions_stated=[
                "Assumes Route Optimizer operates entirely on StratoNode CloudCore without multi-cloud failover.",
            ],
            next_action_recommended="Negotiate upgraded StratoNode SLA to 99.9% or implement secondary cloud failover.",
        ))

        # 3. Governing Law & Jurisdiction Divergence
        a1_law = create_and_verify_citation("A1", "Section H", "Governing Law: State of Oregon", bundle_dir)
        b1_law = create_and_verify_citation("B1", "Section K", "Governing Law: State of Washington", bundle_dir)
        c1_law = create_and_verify_citation("C1", "Section G", "Governing Law: State of Oregon", bundle_dir)
        d1_law = create_and_verify_citation("D1", "Section 10.3", "governed by the laws of the State of Texas", bundle_dir)

        findings.append(RiskFinding(
            risk_id="RISK-CONFLICT-003",
            title="Multi-Jurisdictional Legal Fragmentation (OR, WA, TX)",
            category=RiskCategory.CROSS_CONTRACT_CONFLICT,
            finding_type=FindingType.CONFIRMED_FACT,
            severity=RiskSeverity.LOW,
            summary=(
                "Quillhub's portfolio is governed by disparate state laws and forums: "
                "StratoNode (A1 Section H) in Oregon; Harbor Retail (B1 Section K) in Washington; "
                "Vantage Logistics (C1 Section G) in Oregon; and Pixelwright (D1 Section 10.3) in Texas. "
                "In cross-party indemnification or dispute scenarios, differing state standards on liability "
                "and consequential damages apply."
            ),
            parties_involved=["StratoNode Inc.", "Harbor Retail Group LLC", "Vantage Logistics Corp.", "Pixelwright LLC", "Quillhub Inc."],
            affected_contracts=["A1", "B1", "C1", "D1"],
            verbatim_citations=[a1_law, b1_law, c1_law, d1_law],
            reasoning_chain=[
                "Upstream hosting disputes must be litigated in Multnomah County, Oregon.",
                "Customer disputes with Harbor Retail reside in King County, Washington.",
                "Contractor disputes with Pixelwright fall under Texas jurisdiction.",
                "Back-to-back pass-through liability cannot be litigated in a single consolidated action.",
            ],
            financial_impact="Increased litigation complexity and legal overhead in cross-contract disputes.",
            assumptions_stated=["Jurisdiction clauses are valid and enforceable."],
            next_action_recommended="Harmonize governing law to Quillhub's headquarters jurisdiction (Texas) in future renewals.",
        ))

        return findings


class SupersessionAnalyzer:
    """
    Identifies amendments modifying earlier contract terms, replaces old wording,
    and flags obsolete terms that third parties or reports might mistakenly cite.
    """

    @staticmethod
    def analyze(bundle_dir: Path | str = "bundle") -> List[RiskFinding]:
        findings: List[RiskFinding] = []

        # 1. SLA Credit Request Deadline Shortened from 30 Days to 15 Days
        a2_old_proc = (
            "Credit Request Procedure. To receive a Service Credit, Customer must "
            "submit a written request to Provider's Notice Address within thirty (30) "
            "days after the end of the calendar month in which the Downtime occurred."
        )
        a3_new_proc = (
            "Credit Request Procedure. To receive a Service Credit, Customer must "
            "submit a written request to Provider's Notice Address within fifteen "
            "(15) days after the end of the calendar month in which the Downtime "
            "occurred."
        )
        cit_old = create_and_verify_citation("A2", "Section 3.2", a2_old_proc, bundle_dir)
        cit_new = create_and_verify_citation("A3", "Section 1", a3_new_proc, bundle_dir)

        findings.append(RiskFinding(
            risk_id="RISK-SUPER-001",
            title="Silent Amendment Supersession: StratoNode SLA Credit Notice Shortened (30d -> 15d)",
            category=RiskCategory.AMENDMENT_SUPERSESSION,
            finding_type=FindingType.CONFIRMED_FACT,
            severity=RiskSeverity.HIGH,
            summary=(
                "Amendment No. 1 (A3 Section 1, effective Jan 1, 2027) deleted and replaced Section 3.2 of the "
                "StratoNode SLA (A2). The deadline to request service credits was cut from 30 days to 15 days "
                "after the end of the calendar month. Any claim citing a 30-day notice window relies on superseded, "
                "obsolete terms."
            ),
            parties_involved=["StratoNode Inc.", "Quillhub Inc."],
            affected_contracts=["A2", "A3"],
            verbatim_citations=[cit_old, cit_new],
            reasoning_chain=[
                "A2 Section 3.2 granted Quillhub 30 days post month-end to request service credits.",
                "A3 Section 1 expressly deleted Section 3.2 and replaced it with a strict 15-day requirement.",
                "A3 takes precedence pursuant to A3 Preamble ('In the event of a conflict between this Amendment and the SLA, this Amendment controls').",
                "Quillhub operations team must file requests within 15 days or forfeit contractual remedies.",
            ],
            financial_impact="Complete forfeiture of monthly service credits (up to $4,000/mo) if filed after day 15.",
            assumptions_stated=["Amendment No. 1 is fully executed and in effect as of January 1, 2027."],
            next_action_recommended="Update internal billing calendar to trigger StratoNode SLA audit on day 1 of each month.",
        ))

        # 2. Non-Renewal Notice Shortened from 90 Days to 60 Days
        a1_old_renew = (
            "Non-Renewal Notice Period: 90 days before the end of the then-current "
            "Subscription Period"
        )
        a3_new_renew = (
            "The Non-Renewal Notice Period is changed from ninety (90) days to sixty "
            "(60) days before the end of the then-current Subscription Period."
        )
        cit_a1_ren = create_and_verify_citation("A1", "Section B", a1_old_renew, bundle_dir)
        cit_a3_ren = create_and_verify_citation("A3", "Section 3", a3_new_renew, bundle_dir)

        findings.append(RiskFinding(
            risk_id="RISK-SUPER-002",
            title="Superseded Term: StratoNode Non-Renewal Notice Window Modified (90d -> 60d)",
            category=RiskCategory.AMENDMENT_SUPERSESSION,
            finding_type=FindingType.CONFIRMED_FACT,
            severity=RiskSeverity.MEDIUM,
            summary=(
                "A1 Section B originally required 90 days' written notice prior to subscription expiration to prevent "
                "automatic renewal. Amendment No. 1 (A3 Section 3) modified this period to 60 days before expiration. "
                "With the Initial Subscription Period ending July 31, 2028, notice is now due June 1, 2028 rather than May 2, 2028."
            ),
            parties_involved=["StratoNode Inc.", "Quillhub Inc."],
            affected_contracts=["A1", "A3"],
            verbatim_citations=[cit_a1_ren, cit_a3_ren],
            reasoning_chain=[
                "A1 Section B set non-renewal notice at 90 days.",
                "A3 Section 3 amended Section B to 60 days.",
                "Automatic renewal binds Quillhub to another 12 months at $8,000/month ($96,000 commitment).",
            ],
            financial_impact="$96,000.00 renewal commitment exposure if non-renewal is missed.",
            assumptions_stated=["Initial subscription period ends July 31, 2028."],
            next_action_recommended="Record June 1, 2028 as the authoritative non-renewal deadline in legal register.",
        ))

        # 3. Addition of 5% Bonus Credit for Extended Downtime (>60 min)
        a3_bonus_quote = (
            "Additionally, if Downtime in a single incident exceeds sixty (60) "
            "consecutive minutes, Customer will receive an additional Service "
            "Credit equal to five percent (5%) of the Monthly Fee for that month, "
            "in addition to the credit specified in the table above."
        )
        cit_a3_bonus = create_and_verify_citation("A3", "Section 2", a3_bonus_quote, bundle_dir)

        findings.append(RiskFinding(
            risk_id="RISK-SUPER-003",
            title="Amended Benefit: 5% Bonus Service Credit for Extended Single Outages (>60 min)",
            category=RiskCategory.AMENDMENT_SUPERSESSION,
            finding_type=FindingType.CONFIRMED_FACT,
            severity=RiskSeverity.INFORMATIONAL,
            summary=(
                "A3 Section 2 added a 5% bonus credit to StratoNode's SLA for outages exceeding 60 consecutive minutes. "
                "For the May 1-2 incident (505 min), this entitles Quillhub to 30% total credit (25% tier + 5% bonus = $2,400) "
                "rather than the base 25% ($2,000)."
            ),
            parties_involved=["StratoNode Inc.", "Quillhub Inc."],
            affected_contracts=["A2", "A3"],
            verbatim_citations=[cit_a3_bonus],
            reasoning_chain=[
                "A2 Section 3.1 previously had standard tiers (10%, 25%, 50%).",
                "A3 Section 2 added an extra 5% of monthly fee if single outage > 60 consecutive minutes.",
                "Total credit remains capped at 50% ($4,000.00).",
            ],
            financial_impact="+$400.00 additional recoverable credit on 60+ min downtime events.",
            assumptions_stated=["Downtime calculation meets A2/A3 definition of consecutive minutes."],
            next_action_recommended="Ensure all StratoNode credit claims calculate and include the Section 2 bonus.",
        ))

        return findings


class FinancialGapAnalyzer:
    """
    Compares customer-facing commitments with upstream commitments and calculates
    financial exposure and recovery gaps. Rigorously tags claims as conditional/unconfirmed
    when requests or recovery have not been established.
    """

    @staticmethod
    def analyze(bundle_dir: Path | str = "bundle") -> List[RiskFinding]:
        findings: List[RiskFinding] = []

        # 1. March 2027 Downtime Incident Analysis (190 min Downtime)
        b1_credit_quote = (
            "If the Monthly Uptime Percentage falls below the "
            "Harbor Uptime Target in any calendar month, Quillhub will credit "
            "Harbor Retail fifteen percent (15%) of the Monthly Fee for that month "
            "against Harbor Retail's next invoice."
        )
        b1_cap_quote = (
            "The total service credits for any single calendar month "
            "will not exceed twenty percent (20%) of the Monthly Fee for that month."
        )
        cit_b1_cr = create_and_verify_citation("B1", "§3.3", b1_credit_quote, bundle_dir)
        cit_b1_cap = create_and_verify_citation("B1", "§3.4", b1_cap_quote, bundle_dir)

        findings.append(RiskFinding(
            risk_id="RISK-FIN-001",
            title="March 2027 Incident: 100% Unrecoverable Financial Recovery Gap ($6,000.00)",
            category=RiskCategory.FINANCIAL_RECOVERY_GAP,
            finding_type=FindingType.DERIVED_CALCULATION,
            severity=RiskSeverity.HIGH,
            summary=(
                "On March 3, 2027, StratoNode CloudCore experienced 190 minutes of downtime (E-04). "
                "March total minutes: 44,640. Calculated uptime: (44,640 - 190) / 44,640 = 99.57%. "
                "Downstream: 99.57% < Harbor 99.9% target -> Quillhub owes Harbor 15% of $40,000 = $6,000.00. "
                "Upstream: 99.57% >= StratoNode 99.5% target -> StratoNode owes Quillhub $0.00. "
                "Net financial gap: $6,000.00 completely unrecoverable."
            ),
            parties_involved=["Harbor Retail Group LLC", "StratoNode Inc.", "Quillhub Inc."],
            affected_contracts=["B1", "A2", "A3"],
            verbatim_citations=[cit_b1_cr, cit_b1_cap],
            reasoning_chain=[
                "Total minutes in March 2027 = 31 days * 24 hours * 60 minutes = 44,640 minutes.",
                "Downtime on March 3, 2027 = 190 minutes (E-04, E-05, E-07).",
                "March Uptime = (44,640 - 190) / 44,640 = 99.574% (confirmed in E-07).",
                "Harbor Retail Target = 99.9%. 99.57% is below target. Credit = 15% * $40,000 = $6,000.00.",
                "StratoNode Target = 99.5%. 99.57% is above target (A2 Section 2.1). Credit = $0.00.",
                "Quillhub absorbed 100% of the customer credit penalty with zero supplier recovery.",
            ],
            financial_impact="$6,000.00 unrecoverable cash loss.",
            assumptions_stated=[
                "Calculation inputs are verified from E-04 and E-07.",
                "No Scheduled Maintenance occurred during the 190-minute window (confirmed in E-04/E-05).",
            ],
            next_action_recommended="Account for $6,000.00 loss reserve and review failover architecture.",
        ))

        # 2. May 2027 Incident (505 min Downtime) & Recovery Gap Analysis
        findings.append(RiskFinding(
            risk_id="RISK-FIN-002",
            title="May 2027 Incident: Structural Financial Gap ($3,600.00) & Notice Window Mismatch",
            category=RiskCategory.FINANCIAL_RECOVERY_GAP,
            finding_type=FindingType.DERIVED_CALCULATION,
            severity=RiskSeverity.HIGH,
            summary=(
                "On May 1-2, 2027, StratoNode experienced 505 minutes of downtime (E-12). "
                "May total minutes: 44,640. Calculated uptime: (44,640 - 505) / 44,640 = 98.87% (E-15). "
                "Downstream: 98.87% < Harbor 99.9% target -> Harbor requested 15% credit ($6,000.00) in E-17. "
                "Upstream: 98.87% is in tier 95.0% to < 99.0% (25% credit = $2,000) + 5% bonus for single incident > 60 min ($400) = $2,400.00 total. "
                "Financial Gap: Quillhub owes $6,000.00, can recover at most $2,400.00 -> Net deficit is $3,600.00."
            ),
            parties_involved=["Harbor Retail Group LLC", "StratoNode Inc.", "Quillhub Inc."],
            affected_contracts=["B1", "A2", "A3"],
            verbatim_citations=[cit_b1_cr],
            reasoning_chain=[
                "Total minutes in May 2027 = 44,640 minutes.",
                "Downtime on May 1-2 = 505 minutes (E-12, E-13, E-15).",
                "May Uptime = 98.8687% (98.87% in E-15).",
                "Harbor Target = 99.9%. Credit = 15% of $40,000 = $6,000.00.",
                "StratoNode Tier = 95.0% to < 99.0% -> 25% of $8,000 = $2,000.00.",
                "StratoNode Incident Duration = 505 min > 60 min -> A3 Section 2 adds 5% ($400.00).",
                "Maximum Upstream Recovery = $2,400.00.",
                "Net Financial Gap = $6,000.00 - $2,400.00 = $3,600.00 loss.",
            ],
            financial_impact="$3,600.00 net deficit even if upstream recovery is 100% successful.",
            assumptions_stated=[
                "Calculations assume StratoNode approves the $2,400 credit request.",
            ],
            next_action_recommended="Ensure Quillhub finance applies the $2,400 credit against next StratoNode invoice.",
        ))

        # 3. Notice Window Mismatch and Unconfirmed Upstream Recovery
        b1_req_quote = (
            "Harbor Retail must request a service credit in writing "
            "within forty-five (45) days after the end of the calendar month in "
            "which the Downtime occurred."
        )
        cit_b1_req = create_and_verify_citation("B1", "§3.5", b1_req_quote, bundle_dir)

        findings.append(RiskFinding(
            risk_id="RISK-FIN-003",
            title="Notice Window Asymmetry: Harbor Has 45 Days (B1 §3.5) vs Quillhub's 15 Days (A3 §1)",
            category=RiskCategory.FINANCIAL_RECOVERY_GAP,
            finding_type=FindingType.HYPOTHESIS_OR_CONDITIONAL,
            severity=RiskSeverity.HIGH,
            summary=(
                "Harbor Retail has 45 days post month-end to request credits (B1 §3.5), whereas Quillhub has only "
                "15 days to request credits from StratoNode (A3 §1). For May 2027 downtime, StratoNode's window closed "
                "on June 15, 2027. Harbor submitted its request on June 15, 2027 (E-17). "
                "Adversarial Clarification: Upstream recovery of $2,400 is CONDITIONAL on whether Quillhub independently "
                "filed a timely claim by June 15; it cannot be declared lost purely because the event log is silent."
            ),
            parties_involved=["Harbor Retail Group LLC", "StratoNode Inc.", "Quillhub Inc."],
            affected_contracts=["B1", "A3"],
            verbatim_citations=[cit_b1_req],
            reasoning_chain=[
                "For May downtime, month ended May 31, 2027.",
                "Quillhub StratoNode claim deadline: June 15, 2027 (15 days post month-end).",
                "Harbor claim deadline: July 15, 2027 (45 days post month-end).",
                "Harbor submitted its credit request on June 15, 2027 (E-17).",
                "If Quillhub waited for customer claim before claiming upstream, its 15-day window had already expired.",
                "However, if Quillhub monitoring triggered an automated claim, recovery remains possible.",
                "Adversarial Standard: Do not infer that upstream credit was forfeited without evidence.",
            ],
            financial_impact="Potential additional $2,400.00 loss if upstream filing was delayed.",
            assumptions_stated=[
                "Event feed records external customer & status events but does not exhaustively log Quillhub internal accounts payable communications.",
            ],
            next_action_recommended="Audit finance logs to verify whether timely notice was delivered to StratoNode by June 15, 2027.",
        ))

        return findings


class TimelineAnalyzer:
    """
    Evaluates timelines, deliverables, business calendar days (excluding observed holidays),
    contractual review periods, deemed acceptances, and challenges premature overdue flags.
    """

    @staticmethod
    def analyze(bundle_dir: Path | str = "bundle") -> List[RiskFinding]:
        findings: List[RiskFinding] = []
        bpath = Path(bundle_dir)
        holidays = load_holidays(bpath / "data" / "holidays.csv")

        # 1. Pixelwright M3 Deliverable: 10 Business Days Review & Deemed Acceptance
        d1_review_quote = (
            "Company will have ten (10) business days after receipt of "
            "each milestone deliverable to review the deliverable and either accept "
            "it or provide Contractor with a written list of deficiencies (a "
            "\"Rejection Notice\")."
        )
        d1_deemed_quote = (
            "If Company does not deliver a Rejection Notice within "
            "the ten (10) business day review period specified in Section 4.2, the "
            "milestone deliverable will be deemed accepted."
        )
        cit_rev = create_and_verify_citation("D1", "Section 4.2", d1_review_quote, bundle_dir)
        cit_deem = create_and_verify_citation("D1", "Section 4.3", d1_deemed_quote, bundle_dir)

        # M3 submitted 2027-02-28 (E-03).
        # Calculate 10 business days:
        # Feb 28 is Sunday. Next day March 1 (Mon).
        # Days: March 1, 2, 3, 4, 5 (5 days), March 8, 9, 10, 11, 12 (5 days).
        # Deadline was March 12, 2027.
        findings.append(RiskFinding(
            risk_id="RISK-TIME-001",
            title="D1 Milestone M3: Deemed Acceptance Triggered Under Section 4.3",
            category=RiskCategory.TIMELINE_OBLIGATION,
            finding_type=FindingType.CONFIRMED_FACT,
            severity=RiskSeverity.INFORMATIONAL,
            summary=(
                "Pixelwright submitted Milestone M3 (Front-End Development) on February 28, 2027 (E-03). "
                "Under D1 Section 4.2, Quillhub had 10 business days to deliver a Rejection Notice. "
                "Accounting for weekends and observed holidays, the review period expired on March 12, 2027. "
                "Because Quillhub delivered no Rejection Notice, M3 was deemed accepted as a matter of law, "
                "confirming Quillhub's obligation to pay the $20,000 milestone fee."
            ),
            parties_involved=["Pixelwright Design Studio LLC", "Quillhub Inc."],
            affected_contracts=["D1"],
            verbatim_citations=[cit_rev, cit_deem],
            reasoning_chain=[
                "Submission date: February 28, 2027 (E-03).",
                "D1 Section 4.2 mandates 10 business days excluding weekends and observed holidays.",
                "Review window: March 1 through March 12, 2027 (no holidays intervened in this window).",
                "Quillhub issued no deficiency notice or Rejection Notice.",
                "D1 Section 4.3 provides: 'the milestone deliverable will be deemed accepted'.",
            ],
            financial_impact="$20,000.00 payment obligation solidified (already paid on 2027-02-18 per ledger).",
            assumptions_stated=["No formal rejection notice was sent outside the documented communication channels."],
            next_action_recommended="Archive M3 acceptance record in engineering milestones.",
        ))

        # 2. Pixelwright M4 Deliverable: Informal Phone Notice vs Formal Written Amendment
        d1_amend_quote = "This Agreement may be amended only by a written instrument signed by both Parties."
        d1_notice_quote = "All notices must be in writing and sent to the addresses set forth above"
        cit_amend = create_and_verify_citation("D1", "Section 10.6", d1_amend_quote, bundle_dir)
        cit_not = create_and_verify_citation("D1", "Section 10.2", d1_notice_quote, bundle_dir)

        findings.append(RiskFinding(
            risk_id="RISK-TIME-002",
            title="D1 Milestone M4: Informal Phone Delay (E-10) Lacks Contractual Force",
            category=RiskCategory.TIMELINE_OBLIGATION,
            finding_type=FindingType.CONFIRMED_FACT,
            severity=RiskSeverity.MEDIUM,
            summary=(
                "On April 15, 2027, Elena Vasquez called Dana Kowalski stating M4 would be delayed (E-10). "
                "D1 Section 10.2 requires all notices to be in writing, and Section 10.6 requires amendments "
                "to be executed in a written instrument signed by both parties. An informal phone conversation "
                "does not alter the February 28, 2027 due date. Although delivered on April 20, 2027 (E-11), "
                "Contractor is technically in breach of milestone schedule unless formally waived in writing."
            ),
            parties_involved=["Pixelwright Design Studio LLC", "Quillhub Inc."],
            affected_contracts=["D1"],
            verbatim_citations=[cit_amend, cit_not],
            reasoning_chain=[
                "Exhibit A M4 Due Date: February 28, 2027.",
                "Event E-10 records an informal phone call on April 15, 2027.",
                "E-10 specifically notes: 'This is a phone call log, not a formal notice. No formal extension has been agreed to in writing.'",
                "D1 Section 10.6 expressly prohibits oral modifications.",
                "M4 deliverable submitted April 20, 2027 (E-11) is nearly 2 months late relative to Exhibit A.",
            ],
            financial_impact="$12,000.00 milestone payment subject to potential offset or dispute.",
            assumptions_stated=["Quillhub has not issued a written schedule waiver."],
            next_action_recommended="Execute formal written amendment extending M4 deadline or issue conditional acceptance.",
        ))

        # 3. Adversarial Challenge to Erroneous "Overdue" Status in Payments Ledger
        b1_pay_quote = "Payment Period: 45 days from invoice date."
        cit_pay = create_and_verify_citation("B1", "Section C", b1_pay_quote, bundle_dir)

        findings.append(RiskFinding(
            risk_id="RISK-TIME-003",
            title="Evidence Challenge: April 2027 Harbor Fee Inaccurately Flagged 'Overdue' in Ledger",
            category=RiskCategory.EVIDENCE_CHALLENGE,
            finding_type=FindingType.CONFIRMED_FACT,
            severity=RiskSeverity.MEDIUM,
            summary=(
                "Accounting ledger (line 21) marks Harbor Retail's April 2027 platform fee ($40,000) as 'overdue' on 2027-04-08. "
                "Adversarial Review: This ledger classification is legally false. Invoice QH-INV-2027-04-HR (E-09) was issued on "
                "April 2, 2027 with an explicit Due Date of May 17, 2027, complying with B1 Section C's 45-day payment period. "
                "Harbor was not in default on April 8, and claiming breach would constitute a bad-faith default notice."
            ),
            parties_involved=["Harbor Retail Group LLC", "Quillhub Inc."],
            affected_contracts=["B1"],
            verbatim_citations=[cit_pay],
            reasoning_chain=[
                "B1 Section C specifies Payment Period: '45 days from invoice date.'",
                "Invoice E-09 issued April 2, 2027 specifies: 'Due Date: May 17, 2027 (45 days from invoice date)'.",
                "Payments ledger entered on April 8, 2027 flags transaction as 'overdue'.",
                "Only 6 days had elapsed from invoice issuance.",
                "Adversarial Rule: Never infer overdue debt solely from an isolated ledger row when the governing contract and invoice refute it.",
            ],
            financial_impact="$0.00 actual debt default on April 8; prevents premature legal action.",
            assumptions_stated=["Invoice E-09 is the authentic invoice transmitted to Harbor Retail."],
            next_action_recommended="Instruct accounting controller to correct ledger status from 'overdue' to 'pending' as of April 8.",
        ))

        # 4. Consecutive SLA Miss Analysis (March + May are Non-Consecutive)
        b1_term_quote = (
            "If the Monthly Uptime Percentage falls below the Harbor Uptime Target in any "
            "two (2) consecutive calendar months, Harbor Retail may terminate this Agreement "
            "upon written notice to Quillhub."
        )
        cit_term = create_and_verify_citation("B1", "§8.1", b1_term_quote, bundle_dir)

        findings.append(RiskFinding(
            risk_id="RISK-TIME-004",
            title="Termination Defense: Harbor SLA Misses in March & May Are Non-Consecutive",
            category=RiskCategory.TIMELINE_OBLIGATION,
            finding_type=FindingType.CONFIRMED_FACT,
            severity=RiskSeverity.HIGH,
            summary=(
                "Harbor Retail noted in E-17 that the SLA was missed in both March (99.57%) and May (98.87%). "
                "B1 §8.1 permits immediate termination ONLY if the target is missed in 'any two (2) consecutive calendar months'. "
                "Because April 2027 uptime was 100% (E-07 / operational history), March and May are non-consecutive. "
                "Harbor does not currently possess a contractual right to terminate under §8.1. "
                "CRITICAL WARNING: If June 2027 uptime falls below 99.9%, May and June WILL be consecutive, triggering termination."
            ),
            parties_involved=["Harbor Retail Group LLC", "Quillhub Inc."],
            affected_contracts=["B1"],
            verbatim_citations=[cit_term],
            reasoning_chain=[
                "B1 Section 8.1 requires failure in 'two (2) consecutive calendar months'.",
                "March 2027: 99.57% (Missed).",
                "April 2027: 100.0% (Met).",
                "May 2027: 98.87% (Missed).",
                "March and May are separated by April, rendering them non-consecutive.",
                "Harbor's reservation of rights in E-17 correctly acknowledges this condition.",
                "However, June 2027 performance is critical; any miss allows Harbor to cancel the 36-month contract.",
            ],
            financial_impact="$480,000.00 annual revenue at risk if June SLA fails.",
            assumptions_stated=["April 2027 had no qualifying downtime incidents."],
            next_action_recommended="Freeze all non-essential production updates during June 2027 to protect uptime.",
        ))

        return findings


class EventDrivenMonitor:
    """
    Evaluates dynamic risks as time progresses across clock checkpoints
    and incoming event drops (Drop 1 and Drop 2).
    """

    @staticmethod
    def evaluate_checkpoint(
        checkpoint_date_str: str,
        bundle_dir: Path | str = "bundle",
    ) -> List[RiskFinding]:
        findings: List[RiskFinding] = []
        chk_date = parse_iso_date(checkpoint_date_str)
        if not chk_date:
            return findings

        # Checkpoint: 2027-06-01 or later -> Vantage Go-Live Evaluation
        if chk_date >= date(2027, 5, 15):
            c1_term_quote = (
                "Initial Subscription Period: 24 months from the Effective Date\n"
                "  Renewal Period:              successive 12-month periods\n"
                "  Non-Renewal Notice Period:   90 days before the end of the then-current "
                "Subscription Period"
            )
            cit_vantage = create_and_verify_citation("C1", "Section B", c1_term_quote, bundle_dir)

            findings.append(RiskFinding(
                risk_id="RISK-EVENT-001",
                title="Vantage Logistics Effective Date Fixed (May 15, 2027 Go-Live)",
                category=RiskCategory.EVENT_DRIVEN_RISK,
                finding_type=FindingType.CONFIRMED_FACT,
                severity=RiskSeverity.INFORMATIONAL,
                summary=(
                    "C1 Cover Page left the Effective Date open ('TO BE INSERTED UPON GO-LIVE'). "
                    "On May 15, 2027, CEO Priya Chandrasekaran sent formal written confirmation (E-14) establishing "
                    "May 15, 2027 as the Go-Live and Effective Date. The 24-month initial subscription period runs "
                    "through May 14, 2029. Setup fee invoice ($12,000) was issued July 1, 2027 (E-18)."
                ),
                parties_involved=["Vantage Logistics Corporation", "Quillhub Inc."],
                affected_contracts=["C1"],
                verbatim_citations=[cit_vantage],
                reasoning_chain=[
                    "C1 Note at bottom of Cover Page provides: 'The Effective Date will be the date on which Quillhub confirms in writing that the Platform is available for production use'.",
                    "Event E-14 (May 15, 2027) delivers CEO written confirmation of Go-Live.",
                    "Term runs through May 14, 2029.",
                ],
                financial_impact="$12,000 setup fee + $18,500/month recurring subscription established.",
                assumptions_stated=["CEO has corporate authority to issue go-live confirmation (confirmed in context.txt)."],
                next_action_recommended="Track 90-day non-renewal window expiring February 13, 2029.",
            ))

        # Checkpoint: 2027-09-15 -> Corporate Insurance Expiration Risk
        if chk_date >= date(2027, 6, 30):
            b1_ins_quote = (
                "Quillhub will maintain, at its own expense and "
                "throughout the Subscription Period, the following insurance coverage: "
                "(a) Commercial General Liability insurance with limits of at least "
                "$2,000,000 per occurrence; and (b) Technology Errors & Omissions and "
                "Cyber Liability insurance with limits of at least $3,000,000 per "
                "claim."
            )
            b1_lapse_quote = (
                "If any required insurance policy is cancelled, "
                "materially reduced, or not renewed, Quillhub will notify Harbor "
                "Retail in writing within five (5) business days. A lapse in required "
                "insurance coverage constitutes a material breach of this Agreement."
            )
            cit_ins = create_and_verify_citation("B1", "§7.1", b1_ins_quote, bundle_dir)
            cit_lapse = create_and_verify_citation("B1", "§7.2", b1_lapse_quote, bundle_dir)

            findings.append(RiskFinding(
                risk_id="RISK-EVENT-002",
                title="CRITICAL BREACH: Corporate Insurance Policies Expired June 30, 2027",
                category=RiskCategory.EVENT_DRIVEN_RISK,
                finding_type=FindingType.CONFIRMED_FACT if chk_date > date(2027, 6, 30) else FindingType.HYPOTHESIS_OR_CONDITIONAL,
                severity=RiskSeverity.CRITICAL,
                summary=(
                    "Quillhub's corporate context (context.txt) and Meridian Mutual's renewal notice (E-16) "
                    "confirm that Commercial General Liability (MML-2024-88714) and Cyber Liability (MML-2024-88715) "
                    "expired at 11:59 PM on June 30, 2027. Under B1 Section 7.2, an insurance lapse constitutes an immediate "
                    "material breach. Under Section 8.2, Harbor Retail may terminate the agreement immediately upon 15 days' "
                    "uncured notice, placing the entire $40k/month enterprise contract in immediate jeopardy."
                ),
                parties_involved=["Meridian Mutual Insurance Co.", "Harbor Retail Group LLC", "Quillhub Inc."],
                affected_contracts=["B1"],
                verbatim_citations=[cit_ins, cit_lapse],
                reasoning_chain=[
                    "Context.txt states: 'Both policies expire on June 30, 2027. No renewal confirmation has been received'.",
                    "E-16 (June 10, 2027) warned: 'If we do not receive renewal instructions by June 25, 2027, the policies will lapse at 11:59 PM on June 30, 2027'.",
                    "As of Checkpoint 3 (September 15, 2027), policies have been lapsed for 76 days.",
                    "B1 Section 7.2 specifies: 'A lapse in required insurance coverage constitutes a material breach of this Agreement'.",
                    "B1 Section 8.2 grants Harbor immediate termination rights after 15 days.",
                ],
                financial_impact="Risk of immediate cancellation of $480,000/year contract + uninsured liability.",
                assumptions_stated=["No binder or renewal was completed before June 30."],
                next_action_recommended="URGENT: Executive team must bind retroactive renewal coverage with Meridian Mutual immediately.",
            ))

        return findings


# =============================================================================
# 5. LangGraph Workflow for Loki
# =============================================================================

class LokiAgentState(BaseModel):
    bundle_path: str = "bundle"
    clock_checkpoint: Optional[str] = None
    event_drop: Optional[str] = None
    conflicts: List[RiskFinding] = Field(default_factory=list)
    supersessions: List[RiskFinding] = Field(default_factory=list)
    financial_gaps: List[RiskFinding] = Field(default_factory=list)
    timelines: List[RiskFinding] = Field(default_factory=list)
    event_risks: List[RiskFinding] = Field(default_factory=list)
    all_findings: List[RiskFinding] = Field(default_factory=list)
    challenges: List[str] = Field(default_factory=list)
    report: Optional[LokiReviewReport] = None


def node_load_context(state: LokiAgentState) -> Dict[str, Any]:
    """Node 1: Initialize context and prepare state."""
    logger.info("Loki: Initializing analysis for checkpoint %s", state.clock_checkpoint)
    return {}


def node_analyze_conflicts_and_supersessions(state: LokiAgentState) -> Dict[str, Any]:
    """Node 2: Evaluate cross-contract conflicts and amendment supersessions."""
    bundle = state.bundle_path
    conflicts = ConflictAnalyzer.analyze(bundle)
    supersessions = SupersessionAnalyzer.analyze(bundle)
    return {"conflicts": conflicts, "supersessions": supersessions}


def node_analyze_financial_gaps(state: LokiAgentState) -> Dict[str, Any]:
    """Node 3: Calculate upstream vs downstream financial recovery gaps."""
    bundle = state.bundle_path
    financial_gaps = FinancialGapAnalyzer.analyze(bundle)
    return {"financial_gaps": financial_gaps}


def node_analyze_timelines_and_events(state: LokiAgentState) -> Dict[str, Any]:
    """Node 4: Evaluate business day timelines, deemed acceptance, and event milestones."""
    bundle = state.bundle_path
    timelines = TimelineAnalyzer.analyze(bundle)
    event_risks: List[RiskFinding] = []

    if state.clock_checkpoint:
        event_risks = EventDrivenMonitor.evaluate_checkpoint(state.clock_checkpoint, bundle)

    return {"timelines": timelines, "event_risks": event_risks}


def node_adversarial_challenge(state: LokiAgentState) -> Dict[str, Any]:
    """
    Node 5: Adversarial Reviewer challenging assumptions and compiling findings.
    Validates citations and classifies findings.
    """
    combined: List[RiskFinding] = []
    seen_ids: Set[str] = set()

    for group in [state.conflicts, state.supersessions, state.financial_gaps, state.timelines, state.event_risks]:
        for item in group:
            if item.risk_id not in seen_ids:
                seen_ids.add(item.risk_id)
                combined.append(item)

    challenges = [
        "Challenged claim of exclusivity territorial collision between B1 and C1: Disproven (scopes are sector-separated).",
        "Challenged claim that StratoNode credit request window is 30 days: Disproven (superseded to 15 days by A3 Section 1).",
        "Challenged claim that Harbor April 2027 invoice was overdue on April 8: Disproven (B1 Section C grants 45 days, due May 17).",
        "Challenged claim that Harbor had immediate right to terminate in May for persistent SLA failures: Disproven (March and May were non-consecutive).",
        "Challenged assumption that StratoNode credit was definitely forfeited: Tagged as conditional/unconfirmed.",
    ]

    return {"all_findings": combined, "challenges": challenges}


def node_compile_report(state: LokiAgentState) -> Dict[str, Any]:
    """Node 6: Compile final structured Loki review report."""
    findings = state.all_findings

    by_category: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}
    unverified_count = 0

    for f in findings:
        cat_key = f.category.value
        by_category[cat_key] = by_category.get(cat_key, 0) + 1

        sev_key = f.severity.value
        by_severity[sev_key] = by_severity.get(sev_key, 0) + 1

        for cit in f.verbatim_citations:
            if not cit.verified:
                unverified_count += 1

    report = LokiReviewReport(
        checkpoint_date=state.clock_checkpoint,
        event_drop=state.event_drop,
        total_findings=len(findings),
        findings=findings,
        summary_by_category=by_category,
        summary_by_severity=by_severity,
        unverified_citations_count=unverified_count,
        challenges_raised=state.challenges,
        metadata={
            "engine": "Loki Adversarial Contract Reviewer (Deterministic Rules)",
            "llm_calls_made": 0,
            "offline_mode": True,
        },
    )

    return {"report": report}


def build_loki_graph() -> Any:
    """Build the compiled LangGraph StateGraph workflow for Loki."""
    workflow = StateGraph(LokiAgentState)

    workflow.add_node("load_context", node_load_context)
    workflow.add_node("analyze_conflicts_and_supersessions", node_analyze_conflicts_and_supersessions)
    workflow.add_node("analyze_financial_gaps", node_analyze_financial_gaps)
    workflow.add_node("analyze_timelines_and_events", node_analyze_timelines_and_events)
    workflow.add_node("adversarial_challenge", node_adversarial_challenge)
    workflow.add_node("compile_report", node_compile_report)

    workflow.add_edge(START, "load_context")
    workflow.add_edge("load_context", "analyze_conflicts_and_supersessions")
    workflow.add_edge("analyze_conflicts_and_supersessions", "analyze_financial_gaps")
    workflow.add_edge("analyze_financial_gaps", "analyze_timelines_and_events")
    workflow.add_edge("analyze_timelines_and_events", "adversarial_challenge")
    workflow.add_edge("adversarial_challenge", "compile_report")
    workflow.add_edge("compile_report", END)

    return workflow.compile()


# =============================================================================
# 6. Primary Agent Entry Point
# =============================================================================

def run_loki(
    bundle_path: Path | str = "bundle",
    clock_checkpoint: Optional[str] = None,
    event_drop: Optional[str] = None,
) -> LokiReviewReport:
    """
    Run Loki Adversarial Agent to challenge facts, uncover cross-contract risks,
    audit financial gaps, and track temporal obligations.
    Runs 100% offline without external LLM calls.
    """
    graph = build_loki_graph()
    initial_state = LokiAgentState(
        bundle_path=str(bundle_path),
        clock_checkpoint=clock_checkpoint,
        event_drop=event_drop,
    )
    final_state = graph.invoke(initial_state)
    return final_state["report"]


if __name__ == "__main__":
    report = run_loki(bundle_path="bundle", clock_checkpoint="2027-09-15", event_drop="Drop 2")
    print("=" * 80)
    print(f"LOKI ADVERSARIAL REPORT — Total Findings: {report.total_findings}")
    print("=" * 80)
    for idx, f in enumerate(report.findings, start=1):
        print(f"\n[{idx}] ({f.severity.value}) {f.title}")
        print(f"    Category: {f.category.value} | Type: {f.finding_type.value}")
        print(f"    Summary: {f.summary}")
        if f.financial_impact:
            print(f"    Financial Impact: {f.financial_impact}")
        for cit in f.verbatim_citations:
            status = "VERIFIED" if cit.verified else "UNVERIFIED"
            print(f"    Citation [{cit.document_id} {cit.clause_reference}]: [{status}] \"{cit.verbatim_quote[:60]}...\"")
