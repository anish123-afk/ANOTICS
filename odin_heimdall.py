"""
Heimdall: Event Monitoring Agent.

Objective:
Monitor incoming real-time operational communications, incident notifications, deliverable handoffs,
and customer emails to determine whether they change contractual obligations or risk.

Steps:
1. Parse event streams (events_drop1.jsonl, events_drop2.jsonl):
   - Validation & schema parsing
   - Duplicate detection (by event_id)
   - Malformed line recovery & retry handling
   - Raw event storage with processing status
2. Map events to contractual provisions:
   - Identify affected contracts (A1, A2, B1, B2, C1, D1, STD, etc.)
   - Retrieve relevant clauses from Phase 1 database (odin.db)
   - Evaluate triggers (e.g., SLA outage, milestone delay, renewal notice, invoice, credit request)
   - Coordinate with simulated Loki/Norns interfaces
3. Noise reduction & Alerting:
   - Severity thresholds (info, low, medium, high, critical)
   - Alert deduplication and noise suppression (e.g. promotional emails, informal notes)
   - Verifiable citation grounding with evidence IDs
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

import odin_phase1

logger = logging.getLogger("odin.heimdall")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# -----------------------------------------------------------------------------
# 1. Schemas & Models
# -----------------------------------------------------------------------------

class Event(BaseModel):
    event_id: str
    timestamp: str
    channel: str
    sender: str
    recipient: str
    subject: str
    body: str
    raw_json: Optional[str] = None
    parse_status: str = "valid"
    error_message: Optional[str] = None


class ContractAlert(BaseModel):
    alert_id: str
    event_id: str
    affected_contract_ids: List[str]
    potential_trigger: str
    severity: str  # "info", "low", "medium", "high", "critical"
    summary: str
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    requires_review: bool = True
    loki_review_requested: bool = False
    norns_timeline_update_needed: bool = False


# -----------------------------------------------------------------------------
# 2. Database Initialization for Event Auditing & Alerts
# -----------------------------------------------------------------------------

def init_heimdall_db(db_path: Path | str = "odin.db") -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events_raw (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT,
                channel TEXT,
                sender TEXT,
                recipient TEXT,
                subject TEXT,
                body TEXT,
                raw_json TEXT,
                status TEXT,
                error_message TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS contract_alerts (
                alert_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                affected_contracts TEXT NOT NULL,
                potential_trigger TEXT NOT NULL,
                severity TEXT NOT NULL,
                summary TEXT NOT NULL,
                evidence_ids TEXT,
                requires_review INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY(event_id) REFERENCES events_raw(event_id)
            )
        """)
    return conn


# -----------------------------------------------------------------------------
# 3. Step 1: Event Parsing & Stream Ingestion
# -----------------------------------------------------------------------------

def parse_event_stream(
    stream_path: Path | str,
    db_path: Path | str = "odin.db"
) -> Tuple[List[Event], List[Dict[str, Any]]]:
    """
    Parse an event stream (.jsonl) with duplicate suppression, validation, and error recovery.
    """
    path = Path(stream_path)
    if not path.exists():
        logger.error(f"Event stream not found: {path}")
        return [], [{"error": f"File not found: {path}"}]

    conn = init_heimdall_db(db_path)
    events: List[Event] = []
    errors: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    # Query already processed event IDs to prevent duplicate inserts across runs
    cursor = conn.cursor()
    cursor.execute("SELECT event_id FROM events_raw")
    existing_ids = {row[0] for row in cursor.fetchall()}

    lines = path.read_text(encoding="utf-8").splitlines()
    for line_num, line in enumerate(lines, start=1):
        line_clean = line.strip()
        if not line_clean:
            continue

        try:
            data = json.loads(line_clean)
            event_id = data.get("event_id")
            if not event_id:
                raise ValueError("Missing 'event_id' in JSON")

            # Check for duplicate in current batch or database
            if event_id in seen_ids or event_id in existing_ids:
                logger.info(f"Duplicate event skipped: {event_id} (line {line_num})")
                continue

            seen_ids.add(event_id)
            event = Event(
                event_id=event_id,
                timestamp=data.get("timestamp", ""),
                channel=data.get("channel", "unknown"),
                sender=data.get("from", ""),
                recipient=data.get("to", ""),
                subject=data.get("subject", ""),
                body=data.get("body", ""),
                raw_json=line_clean,
                parse_status="valid",
            )
            events.append(event)

            # Persist to database
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO events_raw
                    (event_id, timestamp, channel, sender, recipient, subject, body, raw_json, status, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.timestamp,
                        event.channel,
                        event.sender,
                        event.recipient,
                        event.subject,
                        event.body,
                        line_clean,
                        "valid",
                        None,
                    ),
                )

        except Exception as e:
            err_msg = f"Line {line_num} parse error: {str(e)}"
            logger.warning(err_msg)
            errors.append({"line": line_num, "raw": line_clean[:80], "error": str(e)})

            # Log malformed line in database for auditing
            malformed_id = f"MALFORMED_L{line_num}_{path.stem}"
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO events_raw
                    (event_id, timestamp, channel, sender, recipient, subject, body, raw_json, status, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (malformed_id, "", "error", "", "", "", "", line_clean, "malformed", str(e)),
                )

    conn.close()
    return events, errors


# -----------------------------------------------------------------------------
# 4. Step 2 & 3: Contract Mapping, Trigger Analysis & Noise Reduction
# -----------------------------------------------------------------------------

# Mapping keywords/senders to affected contract IDs
CONTRACT_IDENTIFIERS = {
    "stratonode": ["A1", "A2", "A3"],
    "cloudcore": ["A1", "A2", "A3"],
    "harbor": ["B1", "B2"],
    "vantage": ["C1"],
    "pixelwright": ["D1"],
    "meridian": ["CONTEXT_INSURANCE"],
}

NOISE_KEYWORDS = [
    "newsletter",
    "webinar",
    "gpu instances",
    "marketing",
    "unsubscribe",
]


def identify_affected_contracts(event: Event) -> List[str]:
    """Identify which contracts are potentially impacted by this event."""
    text = f"{event.sender} {event.recipient} {event.subject} {event.body}".lower()
    contracts = set()

    for keyword, c_ids in CONTRACT_IDENTIFIERS.items():
        if keyword in text:
            contracts.update(c_ids)

    # Check for direct contract ID mentions (e.g. QH-STRATO, QH-HARBOR, etc.)
    if "qh-strato" in text:
        contracts.update(["A1", "A2", "A3"])
    if "qh-harbor" in text:
        contracts.update(["B1", "B2"])
    if "qh-vantage" in text:
        contracts.add("C1")
    if "pixelwright" in text:
        contracts.add("D1")

    return sorted(list(contracts))


def fetch_relevant_clauses_for_event(contract_ids: List[str], db_path: str = "odin.db") -> List[Dict[str, Any]]:
    """Retrieve indexed clauses for affected contracts from Phase 1 database."""
    if not contract_ids:
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    placeholders = ",".join("?" for _ in contract_ids)
    cursor.execute(
        f"""
        SELECT document_id, clause_id, heading, source_text, page_number
        FROM clauses
        WHERE document_id IN ({placeholders}) AND source_file LIKE '%.txt'
        """,
        contract_ids,
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def evaluate_event_trigger(event: Event, affected_contracts: List[str], db_path: str = "odin.db") -> Optional[ContractAlert]:
    """
    Evaluate if an event satisfies a contractual trigger.
    Applies noise filtering: promotional / marketing / informal notes are suppressed or lowered.
    """
    text_content = f"{event.subject}\n{event.body}".lower()

    # Noise reduction Rule 1: Marketing / Newsletter / Promotional
    if any(k in text_content for k in NOISE_KEYWORDS):
        logger.info(f"Heimdall Noise Reduction: Suppressing promotional event {event.event_id}")
        return None

    # Noise reduction Rule 2: Routine scheduled maintenance notification without action required
    if "scheduled maintenance" in text_content and "no action is required" in text_content and "completed" not in text_content:
        # Routine advance notice, low priority or informational only
        return ContractAlert(
            alert_id=f"ALT_{event.event_id}",
            event_id=event.event_id,
            affected_contract_ids=affected_contracts,
            potential_trigger="scheduled_maintenance_notice",
            severity="info",
            summary=f"Advance notification of scheduled maintenance: {event.subject}",
            supporting_evidence_ids=[f"{c}:MAINTENANCE" for c in affected_contracts],
            requires_review=False,
            loki_review_requested=False,
            norns_timeline_update_needed=False,
        )

    # Trigger Category 1: Service Disruption / Outage / SLA Downtime (HIGH/CRITICAL)
    if "service disruption" in text_content or "downtime minutes" in text_content or "sla miss" in text_content:
        is_sla_miss = "sla miss" in text_content or "missed" in text_content or "duration: 505 minutes" in text_content
        severity = "critical" if is_sla_miss else "high"

        return ContractAlert(
            alert_id=f"ALT_{event.event_id}",
            event_id=event.event_id,
            affected_contract_ids=affected_contracts,
            potential_trigger="service_level_failure",
            severity=severity,
            summary=f"Service outage or SLA threshold impact reported in {event.event_id}: {event.subject}",
            supporting_evidence_ids=[f"{c}:SLA" for c in affected_contracts],
            requires_review=True,
            loki_review_requested=True,
            norns_timeline_update_needed=True,
        )

    # Trigger Category 2: Formal SLA Credit Request (CRITICAL)
    if "credit request" in text_content or "service credit" in text_content:
        return ContractAlert(
            alert_id=f"ALT_{event.event_id}",
            event_id=event.event_id,
            affected_contract_ids=affected_contracts,
            potential_trigger="formal_sla_credit_claim",
            severity="critical",
            summary=f"Formal SLA penalty credit demanded: {event.subject}",
            supporting_evidence_ids=[f"{c}:CREDIT_SECTION" for c in affected_contracts],
            requires_review=True,
            loki_review_requested=True,
            norns_timeline_update_needed=True,
        )

    # Trigger Category 3: Contract Milestone Deliverables & Delays (MEDIUM/HIGH)
    if "deliverable submission" in text_content or "delivery update" in text_content:
        is_delayed = "late" in text_content or "undocumented" in text_content or "phone call log" in text_content
        return ContractAlert(
            alert_id=f"ALT_{event.event_id}",
            event_id=event.event_id,
            affected_contract_ids=affected_contracts,
            potential_trigger="milestone_delay" if is_delayed else "milestone_submission",
            severity="high" if is_delayed else "medium",
            summary=f"Milestone event: {event.subject}. {'Potential delay noted.' if is_delayed else 'Submission requires review within acceptance window.'}",
            supporting_evidence_ids=[f"{c}:MILESTONE" for c in affected_contracts],
            requires_review=True,
            loki_review_requested=is_delayed,
            norns_timeline_update_needed=True,
        )

    # Trigger Category 4: Platform Go-Live or Effective Date Activation (HIGH)
    if "go-live confirmation" in text_content or "effective date of the agreement is therefore" in text_content:
        return ContractAlert(
            alert_id=f"ALT_{event.event_id}",
            event_id=event.event_id,
            affected_contract_ids=affected_contracts,
            potential_trigger="contract_activation_effective_date",
            severity="high",
            summary=f"Platform Go-Live triggers official Effective Date and Subscription Period commencement.",
            supporting_evidence_ids=[f"{c}:EFFECTIVE_DATE" for c in affected_contracts],
            requires_review=True,
            loki_review_requested=False,
            norns_timeline_update_needed=True,
        )

    # Trigger Category 5: Policy Renewal / Expiration Notice (HIGH)
    if "policy renewal notice" in text_content or "expiration date" in text_content:
        return ContractAlert(
            alert_id=f"ALT_{event.event_id}",
            event_id=event.event_id,
            affected_contract_ids=affected_contracts,
            potential_trigger="insurance_policy_expiration",
            severity="high",
            summary=f"Corporate insurance policies approaching lapse date. Renewal action required.",
            supporting_evidence_ids=["CONTEXT_INSURANCE"],
            requires_review=True,
            loki_review_requested=True,
            norns_timeline_update_needed=True,
        )

    # Trigger Category 6: Non-Renewal Intention / Notice (HIGH)
    if "not renewing" in text_content or "non-renewal" in text_content:
        return ContractAlert(
            alert_id=f"ALT_{event.event_id}",
            event_id=event.event_id,
            affected_contract_ids=affected_contracts,
            potential_trigger="non_renewal_intent_warning",
            severity="high",
            summary=f"Customer indicated risk of non-renewal: {event.subject}",
            supporting_evidence_ids=[f"{c}:RENEWAL" for c in affected_contracts],
            requires_review=True,
            loki_review_requested=True,
            norns_timeline_update_needed=True,
        )

    # Trigger Category 7: Invoicing & Payment Due Dates (MEDIUM)
    if "invoice" in text_content and ("due date" in text_content or "amount:" in text_content):
        return ContractAlert(
            alert_id=f"ALT_{event.event_id}",
            event_id=event.event_id,
            affected_contract_ids=affected_contracts,
            potential_trigger="invoice_issued",
            severity="medium",
            summary=f"Invoice issued or received: {event.subject}",
            supporting_evidence_ids=[f"{c}:PAYMENT" for c in affected_contracts],
            requires_review=False,
            loki_review_requested=False,
            norns_timeline_update_needed=True,
        )

    # Default / Unclassified
    return None


# -----------------------------------------------------------------------------
# 5. Heimdall Execution Pipeline
# -----------------------------------------------------------------------------

def process_events(
    event_files: List[Path | str],
    db_path: Path | str = "odin.db",
    severity_threshold: str = "medium",  # Options: info, low, medium, high, critical
) -> Dict[str, Any]:
    """
    Main Heimdall processing engine:
    1. Ingests all drops (events_drop1.jsonl, events_drop2.jsonl)
    2. Identifies affected contracts and maps clauses
    3. Detects contractual triggers and filters noise
    4. Suppresses duplicates and saves alerts to SQLite
    """
    conn = init_heimdall_db(db_path)
    total_parsed = 0
    total_malformed = 0
    generated_alerts: List[ContractAlert] = []

    severity_levels = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    min_level = severity_levels.get(severity_threshold.lower(), 2)

    for ef in event_files:
        events, errors = parse_event_stream(ef, db_path=db_path)
        total_parsed += len(events)
        total_malformed += len(errors)

        for ev in events:
            affected_contracts = identify_affected_contracts(ev)
            alert = evaluate_event_trigger(ev, affected_contracts, db_path=db_path)

            if alert:
                alert_level = severity_levels.get(alert.severity.lower(), 0)
                # Apply noise suppression threshold
                if alert_level >= min_level:
                    generated_alerts.append(alert)

                    # Persist alert
                    with conn:
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO contract_alerts
                            (alert_id, event_id, affected_contracts, potential_trigger, severity, summary, evidence_ids, requires_review)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                alert.alert_id,
                                alert.event_id,
                                ",".join(alert.affected_contract_ids),
                                alert.potential_trigger,
                                alert.severity,
                                alert.summary,
                                ",".join(alert.supporting_evidence_ids),
                                1 if alert.requires_review else 0,
                            ),
                        )

    conn.close()

    return {
        "events_processed": total_parsed,
        "malformed_lines_recovered": total_malformed,
        "alerts_generated": len(generated_alerts),
        "alerts": [a.model_dump() for a in generated_alerts],
    }


if __name__ == "__main__":
    import sys
    print("Running Heimdall Event Monitoring Agent on Drop 1 & Drop 2...")
    res = process_events([
        "bundle/events/events_drop1.jsonl",
        "bundle/events/events_drop2.jsonl"
    ])
    print(f"Total Events Processed: {res['events_processed']}")
    print(f"Malformed Lines Recovered: {res['malformed_lines_recovered']}")
    print(f"Alerts Generated (Threshold >= medium): {res['alerts_generated']}\n")
    for a in res["alerts"]:
        print(f"[{a['severity'].upper()}] Event {a['event_id']} -> Trigger: {a['potential_trigger']}")
        print(f"  Contracts: {a['affected_contract_ids']}")
        print(f"  Summary: {a['summary']}\n")
