"""
Tests for Heimdall - Event Monitoring Agent.
"""

from pathlib import Path
import sqlite3
import pytest

import odin_heimdall
from odin_heimdall import (
    Event,
    ContractAlert,
    parse_event_stream,
    identify_affected_contracts,
    evaluate_event_trigger,
    process_events,
)


@pytest.fixture
def test_db(tmp_path):
    return tmp_path / "test_heimdall.db"


def test_parse_event_stream_and_duplicate_handling(tmp_path, test_db):
    """Test parsing events, duplicate suppression, and malformed line recovery."""
    event_file = tmp_path / "test_events.jsonl"
    content = (
        '{"event_id": "EV-01", "timestamp": "2027-01-01T00:00:00Z", "channel": "email", "from": "A", "to": "B", "subject": "Test", "body": "Body"}\n'
        '{"event_id": "EV-01", "timestamp": "2027-01-01T00:00:00Z", "channel": "email", "from": "A", "to": "B", "subject": "Duplicate", "body": "Body"}\n'
        'THIS IS A MALFORMED LINE\n'
        '{"event_id": "EV-02", "timestamp": "2027-01-02T00:00:00Z", "channel": "email", "from": "A", "to": "B", "subject": "Test 2", "body": "Body 2"}\n'
    )
    event_file.write_text(content, encoding="utf-8")

    events, errors = parse_event_stream(event_file, db_path=test_db)
    assert len(events) == 2  # EV-01 and EV-02 (duplicate EV-01 suppressed)
    assert events[0].event_id == "EV-01"
    assert events[1].event_id == "EV-02"
    assert len(errors) == 1  # 1 malformed line captured


def test_contract_mapping():
    """Test mapping events to affected contracts by participants and subjects."""
    ev_harbor = Event(
        event_id="T-1",
        timestamp="2027-02-10T00:00:00Z",
        channel="email",
        sender="Nora Chen, Harbor Retail",
        recipient="Raj Subramanian, Quillhub",
        subject="Uptime concerns",
        body="Regarding QH-HARBOR agreement",
    )
    contracts = identify_affected_contracts(ev_harbor)
    assert "B1" in contracts or "B2" in contracts

    ev_strato = Event(
        event_id="T-2",
        timestamp="2027-03-04T00:00:00Z",
        channel="status_log",
        sender="StratoNode Status Page",
        recipient="All Customers",
        subject="Service Disruption",
        body="CloudCore us-west-2 failure",
    )
    contracts_strato = identify_affected_contracts(ev_strato)
    assert "A1" in contracts_strato


def test_noise_reduction_suppression():
    """Test that marketing and newsletter events are suppressed."""
    promo_ev = Event(
        event_id="T-3",
        timestamp="2027-03-10T00:00:00Z",
        channel="email",
        sender="marketing@stratonode.com",
        recipient="newsletter@quillhub.io",
        subject="StratoNode Q1 Product Update - New GPU Instances",
        body="Register here for webinar. Unsubscribe link below.",
    )
    alert = evaluate_event_trigger(promo_ev, ["A1"])
    assert alert is None  # Suppressed


def test_sla_failure_alert_trigger():
    """Test that severe downtime generates critical/high alerts requesting Loki & Norns review."""
    outage_ev = Event(
        event_id="T-4",
        timestamp="2027-05-02T00:00:00Z",
        channel="status_log",
        sender="StratoNode Status",
        recipient="All Customers",
        subject="Service Disruption - CloudCore Networking",
        body="Duration: 505 minutes. Complete failure across us-west-2.",
    )
    alert = evaluate_event_trigger(outage_ev, ["A1", "A2"])
    assert alert is not None
    assert alert.severity in ["high", "critical"]
    assert alert.potential_trigger == "service_level_failure"
    assert alert.loki_review_requested is True
    assert alert.norns_timeline_update_needed is True


def test_full_bundle_event_processing(test_db):
    """Test Heimdall against Drop 1 and Drop 2 event streams from bundle."""
    drop1 = Path("bundle/events/events_drop1.jsonl")
    drop2 = Path("bundle/events/events_drop2.jsonl")
    assert drop1.exists() and drop2.exists()

    res = process_events([drop1, drop2], db_path=test_db, severity_threshold="medium")
    assert res["events_processed"] == 18
    assert res["malformed_lines_recovered"] == 1
    assert res["alerts_generated"] >= 10

    # Ensure formal credit claim E-17 generated critical alert
    credit_alert = next((a for a in res["alerts"] if a["event_id"] == "E-17"), None)
    assert credit_alert is not None
    assert credit_alert["severity"] == "critical"
    assert credit_alert["potential_trigger"] == "formal_sla_credit_claim"

