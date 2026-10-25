"""
Unit & Integration Tests for Norns LLM Agent.
Verifies prompt construction, schema compliance, JSON parsing, and mock/live handling.
"""

import unittest
from datetime import date
from pathlib import Path
import sys

ANOTICS_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = ANOTICS_DIR.parent
for p in (str(ANOTICS_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from norns.non_llm import NornsTimelineAgent, ObligationItem
from norns.llm import (
    NornsLLMAgent,
    MockNornsLLM,
    TimelineCheckpointAnalysis,
    MultiCheckpointTimelineReport,
    build_checkpoint_user_prompt,
)


class TestNornsLLM(unittest.TestCase):
    def setUp(self):
        self.det_agent = NornsTimelineAgent(
            clock_file=ANOTICS_DIR / "clock.json",
            holidays_file=ANOTICS_DIR / "bundle" / "data" / "holidays.csv",
            payments_file=ANOTICS_DIR / "bundle" / "data" / "payments_ledger.csv",
        )
        self.sample_ob = ObligationItem(
            id="OB-TEST-RENEWAL",
            contract_id="B1",
            description="Harbor Retail Non-Renewal Notice Deadline",
            responsible_party="Quillhub Inc.",
            due_date=date(2027, 7, 1),
            contract_amount=0.0,
            penalty_rate_percent=0.0,
            evidence_ids=["B1-Sec-2.2"],
        )

    def test_prompt_builder(self):
        snapshot = self.det_agent.generate_checkpoint_snapshot(0, [self.sample_ob])
        prompt = build_checkpoint_user_prompt(snapshot.model_dump_json())
        self.assertIn("OB-TEST-RENEWAL", prompt)
        self.assertIn("DETERMINISTIC SNAPSHOT INPUT", prompt)

    def test_mock_analysis_structure(self):
        # Snapshot 2 is after 2027-07-01 -> OVERDUE
        snapshot = self.det_agent.generate_checkpoint_snapshot(2, [self.sample_ob])
        mock_llm = MockNornsLLM()
        analysis = mock_llm.analyze_snapshot(snapshot)

        self.assertIsInstance(analysis, TimelineCheckpointAnalysis)
        self.assertEqual(analysis.simulated_date, "2027-09-15")
        self.assertEqual(len(analysis.critical_alerts), 1)
        alert = analysis.critical_alerts[0]
        self.assertEqual(alert.obligation_id, "OB-TEST-RENEWAL")
        self.assertEqual(alert.status, "overdue")
        self.assertIn("B1-Sec-2.2", alert.legal_basis)
        self.assertIsNotNone(alert.worst_case_scenario)

    def test_agent_multi_checkpoint_pipeline(self):
        agent = NornsLLMAgent(use_mock=True)
        snapshots = self.det_agent.generate_all_snapshots([self.sample_ob])
        report = agent.analyze_all_checkpoints(snapshots)

        self.assertIsInstance(report, MultiCheckpointTimelineReport)
        self.assertEqual(len(report.checkpoints_analyzed), 3)
        self.assertIn("OVERDUE", report.overall_timeline_trend)

    def test_json_extractor_resilience(self):
        # Markdown wrapped
        markdown_json = '```json\n{"simulated_date": "2027-03-15", "checkpoint_index": 0, "executive_summary": "All good", "total_confirmed_due": 0.0, "total_potential_penalties": 0.0, "critical_alerts": [], "recommended_immediate_actions": [], "missing_information_requests": []}\n```'
        parsed = NornsLLMAgent._extract_json(markdown_json)
        self.assertEqual(parsed["simulated_date"], "2027-03-15")

        # Prose wrapped
        prose_json = 'Here is the analysis:\n{"simulated_date": "2027-06-01", "checkpoint_index": 1, "executive_summary": "Pending", "total_confirmed_due": 0.0, "total_potential_penalties": 0.0, "critical_alerts": [], "recommended_immediate_actions": [], "missing_information_requests": []}\nHope this helps!'
        parsed2 = NornsLLMAgent._extract_json(prose_json)
        self.assertEqual(parsed2["simulated_date"], "2027-06-01")


if __name__ == "__main__":
    unittest.main()
