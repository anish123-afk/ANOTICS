"""
Norns Timeline Agent - LLM Reasoning Agent.
Orchestrates Claude 3.5 Sonnet (with automatic offline fallback) to produce
structured legal risk analyses and worst-case scenario reports.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Union
from ..non_llm.schemas import CheckpointSnapshot
from .client import AnthropicClient
from .mock_llm import MockNornsLLM
from .prompts import SYSTEM_PROMPT, build_checkpoint_user_prompt
from .schemas import (
    MultiCheckpointTimelineReport,
    ObligationNarrative,
    TimelineCheckpointAnalysis,
)


class NornsLLMAgent:
    """
    LLM Intelligence Agent for Norns.
    Takes deterministic CheckpointSnapshot objects and synthesizes verified legal analyses.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        use_mock: bool = False,
    ):
        self.use_mock = use_mock
        self.mock_engine = MockNornsLLM()
        self.client = AnthropicClient(api_key=api_key, model=model)

    def analyze_checkpoint(self, snapshot: CheckpointSnapshot) -> TimelineCheckpointAnalysis:
        """
        Synthesizes an executive legal analysis for a single checkpoint snapshot.
        If use_mock=True or ANTHROPIC_API_KEY is not set, seamlessly uses MockNornsLLM.
        """
        if self.use_mock or not self.client.is_configured:
            return self.mock_engine.analyze_snapshot(snapshot)

        # Prepare JSON string of deterministic inputs
        snapshot_dict = snapshot.model_dump()
        user_prompt = build_checkpoint_user_prompt(json.dumps(snapshot_dict, indent=2))

        try:
            raw_response = self.client.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
            )
            parsed_json = self._extract_json(raw_response)
            return TimelineCheckpointAnalysis.model_validate(parsed_json)
        except Exception as e:
            # Safe degradation with audit note
            analysis = self.mock_engine.analyze_snapshot(snapshot)
            analysis.executive_summary += f" [Note: LLM fallback engaged: {e}]"
            return analysis

    def analyze_all_checkpoints(
        self,
        snapshots: List[CheckpointSnapshot]
    ) -> MultiCheckpointTimelineReport:
        """
        Analyzes full progression across all checkpoints and generates a trend report.
        """
        analyses = [self.analyze_checkpoint(snap) for snap in snapshots]

        # Calculate overall timeline trend
        first_date = snapshots[0].simulated_date if snapshots else "N/A"
        last_date = snapshots[-1].simulated_date if snapshots else "N/A"
        final_overdue = snapshots[-1].status_counts.get("overdue", 0) if snapshots else 0

        if final_overdue > 0:
            trend = (
                f"Contractual risk escalated significantly between {first_date} and {last_date}. "
                f"While early checkpoints were in compliance, {final_overdue} obligation(s) entered OVERDUE "
                f"status by {last_date}, requiring active arbitration and potential breach mitigation."
            )
        else:
            trend = (
                f"Contractual performance remained stable across the simulated window ({first_date} to {last_date}). "
                f"All obligations were settled or maintained within active notice windows."
            )

        unresolved = []
        for a in analyses:
            unresolved.extend(a.missing_information_requests)
        unresolved = list(set(unresolved))

        return MultiCheckpointTimelineReport(
            checkpoints_analyzed=analyses,
            overall_timeline_trend=trend,
            unresolved_legal_questions=unresolved,
        )

    @staticmethod
    def _extract_json(text: str) -> dict:
        """
        Extracts JSON object from LLM response even if wrapped in markdown code blocks.
        """
        text = text.strip()
        # Look for ```json ... ``` blocks
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        # Direct JSON curly brace match
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        return json.loads(text)
