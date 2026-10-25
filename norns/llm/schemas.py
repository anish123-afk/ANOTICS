"""
Norns Timeline Agent - LLM Schemas.
Structured Pydantic models for legal narrative synthesis, worst-case risk projections,
and feedback requests to Thor.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class ObligationNarrative(BaseModel):
    obligation_id: str
    contract_id: str
    status: str
    summary_explanation: str = Field(
        ...,
        description="Clear, evidence-backed narrative explaining the obligation's legal position as of this checkpoint.",
    )
    legal_basis: str = Field(
        ...,
        description="Precise contractual clauses and dates cited from the deterministic snapshot.",
    )
    cited_evidence_ids: List[str] = Field(
        default_factory=list,
        description="Exact evidence IDs backing this narrative (e.g. clause numbers, ledger rows).",
    )
    worst_case_scenario: Optional[str] = Field(
        None,
        description="What happens if the deadline is missed or payment remains unpaid (penalties, breach, auto-renewal).",
    )


class TimelineCheckpointAnalysis(BaseModel):
    simulated_date: str
    checkpoint_index: int
    executive_summary: str = Field(
        ...,
        description="High-level legal and operational summary of the company's timeline position.",
    )
    total_confirmed_due: float = Field(
        default=0.0,
        description="Exact confirmed debt strictly quoted from deterministic calculations.",
    )
    total_potential_penalties: float = Field(
        default=0.0,
        description="Exact potential penalties strictly quoted from deterministic calculations.",
    )
    critical_alerts: List[ObligationNarrative] = Field(
        default_factory=list,
        description="Detailed legal analysis of overdue, upcoming, or high-risk obligations.",
    )
    recommended_immediate_actions: List[str] = Field(
        default_factory=list,
        description="Concrete, actionable recommendations (e.g. notices to draft, payments to release).",
    )
    missing_information_requests: List[str] = Field(
        default_factory=list,
        description="Targeted questions or re-extraction requests for the Thor agent if obligations lack clarity.",
    )


class MultiCheckpointTimelineReport(BaseModel):
    checkpoints_analyzed: List[TimelineCheckpointAnalysis] = Field(default_factory=list)
    overall_timeline_trend: str = Field(
        ...,
        description="Summary of how contractual risk and exposure evolved from checkpoint 1 to checkpoint 3.",
    )
    unresolved_legal_questions: List[str] = Field(default_factory=list)
