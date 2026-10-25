"""
Norns Timeline Agent - Models and Schemas (Non-LLM Core).
Strictly typed Pydantic models for obligations, payments, exposures, and timeline snapshots.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ObligationStatus(str, Enum):
    UPCOMING = "upcoming"
    DUE = "due"
    OVERDUE = "overdue"
    COMPLETED = "completed"
    NO_DUE_DATE = "no_due_date"


class ExposureType(str, Enum):
    CONFIRMED_DUE = "confirmed_amount_due"
    POTENTIAL_PENALTY = "potential_penalty"
    CONDITIONAL_LIABILITY = "conditional_liability"
    POTENTIAL_UPSTREAM_RECOVERY = "potential_upstream_recovery"
    UNRECOVERABLE_EXPOSURE = "unrecoverable_exposure"
    UNKNOWN = "unknown_financial_exposure"


class EvidenceCitation(BaseModel):
    document_id: str
    clause_id: Optional[str] = None
    source_text: str = ""
    evidence_id: Optional[str] = None


class ObligationItem(BaseModel):
    id: str
    contract_id: str
    description: str
    responsible_party: Optional[str] = None
    due_date: Optional[date] = None
    contract_amount: float = 0.0
    penalty_rate_percent: float = 0.0
    evidence_ids: List[str] = Field(default_factory=list)


class PaymentRecord(BaseModel):
    payment_id: str
    contract_id: str
    date: date
    from_party: str
    to_party: str
    amount: float
    reference: str
    status: str
    evidence_id: str


class ObligationTimelineState(BaseModel):
    obligation_id: str
    contract_id: str
    description: str
    responsible_party: Optional[str] = None
    due_date: Optional[str] = None
    simulated_date: str
    status: ObligationStatus
    days_remaining: Optional[int] = None
    contract_amount: float = 0.0
    total_paid: float = 0.0
    outstanding_amount: float = 0.0
    potential_penalty: float = 0.0
    exposure_type: ExposureType
    evidence_ids: List[str] = Field(default_factory=list)


class CheckpointSnapshot(BaseModel):
    simulated_date: str
    checkpoint_index: int
    total_obligations: int
    status_counts: Dict[str, int]
    total_paid: float
    total_outstanding: float
    total_penalties: float
    obligations: List[ObligationTimelineState]
    generated_at: str
