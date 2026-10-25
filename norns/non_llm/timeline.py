"""
Norns - Deterministic Timeline Agent Core.
Coordinates clock checkpoints, calendar adjustments, payment reconciliation, and exposure math.
Zero LLM calls. Fully testable, deterministic, and auditable.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .calendar import HolidayCalendar
from .clock import load_clock_checkpoints
from .exposure import ExposureEngine
from .payments import PaymentReconciler
from .schemas import (
    CheckpointSnapshot,
    EvidenceCitation,
    ExposureType,
    ObligationItem,
    ObligationStatus,
    ObligationTimelineState,
)


class NornsTimelineAgent:
    """
    Norns Timeline Agent Engine.
    Executes time-travel across clock checkpoints, evaluating obligations deterministically.
    """

    def __init__(
        self,
        clock_file: Union[str, Path],
        holidays_file: Optional[Union[str, Path]] = None,
        payments_file: Optional[Union[str, Path]] = None,
    ):
        self.clock_path = Path(clock_file)
        self.start_date, self.checkpoints = load_clock_checkpoints(self.clock_path)
        self.calendar = HolidayCalendar(holidays_file)
        self.reconciler = PaymentReconciler(payments_file)
        self.exposure_engine = ExposureEngine()

    def evaluate_obligation_at_date(
        self,
        obligation: ObligationItem,
        simulated_date: date
    ) -> ObligationTimelineState:
        """
        Determines the state of an obligation at a specific checkpoint date.
        """
        # 1. Payment Reconciliation
        total_paid, outstanding, payment_evidences, _ = self.reconciler.reconcile(
            contract_id=obligation.contract_id,
            obligation_id=obligation.id,
            expected_amount=obligation.contract_amount,
            as_of_date=simulated_date,
        )

        # 2. Determine Lifecycle Status
        if not obligation.due_date:
            status = ObligationStatus.NO_DUE_DATE
            days_remaining = None
        else:
            diff_days = (obligation.due_date - simulated_date).days
            days_remaining = diff_days

            if obligation.contract_amount > 0 and outstanding <= 0.0 and total_paid >= obligation.contract_amount:
                status = ObligationStatus.COMPLETED
            elif diff_days > 0:
                status = ObligationStatus.UPCOMING
            elif diff_days == 0:
                status = ObligationStatus.DUE
            else:
                if obligation.contract_amount > 0 and outstanding <= 0.0:
                    status = ObligationStatus.COMPLETED
                else:
                    status = ObligationStatus.OVERDUE

        # 3. Calculate Exposure & Penalties
        days_overdue = abs(days_remaining) if (days_remaining is not None and days_remaining < 0) else 0
        exp_type, penalty, total_exp = self.exposure_engine.calculate_obligation_exposure(
            status=status,
            contract_amount=obligation.contract_amount,
            total_paid=total_paid,
            outstanding_amount=outstanding,
            penalty_rate_percent=obligation.penalty_rate_percent,
            days_overdue=days_overdue,
        )

        all_evidences = list(set(obligation.evidence_ids + payment_evidences))

        return ObligationTimelineState(
            obligation_id=obligation.id,
            contract_id=obligation.contract_id,
            description=obligation.description,
            responsible_party=obligation.responsible_party,
            due_date=obligation.due_date.isoformat() if obligation.due_date else None,
            simulated_date=simulated_date.isoformat(),
            status=status,
            days_remaining=days_remaining,
            contract_amount=obligation.contract_amount,
            total_paid=total_paid,
            outstanding_amount=outstanding,
            potential_penalty=penalty,
            exposure_type=exp_type,
            evidence_ids=all_evidences,
        )

    def generate_checkpoint_snapshot(
        self,
        checkpoint_index: int,
        obligations: List[ObligationItem]
    ) -> CheckpointSnapshot:
        """
        Generates a complete frozen snapshot for a specific checkpoint index.
        """
        if checkpoint_index < 0 or checkpoint_index >= len(self.checkpoints):
            raise IndexError(f"Checkpoint index {checkpoint_index} out of bounds (0..{len(self.checkpoints)-1})")

        sim_date = self.checkpoints[checkpoint_index]
        states: List[ObligationTimelineState] = []
        status_counts: Dict[str, int] = {s.value: 0 for s in ObligationStatus}
        tot_paid = 0.0
        tot_outstanding = 0.0
        tot_penalties = 0.0

        for ob in obligations:
            state = self.evaluate_obligation_at_date(ob, sim_date)
            states.append(state)
            status_counts[state.status.value] += 1
            tot_paid += state.total_paid
            tot_outstanding += state.outstanding_amount
            tot_penalties += state.potential_penalty

        return CheckpointSnapshot(
            simulated_date=sim_date.isoformat(),
            checkpoint_index=checkpoint_index,
            total_obligations=len(obligations),
            status_counts=status_counts,
            total_paid=round(tot_paid, 2),
            total_outstanding=round(tot_outstanding, 2),
            total_penalties=round(tot_penalties, 2),
            obligations=states,
            generated_at=datetime.now().isoformat(),
        )

    def generate_all_snapshots(
        self,
        obligations: List[ObligationItem]
    ) -> List[CheckpointSnapshot]:
        """
        Runs the simulation across all checkpoints in clock.json and returns a list of snapshots.
        """
        return [
            self.generate_checkpoint_snapshot(idx, obligations)
            for idx in range(len(self.checkpoints))
        ]
