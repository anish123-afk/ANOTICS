"""
Deterministic Exposure and Liability Calculation Engine for Norns.
Calculates confirmed exposure, penalties, and upstream recovery balances strictly using arithmetic formulas.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from .schemas import ExposureType, ObligationStatus


class ExposureEngine:
    @staticmethod
    def calculate_obligation_exposure(
        status: ObligationStatus,
        contract_amount: float,
        total_paid: float,
        outstanding_amount: float,
        penalty_rate_percent: float = 0.0,
        days_overdue: int = 0
    ) -> Tuple[ExposureType, float, float]:
        """
        Determines financial exposure type and calculates potential penalties.
        Returns: (exposure_type, potential_penalty, total_financial_exposure)
        """
        penalty = 0.0
        
        if status == ObligationStatus.COMPLETED:
            return ExposureType.CONFIRMED_DUE, 0.0, 0.0

        if status == ObligationStatus.OVERDUE:
            if penalty_rate_percent > 0.0 and outstanding_amount > 0.0:
                # E.g., 5% late fee or compounding monthly interest
                penalty = round(outstanding_amount * (penalty_rate_percent / 100.0), 2)
                exposure = outstanding_amount + penalty
                return ExposureType.POTENTIAL_PENALTY, penalty, exposure
            else:
                return ExposureType.CONFIRMED_DUE, 0.0, outstanding_amount

        if status in (ObligationStatus.UPCOMING, ObligationStatus.DUE):
            return ExposureType.CONFIRMED_DUE, 0.0, outstanding_amount

        # Unknown / No due date
        if contract_amount > 0:
            return ExposureType.CONDITIONAL_LIABILITY, 0.0, contract_amount
        return ExposureType.UNKNOWN, 0.0, 0.0
