"""
Deterministic Payment Reconciliation Engine for Norns.
Loads payments_ledger.csv and performs deterministic matching and reconciliation against obligations.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .schemas import PaymentRecord


class PaymentReconciler:
    def __init__(self, ledger_csv_path: Optional[Union[str, Path]] = None):
        self.payments: List[PaymentRecord] = []
        if ledger_csv_path:
            self.load_ledger(ledger_csv_path)

    def load_ledger(self, ledger_csv_path: Union[str, Path]) -> None:
        path = Path(ledger_csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Payments ledger not found at {path}")

        self.payments = []
        with path.open(mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            idx = 1
            for row in reader:
                d_str = row.get("date", "").strip()
                if not d_str:
                    continue
                d = date.fromisoformat(d_str)
                amt = float(row.get("amount", "0").replace(",", ""))
                ref = row.get("reference", "").strip()
                
                # Derive contract ID from reference code (e.g., QH-STRATO-2026-001 -> A1/STRATO)
                contract_hint = ""
                if "STRATO" in ref:
                    contract_hint = "A1"
                elif "HARBOR" in ref:
                    contract_hint = "B1"
                elif "PIXEL" in ref:
                    contract_hint = "D1"
                elif "LOGIX" in ref:
                    contract_hint = "C1"

                rec = PaymentRecord(
                    payment_id=f"pay_{idx:03d}",
                    contract_id=contract_hint,
                    date=d,
                    from_party=row.get("from", "").strip(),
                    to_party=row.get("to", "").strip(),
                    amount=amt,
                    reference=ref,
                    status=row.get("status", "unknown").strip(),
                    evidence_id=f"ledger_row_{idx}"
                )
                self.payments.append(rec)
                idx += 1

    def reconcile(
        self,
        contract_id: str,
        obligation_id: str,
        expected_amount: float,
        as_of_date: date
    ) -> Tuple[float, float, List[str], List[PaymentRecord]]:
        """
        Calculates payments made on or before as_of_date matching the contract.
        Returns: (total_paid, outstanding_amount, evidence_ids, matching_records)
        """
        matching: List[PaymentRecord] = []
        evidence_ids: List[str] = []
        total_paid = 0.0

        for p in self.payments:
            if p.date <= as_of_date:
                # Direct match by contract_id or obligation_id mentioned in reference
                if (p.contract_id and p.contract_id == contract_id) or (obligation_id in p.reference):
                    if p.status.lower() == "paid":
                        total_paid += p.amount
                        evidence_ids.append(p.evidence_id)
                        matching.append(p)
                    elif p.status.lower() in ("overdue", "pending"):
                        evidence_ids.append(p.evidence_id)

        outstanding = max(0.0, expected_amount - total_paid) if expected_amount > 0 else 0.0
        return total_paid, outstanding, evidence_ids, matching
