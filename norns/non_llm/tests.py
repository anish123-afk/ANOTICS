"""
Test suite for Norns Non-LLM Deterministic Engine and Timeline Agent.
Uses standard library unittest so it runs seamlessly across Python versions without extra test runners.
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

from norns.non_llm import (
    HolidayCalendar,
    PaymentReconciler,
    ExposureEngine,
    NornsTimelineAgent,
    ObligationItem,
    ObligationStatus,
    ExposureType,
    load_clock_checkpoints,
)


class TestNornsClockAndCalendar(unittest.TestCase):
    def setUp(self):
        self.clock_file = ANOTICS_DIR / "clock.json"
        self.holidays_file = ANOTICS_DIR / "bundle" / "data" / "holidays.csv"
        self.calendar = HolidayCalendar(self.holidays_file)

    def test_clock_loading(self):
        start_d, checkpoints = load_clock_checkpoints(self.clock_file)
        self.assertEqual(start_d, date(2027, 1, 15))
        self.assertEqual(len(checkpoints), 3)
        self.assertEqual(checkpoints, [date(2027, 3, 15), date(2027, 6, 1), date(2027, 9, 15)])

    def test_holiday_detection(self):
        # Christmas 2026-12-25 is observed by All
        self.assertTrue(self.calendar.is_holiday(date(2026, 12, 25)))
        # Regular business day
        self.assertFalse(self.calendar.is_holiday(date(2026, 12, 10)))

    def test_business_day_math(self):
        # 2026-12-24 (Christmas Eve), 2026-12-25 (Christmas Day), 26-27 (Weekend)
        # Adding 1 business day from 2026-12-23 should jump to 2026-12-28
        res = self.calendar.add_business_days(date(2026, 12, 23), 1, observed_by="Quillhub")
        self.assertEqual(res, date(2026, 12, 28))

    def test_notice_deadline_calculation(self):
        # Event on 2027-01-05, notice required 5 calendar days prior -> 2026-12-31 (Quillhub holiday)
        # Notice rolls back to preceding business day: 2026-12-30
        deadline = self.calendar.get_effective_notice_deadline(
            event_or_renewal_date=date(2027, 1, 5),
            notice_days=5,
            is_business_days=False,
            observed_by="Quillhub",
        )
        self.assertEqual(deadline, date(2026, 12, 30))


class TestPaymentReconciler(unittest.TestCase):
    def setUp(self):
        self.ledger_file = ANOTICS_DIR / "bundle" / "data" / "payments_ledger.csv"
        self.reconciler = PaymentReconciler(self.ledger_file)

    def test_reconcile_as_of_checkpoint(self):
        # Harbor Retail payment on 2026-09-20 ($40,000)
        # As of 2026-08-01, paid should be 0.0
        paid, out, ev, _ = self.reconciler.reconcile("B1", "OB-TEST", 40000.0, date(2026, 8, 1))
        self.assertEqual(paid, 0.0)
        self.assertEqual(out, 40000.0)

        # As of 2026-10-01, should include Sept payment ($40,000)
        paid, out, ev, _ = self.reconciler.reconcile("B1", "OB-TEST", 40000.0, date(2026, 10, 1))
        self.assertGreaterEqual(paid, 40000.0)
        self.assertEqual(out, 0.0)
        self.assertGreater(len(ev), 0)


class TestExposureEngine(unittest.TestCase):
    def test_completed_obligation(self):
        exp_type, penalty, total = ExposureEngine.calculate_obligation_exposure(
            status=ObligationStatus.COMPLETED,
            contract_amount=5000.0,
            total_paid=5000.0,
            outstanding_amount=0.0,
        )
        self.assertEqual(exp_type, ExposureType.CONFIRMED_DUE)
        self.assertEqual(penalty, 0.0)
        self.assertEqual(total, 0.0)

    def test_overdue_penalty_calculation(self):
        exp_type, penalty, total = ExposureEngine.calculate_obligation_exposure(
            status=ObligationStatus.OVERDUE,
            contract_amount=10000.0,
            total_paid=0.0,
            outstanding_amount=10000.0,
            penalty_rate_percent=5.0,
            days_overdue=15,
        )
        self.assertEqual(exp_type, ExposureType.POTENTIAL_PENALTY)
        self.assertEqual(penalty, 500.0)
        self.assertEqual(total, 10500.0)


class TestNornsTimelineTransitions(unittest.TestCase):
    def setUp(self):
        self.agent = NornsTimelineAgent(
            clock_file=ANOTICS_DIR / "clock.json",
            holidays_file=ANOTICS_DIR / "bundle" / "data" / "holidays.csv",
            payments_file=ANOTICS_DIR / "bundle" / "data" / "payments_ledger.csv",
        )

    def test_obligation_status_transitions_across_checkpoints(self):
        # Target obligation due on 2027-05-01 (Between checkpoint 1 and checkpoint 2)
        ob = ObligationItem(
            id="OB-CUSTOM-RENEWAL",
            contract_id="CUSTOM",
            description="Renewal decision notice",
            due_date=date(2027, 5, 1),
            contract_amount=0.0,
            evidence_ids=["CUSTOM-1.0"],
        )

        # CP1: 2027-03-15 -> Before 2027-05-01 -> UPCOMING
        st1 = self.agent.evaluate_obligation_at_date(ob, date(2027, 3, 15))
        self.assertEqual(st1.status, ObligationStatus.UPCOMING)
        self.assertGreater(st1.days_remaining, 0)

        # CP2: 2027-06-01 -> After 2027-05-01 -> OVERDUE
        st2 = self.agent.evaluate_obligation_at_date(ob, date(2027, 6, 1))
        self.assertEqual(st2.status, ObligationStatus.OVERDUE)
        self.assertLess(st2.days_remaining, 0)


if __name__ == "__main__":
    unittest.main()
