"""
Deterministic Calendar and Holiday Engine for Norns.
Uses holidays.csv to adjust deadlines based on business day / calendar day rules.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Union


class HolidayCalendar:
    def __init__(self, holidays_csv_path: Optional[Union[str, Path]] = None):
        self.holidays: Dict[date, Dict[str, str]] = {}
        if holidays_csv_path:
            self.load_holidays(holidays_csv_path)

    def load_holidays(self, holidays_csv_path: Union[str, Path]) -> None:
        path = Path(holidays_csv_path)
        if not path.is_file():
            return
        with path.open(mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d_str = row.get("date", "").strip()
                if not d_str:
                    continue
                d = date.fromisoformat(d_str)
                self.holidays[d] = {
                    "name": row.get("name", "").strip(),
                    "observed_by": row.get("observed_by", "All").strip(),
                }

    def is_holiday(self, d: date, observed_by: str = "All") -> bool:
        if d not in self.holidays:
            return False
        obs = self.holidays[d]["observed_by"]
        return obs in ("All", observed_by) or observed_by == "All"

    def is_business_day(self, d: date, observed_by: str = "All") -> bool:
        # 5 = Saturday, 6 = Sunday
        if d.weekday() >= 5:
            return False
        return not self.is_holiday(d, observed_by)

    def add_calendar_days(self, start_date: date, days: int) -> date:
        return start_date + timedelta(days=days)

    def add_business_days(self, start_date: date, business_days: int, observed_by: str = "All") -> date:
        current = start_date
        step = 1 if business_days >= 0 else -1
        remaining = abs(business_days)

        while remaining > 0:
            current += timedelta(days=step)
            if self.is_business_day(current, observed_by):
                remaining -= 1
        return current

    def get_effective_notice_deadline(
        self,
        event_or_renewal_date: date,
        notice_days: int,
        is_business_days: bool = False,
        observed_by: str = "All"
    ) -> date:
        """
        Calculates the deadline date by which notice must be given prior to an event/renewal.
        If is_business_days is True, steps back counting only business days.
        """
        if not is_business_days:
            target = event_or_renewal_date - timedelta(days=notice_days)
            # If notice deadline lands on non-business day, adjust to previous business day
            while not self.is_business_day(target, observed_by):
                target -= timedelta(days=1)
            return target
        else:
            return self.add_business_days(event_or_renewal_date, -notice_days, observed_by)
