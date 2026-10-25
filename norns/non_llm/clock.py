"""
Clock Loader for Norns.
Loads simulated start_date and checkpoint dates from clock.json.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import List, Tuple, Union


def load_clock_checkpoints(clock_file_path: Union[str, Path]) -> Tuple[date, List[date]]:
    """
    Load clock.json and extract start date and chronological checkpoint dates.
    
    Supports formats:
      - {"start_date": "2027-01-15", "checkpoints": ["2027-03-15", "2027-06-01", "2027-09-15"]}
      - {"checkpoints": [{"name": "t1", "date": "2027-03-15"}, ...]}
    """
    path = Path(clock_file_path)
    if not path.is_file():
        raise FileNotFoundError(f"clock.json not found at {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    
    raw_start = data.get("start_date")
    start_d = datetime.fromisoformat(raw_start).date() if raw_start else None

    raw_checkpoints = data.get("checkpoints", [])
    checkpoints: List[date] = []

    for item in raw_checkpoints:
        if isinstance(item, str):
            checkpoints.append(datetime.fromisoformat(item).date())
        elif isinstance(item, dict) and "date" in item:
            checkpoints.append(datetime.fromisoformat(item["date"]).date())

    checkpoints.sort()
    if not start_d and checkpoints:
        start_d = checkpoints[0]

    return start_d, checkpoints
