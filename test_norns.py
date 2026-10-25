"""
Root pytest discovery entry point for Norns Test Suite.
Imports all test cases from norns/non_llm/tests.py to enable seamless pytest execution.
"""

from norns.non_llm.tests import (
    TestNornsClockAndCalendar,
    TestPaymentReconciler,
    TestExposureEngine,
    TestNornsTimelineTransitions,
)
from norns.llm.tests import TestNornsLLM

__all__ = [
    "TestNornsClockAndCalendar",
    "TestPaymentReconciler",
    "TestExposureEngine",
    "TestNornsTimelineTransitions",
    "TestNornsLLM",
]
