"""Phase 56 — comparison must be fact-first without acquisition voice."""

from __future__ import annotations

import re

import pytest

from tests.e2e.broker_certification_helpers import broker_certify

_FORBIDDEN = re.compile(
    r"(?is)\b(?:if\s+i\s+were\s+buying\s+today|i\s+would\s+buy|i\s+would\s+focus\s+on)\b"
)


def test_comparison_fact_first_no_executive_voice():
    answer, du, path = broker_certify("G280 vs Longitude", prefer_e2e=False)
    assert path == "layers"
    assert not _FORBIDDEN.search(answer), f"acquisition voice in comparison: {answer[:400]}"
    low = answer.lower()
    assert "range" in low or "cabin" in low or du.get("broker_execution_category") == "comparison"
