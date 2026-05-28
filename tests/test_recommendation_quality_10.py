"""Pytest wrapper for recommendation quality FINAL 10 suite."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_SUITE = _BACKEND / "evals" / "recommendation_quality_10_suite.json"

_GENERIC_DUMP_RE = re.compile(
    r"\b(?:global\s*7500|g\s*650(?:er)?|gulfstream\s+g\s*650)\b",
    re.I,
)


def _scenarios():
    raw = json.loads(_SUITE.read_text(encoding="utf-8"))
    return list(raw.get("scenarios") or [])


@pytest.mark.parametrize("scenario", _scenarios(), ids=lambda s: s["id"])
def test_recommendation_quality_scenario(scenario):
    from runners.run_recommendation_quality_10 import run_scenario

    result = run_scenario(scenario)
    assert result.grade() in ("PASS", "PARTIAL"), (
        f"{scenario['id']} graded {result.grade()}: "
        f"failed={[k for k, v in result.checks.items() if not v]} errors={result.errors}"
    )


@pytest.mark.parametrize(
    "query",
    [
        "We support Permian Basin drilling and West African offshore operations, but executives also fly Houston–London and Houston–Zurich. Dispatch reliability has suffered under a single-aircraft strategy. What is structurally wrong?",
        "We operate: Arctic gravel strips, Miami Caribbean utilization, Houston energy logistics, London executive HQ, Singapore continuation traffic. Leadership still wants one aircraft category globally. Is that operationally coherent?",
        "Executives fly Los Angeles–Tokyo, New York–London, and Miami–Caribbean, with occasional continuation through Dubai and Singapore. Which routes should actually drive aircraft selection?",
    ],
)
def test_structure_queries_suppress_generic_dump(query: str):
    from runners.run_recommendation_quality_10 import run_scenario

    sc = {"id": "inline", "title": "inline", "query": query, "checks": [], "allows_aircraft": False}
    result = run_scenario(sc)
    assert not _GENERIC_DUMP_RE.search(result.answer or ""), "Generic ULR dump in structure-first query"
    assert result.metrics.get("render_interpretation_only") or not result.metrics.get("recommendations")


def test_explicit_shortlist_query_may_recommend():
    from runners.run_recommendation_quality_10 import run_scenario

    sc = {
        "id": "shortlist",
        "title": "shortlist",
        "query": (
            "I usually fly 8 passengers between San Francisco and Hawaii, occasionally Tokyo, "
            "and care far more about operating economics than prestige. What should I realistically shortlist?"
        ),
        "checks": [],
        "allows_aircraft": True,
        "requires_explicit_recommendation": True,
    }
    result = run_scenario(sc)
    assert result.metrics.get("response_mode") in (
        "recommendation_mode",
        "interpretation_mode",
        "structure_mode",
        None,
    )
    assert "economics" in (result.answer or "").lower() or "operating" in (result.answer or "").lower()
