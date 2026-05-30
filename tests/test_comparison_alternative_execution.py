"""Phase 5 Step 3 — comparison and alternative execution tests."""

import re

from services.comparison.alternative_pipeline_responder import (
    is_alternative_execution_query,
    is_explicit_comparison_query,
    respond_aircraft_alternative,
)
from services.comparison.comparison_pipeline_v2_responder import respond_aircraft_comparison
from services.routing.unified_intent_execution import (
    should_enforce_alternative_path,
    should_enforce_comparison_path,
)
from services.routing.unified_intent_router import UnifiedIntent, classify_unified_intent

_FORBIDDEN = re.compile(
    r"\b(?:good\s+fit|operational\s+synthesis|approved\s+shortlist|shortlist|recommend)\b",
    re.I,
)


def test_explicit_comparison_query_detection():
    assert is_explicit_comparison_query("Compare Longitude vs Legacy 600") is True
    assert is_explicit_comparison_query("Lower-cost alternative to Falcon 8X") is False


def test_alternative_query_detection():
    assert is_alternative_execution_query("Credible alternatives to a Gulfstream G650") is True
    assert is_alternative_execution_query("Compare G650 vs Falcon 8X") is False


def test_comparison_responder_structured_contrast():
    answer = respond_aircraft_comparison("Compare Gulfstream G650ER vs Global 7500")
    assert answer
    assert "Verified catalog comparison" in answer
    assert "G650ER" in answer or "G650" in answer
    assert "Global 7500" in answer
    assert not _FORBIDDEN.search(answer)
    assert "GOOD FIT" not in answer.upper()


def test_alternative_responder_tier_peers():
    answer = respond_aircraft_alternative("What are credible alternatives to a Gulfstream G650?")
    assert answer
    assert "G650" in answer or "Gulfstream" in answer
    assert "tier-peer" in answer.lower() or "alternatives" in answer.lower()
    assert not _FORBIDDEN.search(answer)
    assert "rank" not in answer.lower()


def test_should_enforce_comparison_path():
    q = "Compare Challenger 650 vs Praetor 600"
    route = classify_unified_intent(q)
    assert route.intent == UnifiedIntent.OTHER
    assert should_enforce_comparison_path(route, q) is True


def test_should_enforce_alternative_not_comparison():
    q = "What are credible alternatives to a Gulfstream G650?"
    route = classify_unified_intent(q)
    assert should_enforce_alternative_path(route, q) is True
    assert should_enforce_comparison_path(route, q) is False


def test_fact_query_does_not_enforce_comparison_or_alternative():
    route = classify_unified_intent("How many seats does a Falcon 8X have?")
    assert should_enforce_comparison_path(route, "How many seats does a Falcon 8X have?") is False
    assert should_enforce_alternative_path(route, "How many seats does a Falcon 8X have?") is False
