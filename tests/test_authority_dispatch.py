"""Phase 11.5 — authority dispatch layer regression tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.deterministic

import re

from services.recommendation.query_recommendation_intent import classify_query_recommendation_intent
from services.routing.authority_dispatch import consult_authority_dispatch
from services.routing.unified_intent_router import classify_unified_intent

_FORBIDDEN_KERNEL = re.compile(
    r"\b(?:operational\s+synthesis|viabl(?:e|ity)\s+with\s+compromises|approved\s+shortlist)\b",
    re.I,
)


def _dispatch(query: str):
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    return consult_authority_dispatch(query, qri=qri, unified_route=route, context={"db": None})


def test_comparison_dispatch_g650_vs_falcon_8x():
    query = "G650 vs Falcon 8X"
    result = _dispatch(query)
    assert result is not None
    assert result.dispatch_kind == "comparison"
    assert result.data_used.get("authority_dispatch_safety_fallback") is None
    assert (result.data_used.get("comparison_v2") or {}).get("status") == "OK"
    assert "G650" in result.answer or "Gulfstream" in result.answer
    assert "Falcon 8X" in result.answer or "8X" in result.answer
    assert "Verified catalog comparison" in result.answer
    assert not _FORBIDDEN_KERNEL.search(result.answer)
    assert result.data_used.get("authority_dispatch_kind") == "comparison"


def test_comparison_dispatch_compare_phrasing():
    query = "Compare G650 vs Falcon 8X"
    result = _dispatch(query)
    assert result is not None
    assert result.dispatch_kind == "comparison"
    assert not _FORBIDDEN_KERNEL.search(result.answer)


def test_comparison_dispatch_g650_vs_global_7500_returns_catalog_answer():
    """Phase 34.2 — structured acceptance; verdict may include INSUFFICIENT_DATA text."""
    query = "G650 vs Global 7500"
    result = _dispatch(query)
    assert result is not None
    assert result.dispatch_kind == "comparison"
    assert result.data_used.get("authority_dispatch_safety_fallback") is None
    assert (result.data_used.get("comparison_v2") or {}).get("status") == "OK"
    assert "Verified catalog comparison" in result.answer
    assert "VERDICT" in result.answer.upper()
    assert "Global 7500" in result.answer or "7500" in result.answer


def test_comparison_dispatch_g650_vs_g700_catalog_ok():
    """Phase 34.4 — G700 resolves to Gulfstream G700 in comparison registry."""
    query = "G650 vs G700"
    result = _dispatch(query)
    assert result is not None
    assert result.dispatch_kind == "comparison"
    assert result.data_used.get("authority_dispatch_safety_fallback") is None
    assert (result.data_used.get("comparison_v2") or {}).get("status") == "OK"
    assert "VERDICT" in result.answer.upper()
    models = result.data_used.get("authority_dispatch_models") or []
    assert len(models) >= 2
    assert any("G700" in m or "G650" in m for m in models)


def test_comparison_dispatch_g650_vs_longitude_catalog_ok():
    """Phase 34.4 — bare Longitude resolves to Citation Longitude."""
    query = "G650 vs Longitude"
    result = _dispatch(query)
    assert result is not None
    assert result.dispatch_kind == "comparison"
    assert result.data_used.get("authority_dispatch_safety_fallback") is None
    assert (result.data_used.get("comparison_v2") or {}).get("status") == "OK"
    assert "VERDICT" in result.answer.upper()


def test_comparison_dispatch_success_sets_models_not_fallback():
    query = "G650 vs Falcon 8X"
    result = _dispatch(query)
    assert result is not None
    assert result.data_used.get("authority_dispatch_safety_fallback") is None
    models = result.data_used.get("authority_dispatch_models") or []
    assert len(models) >= 2
    assert result.progress_step == "path_authority_dispatch_comparison"


def test_alternative_dispatch_longitude_shorthand():
    query = "Alternatives to Longitude"
    result = _dispatch(query)
    assert result is not None
    assert result.dispatch_kind == "alternative"
    assert "Longitude" in result.answer or "Citation" in result.answer
    assert "tier-peer" in result.answer.lower() or "alternatives" in result.answer.lower()
    assert not _FORBIDDEN_KERNEL.search(result.answer)
    assert result.data_used.get("authority_dispatch_target")


def test_buy_decision_dispatch_latitude_price():
    query = "2016 Latitude $10M good deal?"
    result = _dispatch(query)
    assert result is not None
    assert result.dispatch_kind == "buy_decision"
    assert "Latitude" in result.answer or "Citation" in result.answer
    assert "Verdict:" in result.answer
    assert "Market Reality:" in result.answer
    assert not _FORBIDDEN_KERNEL.search(result.answer)


def test_mission_query_not_dispatched_to_comparison():
    query = (
        "I need a jet for 2000 nm legs, 6 passengers, runway under 5000 ft. "
        "What should I buy?"
    )
    result = _dispatch(query)
    assert result is None


def test_ambiguous_query_falls_through():
    query = "Tell me about business aviation trends"
    result = _dispatch(query)
    assert result is None


def test_unresolved_comparison_returns_safety_fallback():
    query = "G650 vs UnknownJetXYZ"
    result = _dispatch(query)
    assert result is not None
    assert result.dispatch_kind == "comparison"
    assert "Insufficient verified data" in result.answer
    assert result.data_used.get("authority_dispatch_safety_fallback") == "comparison"


def test_budget_filtered_comparison_fail_closed():
    query = "G650 vs Falcon 8X vs Global 7500 under $30M"
    result = _dispatch(query)
    assert result is not None
    assert result.dispatch_kind == "comparison"
    assert result.data_used.get("authority_dispatch_safety_fallback") == "comparison"
    assert "Insufficient verified data" in result.answer
