"""Phase 29 — Budget constraint matrix for authority dispatch."""

from __future__ import annotations

import pytest

from services.core.semantic_intent_lock_engine import build_intent_lock
from services.recommendation.query_recommendation_intent import classify_query_recommendation_intent
from services.routing.authority_dispatch import consult_authority_dispatch
from services.routing.intent_conflict_resolution import _apply_budget_filter, build_intent_graph
from services.routing.unified_intent_router import classify_unified_intent



pytestmark = pytest.mark.deterministic
AIRCRAFT = ["G650", "Falcon 8X", "Global 7500", "Longitude", "Challenger 3500"]
BUDGETS_M = [5, 10, 20, 30, 50, 70, 100]

COMPARE_PAIRS = [
    ("G650", "Falcon 8X"),
    ("G650", "Global 7500"),
    ("G650", "Longitude"),
    ("Falcon 8X", "Challenger 3500"),
    ("Longitude", "Challenger 3500"),
]


def _dispatch_with_budget(a1: str, a2: str, budget_m: float):
    query = f"{a1} vs {a2} under ${budget_m}M"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    lock = build_intent_lock(query, qri=qri, unified_route=route)
    result = consult_authority_dispatch(
        query,
        qri=qri,
        unified_route=route,
        context={"db": None, "intent_lock": lock},
    )
    return query, lock, result


def _expected_passes_budget(a1: str, a2: str, budget_m: float) -> bool:
    filtered, _ = _apply_budget_filter([a1, a2], budget_m)
    return len(filtered) >= 2


@pytest.mark.parametrize("a1,a2", COMPARE_PAIRS)
@pytest.mark.parametrize("budget_m", BUDGETS_M)
def test_budget_matrix_dispatch_fail_closed_or_success(a1, a2, budget_m):
    query, lock, result = _dispatch_with_budget(a1, a2, budget_m)
    assert result is not None
    assert result.dispatch_kind == "comparison"
    assert lock.constraints.get("budget_m") == float(budget_m)

    budget_filter = (result.data_used or {}).get("authority_dispatch_budget_filter") or {}
    if budget_filter:
        assert budget_filter.get("budget_m") == float(budget_m)

    should_pass = _expected_passes_budget(a1, a2, budget_m)
    if should_pass:
        assert result.data_used.get("authority_dispatch_safety_fallback") is None
        assert "Insufficient verified data" not in result.answer
        models = result.data_used.get("authority_dispatch_models") or []
        assert len(models) >= 2
    else:
        assert result.data_used.get("authority_dispatch_safety_fallback") == "comparison"
        assert "Insufficient verified data" in result.answer


def test_triple_comparison_under_30m_fail_closed():
    query = "G650 vs Falcon 8X vs Global 7500 under $30M"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    lock = build_intent_lock(query, qri=qri, unified_route=route)
    result = consult_authority_dispatch(
        query,
        qri=qri,
        unified_route=route,
        context={"db": None, "intent_lock": lock},
    )
    assert result is not None
    assert result.data_used.get("authority_dispatch_safety_fallback") == "comparison"
    filtered = (result.data_used.get("authority_dispatch_budget_filter") or {}).get("filtered_models") or []
    assert len(filtered) < 2


def test_triple_comparison_under_100m_succeeds():
    query = "G650 vs Falcon 8X vs Global 7500 under $100M"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    lock = build_intent_lock(query, qri=qri, unified_route=route)
    result = consult_authority_dispatch(
        query,
        qri=qri,
        unified_route=route,
        context={"db": None, "intent_lock": lock},
    )
    assert result is not None
    assert result.data_used.get("authority_dispatch_safety_fallback") is None
    models = result.data_used.get("authority_dispatch_models") or []
    assert len(models) >= 2


def test_unknown_price_entity_rejected_by_budget_filter():
    filtered, results = _apply_budget_filter(["FakeJet9000", "G650"], 100.0)
    assert "FakeJet9000" not in filtered
    assert results.get("FakeJet9000") is False
    assert len(filtered) < 2


def test_lock_budget_wins_over_reparse():
    query = "G650 vs Falcon 8X under $30M"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    lock = build_intent_lock(query, qri=qri, unified_route=route)
    assert lock.constraints.get("budget_m") == 30.0
    graph = build_intent_graph(query, qri=qri, unified_intent=route)
    assert graph.constraints.get("budget_m") == 30.0
    result = consult_authority_dispatch(
        query,
        qri=qri,
        unified_route=route,
        context={"db": None, "intent_lock": lock},
    )
    budget_meta = (result.data_used or {}).get("authority_dispatch_budget_filter") or {}
    assert budget_meta.get("budget_m") == 30.0


@pytest.mark.parametrize("budget_m", BUDGETS_M)
def test_best_jet_under_budget_mission_not_dispatch(budget_m):
    query = f"Best jet under ${budget_m}M"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    result = consult_authority_dispatch(
        query,
        qri=qri,
        unified_route=route,
        context={"db": None},
    )
    assert result is None


@pytest.mark.parametrize("aircraft", AIRCRAFT)
def test_single_aircraft_comparison_fail_closed(aircraft):
    query = f"Compare {aircraft}"
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    result = consult_authority_dispatch(
        query,
        qri=qri,
        unified_route=route,
        context={"db": None},
    )
    if result is not None:
        assert result.data_used.get("authority_dispatch_safety_fallback") == "comparison"
