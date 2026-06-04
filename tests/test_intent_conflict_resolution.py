"""Phase 16 — Intent Conflict Resolution Layer (ICRL) tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.deterministic

from services.recommendation.query_recommendation_intent import classify_query_recommendation_intent
from services.routing.authority_dispatch import consult_authority_dispatch
from services.routing.intent_conflict_resolution import (
    ConflictType,
    build_intent_graph,
    classify_conflict_type,
    execute_icrl_plan,
    resolve_intent_conflicts,
    resolve_multi_intent_execution,
)
from services.routing.unified_intent_router import classify_unified_intent


def _resolve(query: str):
    qri = classify_query_recommendation_intent(query, [])
    route = classify_unified_intent(query)
    return resolve_intent_conflicts(
        {"query": query, "qri": qri, "unified_route": route, "db": None}
    )


def test_triple_comparison_with_budget_constraint():
    query = "G650 vs Falcon 8X vs Global 7500 under $30M"
    result = _resolve(query)
    assert result.conflict_type == ConflictType.TRIPLE_PLUS_COMPARISON
    assert result.plan.comparison_mode == "comparison_matrix"
    assert result.plan.layout_type == "comparison_matrix_with_filter"
    assert result.plan.execution_strategy == "deterministic_only"
    assert result.handled_by_icrl is True
    assert len(result.graph.entities) >= 3
    assert result.graph.constraints.get("budget_m") == 30.0
    assert "budget_cap" in result.graph.modifiers
    assert result.plan.constraint_result
    assert len(result.plan.filtered_entities) >= 0


def test_simple_two_way_comparison_single_intent():
    query = "G650 vs Falcon 8X"
    result = _resolve(query)
    assert result.conflict_type == ConflictType.SINGLE_INTENT
    assert result.handled_by_icrl is False
    assert result.plan.primary_mode == "comparison"
    assert len(result.graph.entities) == 2


def test_comparison_plus_constraint():
    query = "G650 vs Falcon 8X under $20M"
    result = _resolve(query)
    assert result.conflict_type == ConflictType.COMPARISON_PLUS_CONSTRAINT
    assert result.plan.execution_strategy == "deterministic_only"
    assert result.handled_by_icrl is True
    assert result.plan.layout_type == "comparison_matrix_with_filter"
    assert "constraint_filter" in result.plan.secondary_modes
    assert result.graph.constraints.get("budget_m") == 20.0
    assert set(result.plan.constraint_result.keys()) == set(result.graph.entities)


def test_mission_single_intent_no_conflict():
    query = "What should I buy for 8 pax LA to Miami"
    result = _resolve(query)
    assert result.conflict_type == ConflictType.SINGLE_INTENT
    assert result.handled_by_icrl is False
    assert result.plan.execution_strategy == "hybrid_safe"
    assert result.plan.primary_mode == "mission"


def test_deterministic_only_never_llm_kind():
    query = "G650 vs Falcon 8X vs Global 7500 under $70M"
    resolution = _resolve(query)
    assert resolution.execution_strategy == "deterministic_only"
    assert len(resolution.plan.filtered_entities) >= 2
    out = execute_icrl_plan({"query": query, "pre_llm_pipeline_patch": {}}, resolution)
    assert out is not None
    kind, payload = out
    assert kind == "professional"
    assert kind != "llm"
    meta = payload["data_used"]["deterministic_execution"]
    assert meta["bypassed_llm"] is True
    assert meta["final_responder"] == "icrl_comparison_matrix"
    assert payload["data_used"]["icrl_execution_plan"]["ui_intent"] == "multi_intent_comparison_decision"


def test_icrl_does_not_restore_budget_rejected_entities():
    query = "G650 vs Falcon 8X vs Global 7500 under $30M"
    resolution = _resolve(query)
    assert resolution.handled_by_icrl is True
    assert len(resolution.plan.filtered_entities) < 2
    out = execute_icrl_plan({"query": query, "pre_llm_pipeline_patch": {}}, resolution)
    assert out is None


def test_build_intent_graph_modifiers():
    graph = build_intent_graph("G650 vs Falcon 8X under $20M")
    assert "vs" in graph.modifiers
    assert "budget_cap" in graph.modifiers
    assert graph.constraints["budget_m"] == 20.0


def test_classify_and_resolve_pipeline():
    query = "G650 vs Falcon 8X under $20M"
    graph = build_intent_graph(query)
    conflict = classify_conflict_type(graph)
    plan = resolve_multi_intent_execution(graph)
    assert conflict == ConflictType.COMPARISON_PLUS_CONSTRAINT
    assert plan.ui_intent == "multi_intent_comparison_decision"
    assert plan.primary_mode == "comparison"


def test_icrl_skips_when_authority_dispatch_present():
    query = "G650 vs Falcon 8X vs Global 7500 under $30M"
    resolution = _resolve(query)
    dispatch = consult_authority_dispatch(
        query,
        qri=classify_query_recommendation_intent(query, []),
        unified_route=classify_unified_intent(query),
        context={"db": None},
    )
    assert dispatch is not None
    out = execute_icrl_plan(
        {
            "query": query,
            "pre_llm_pipeline_patch": {},
            "authority_dispatch_result": dispatch,
        },
        resolution,
    )
    assert out is None
