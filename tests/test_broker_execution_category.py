"""Unit tests for Phase 55 broker execution categories."""

from __future__ import annotations

from services.broker_execution.broker_execution_category import (
    BrokerExecutionCategory,
    classify_broker_execution_category,
    comparison_requests_recommendation,
    executive_layer_allowed,
    tail_memory_isolated,
)


def test_tail_ownership_classification():
    assert (
        classify_broker_execution_category("Who owns N807JS?")
        == BrokerExecutionCategory.TAIL_OWNERSHIP
    )


def test_comparison_without_buy_intent():
    cat = classify_broker_execution_category("G280 vs Longitude")
    assert cat == BrokerExecutionCategory.COMPARISON
    assert not comparison_requests_recommendation("G280 vs Longitude")
    assert not executive_layer_allowed(cat, "G280 vs Longitude")


def test_comparison_with_explicit_buy():
    q = "G280 vs Longitude — which should I buy?"
    cat = classify_broker_execution_category(q)
    assert comparison_requests_recommendation(q)
    assert cat in (BrokerExecutionCategory.COMPARISON, BrokerExecutionCategory.ACQUISITION)
    assert executive_layer_allowed(cat, q) or cat == BrokerExecutionCategory.ACQUISITION


def test_tail_memory_isolated_categories():
    for q in ("Who owns N123AB?", "N807JS registry lookup"):
        cat = classify_broker_execution_category(q)
        if cat.value.startswith("tail") or cat == BrokerExecutionCategory.REGISTRY_LOOKUP:
            assert tail_memory_isolated(cat)
