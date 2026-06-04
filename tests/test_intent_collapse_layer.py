"""Phase 45 — unified intent collapse layer tests."""

from __future__ import annotations

import pytest

from services.adversarial.adversarial_preprocessor import preprocess_adversarial_query
from services.broker_reasoning.broker_reasoning_layer import apply_broker_reasoning_layer
from services.intent_collapse.canonical_intent_frame import (
    AircraftScopeType,
    PrimaryIntent,
)
from services.intent_collapse.intent_collapse_engine import apply_intent_collapse, collapse_intent


def _collapse(query: str, du: dict | None = None) -> dict:
    data = du if du is not None else {}
    preprocess_adversarial_query(query, data_used=data)
    frame = apply_intent_collapse(query, data_used=data, normalized_query=query)
    return data["canonical_intent_frame"], frame


def test_cheap_gulfstream_frame():
    frame_dict, frame = _collapse("cheap gulfstream")
    assert frame.primary_intent == PrimaryIntent.BUY.value
    assert frame.aircraft_scope.scope_type == AircraftScopeType.ENTRY_LEVEL_GULFSTREAM_SCOPE.value
    assert frame.aircraft_scope.manufacturer == "Gulfstream"
    assert "PRICING_UNCLEAR" in frame.ambiguity_flags
    assert frame.budget.tier_hint == "LOW"
    assert frame.budget.unknown is True
    assert frame_dict["aircraft_scope"]["entry_level_only"] is True


def test_g650_vs_longitude_under_10m():
    frame_dict, frame = _collapse("G650 vs Longitude but under 10M")
    assert frame.primary_intent == PrimaryIntent.COMPARE.value
    assert frame.budget.cap_musd == pytest.approx(10.0)
    assert frame.budget.unknown is False
    assert "BUDGET_CONSTRAINT_ON_COMPARE" in frame.ambiguity_flags
    models = frame.aircraft_scope.models
    assert len(models) >= 2
    assert any("G650" in m for m in models)
    assert any("Longitude" in m for m in models)


def test_reasoning_executes_frame_not_reinterpret():
    du: dict = {}
    preprocess_adversarial_query("cheap gulfstream", data_used=du)
    apply_intent_collapse("cheap gulfstream", data_used=du)
    apply_broker_reasoning_layer("cheap gulfstream", data_used=du)
    br = du["broker_reasoning"]
    assert br.get("canonical_execution") is True
    assert du.get("broker_reasoning_from_canonical_frame") == 1
    candidates = br["category"]["candidates"]
    assert candidates
    assert candidates[0] == "Gulfstream G280"


def test_client_context_does_not_override_budget():
    du: dict = {
        "client_context": {"remembered_budget_musd": 20.0},
        "adversarial": {},
    }
    frame = collapse_intent(
        "cheap gulfstream",
        client_context=du["client_context"],
        adversarial=du["adversarial"],
    )
    assert frame.budget.cap_musd is None
    assert frame.budget.unknown is True


def test_compare_clarification_when_unresolvable():
    frame = collapse_intent("latitude vs something unknown jet")
    if "COMPARISON_AMBIGUOUS" in frame.ambiguity_flags:
        assert frame.clarification_request


def test_valuation_intent():
    _, frame = _collapse("is a 2015 Citation Latitude for $9.5M a good deal?")
    assert frame.primary_intent == PrimaryIntent.VALUATION.value
