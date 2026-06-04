"""Phase 38 — adversarial preprocessor tests (deterministic, no LLM)."""

from __future__ import annotations

import pytest

from services.adversarial.adversarial_preprocessor import (
    check_comparison_safety,
    preprocess_adversarial_query,
    to_unified_adversarial_metadata,
    try_adversarial_buy_block,
)
from services.adversarial.budget_conflict_normalizer import (
    BudgetFeasibility,
    PriceSignalKind,
    classify_price_signals,
    normalize_budget_conflicts,
)
from services.adversarial.intent_sanitizer import sanitize_intents
from services.adversarial.model_adversary_resolver import resolve_adversary_models
from services.adversarial.query_conflict_detector import (
    ConflictSeverity,
    ConflictType,
    detect_query_conflicts,
)
from services.consistency.consistency_injection_layer import prepare_buy_decision_state, render_buy_decision_answer
from services.market_intelligence.deal_quality_engine import evaluate_deal_quality
from services.market_intelligence.market_band_builder import BandConfidence, MarketBand

pytestmark = pytest.mark.deterministic


# --- CONFLICT_DETECTION_ACCURACY ---


def test_conflict_detection_g700_under_5m() -> None:
    report = detect_query_conflicts("cheap G700 under $5M good deal")
    assert ConflictType.BUDGET_MODEL_INFEASIBLE in report.conflict_type
    assert report.severity == ConflictSeverity.HIGH


def test_listing_ask_not_treated_as_acquisition_budget() -> None:
    signals = classify_price_signals("Is a 2015 Citation Latitude for $5M a good deal?")
    kinds = {s.kind for s in signals}
    assert PriceSignalKind.LISTING_ASK in kinds
    assert PriceSignalKind.ACQUISITION_BUDGET not in kinds
    state = normalize_budget_conflicts("Is a 2015 Citation Latitude for $5M a good deal?")
    assert state.feasibility == BudgetFeasibility.FEASIBLE
    assert state.listing_ask_musd == 5.0


# --- INTENT_SANITIZATION_PRIORITY_RULES ---


def test_intent_override_only_when_conflict() -> None:
    assert sanitize_intents("G650 vs Falcon 8X") is None
    assert sanitize_intents("What is a G650 worth?") is None


def test_intent_sanitization_buy_over_compare() -> None:
    assert sanitize_intents("Compare G650 vs Falcon 8X and is it a good deal at $10M") == "buy"


def test_intent_sanitization_buy_over_valuation() -> None:
    assert sanitize_intents("What is a G650 worth and should I buy one") == "buy"


def test_intent_sanitization_compare_over_valuation() -> None:
    assert sanitize_intents("What is a G650 worth vs Falcon 8X") == "compare"


# --- MODEL_AMBIGUITY_RESOLUTION ---


def test_model_ambiguity_longitude_registry() -> None:
    models = resolve_adversary_models("longitude jet vs phenom 300")
    assert any(m.canonical_model == "Citation Longitude" for m in models)
    from services.comparison.aircraft_registry_lock import CANONICAL_COMPARISON_REGISTRY

    for m in models:
        assert m.canonical_model in CANONICAL_COMPARISON_REGISTRY


def test_cheap_gulfstream_resolves_to_verified_model() -> None:
    models = resolve_adversary_models("cheap gulfstream under $8M")
    assert len(models) >= 1
    assert models[0].canonical_model.startswith("Gulfstream")


# --- BUDGET_CONFLICT_CLASSIFICATION ---


def test_budget_conflict_infeasible_acquisition() -> None:
    state = normalize_budget_conflicts("Gulfstream G700 under $5M")
    assert state.feasibility == BudgetFeasibility.INFEASIBLE
    assert state.acquisition_cap_musd == 5.0


# --- DOWNSTREAM_QUERY_NORMALIZATION ---


def test_downstream_query_normalization_stamps_data_used() -> None:
    du: dict = {}
    clean = preprocess_adversarial_query("cheap G700 under $5M", data_used=du)
    assert "clean_normalized_query" in du
    assert "adversarial" in du
    assert du["adversarial"]["normalized_query"] == clean.normalized_query
    assert "conflict_report" in du["adversarial"]
    assert "resolved_models" in du["adversarial"]


def test_deterministic_preprocess_twice() -> None:
    a = preprocess_adversarial_query("G650 vs Falcon 8X under $3M")
    b = preprocess_adversarial_query("G650 vs Falcon 8X under $3M")
    assert a.conflict_report.severity == b.conflict_report.severity
    assert list(a.conflict_report.conflict_type) == list(b.conflict_report.conflict_type)


def test_unified_adversarial_metadata_shape() -> None:
    clean = preprocess_adversarial_query("cheap G700 under $5M")
    meta = to_unified_adversarial_metadata(clean)
    assert set(meta.keys()) >= {"normalized_query", "conflict_report", "resolved_models"}


# --- BUY / COMPARISON GATES ---


def test_buy_block_infeasible_budget() -> None:
    du: dict = {}
    preprocess_adversarial_query("Is a Gulfstream G700 under $5M a good deal?", data_used=du)
    body = try_adversarial_buy_block(du["clean_normalized_query"]["normalized_query"], du)
    assert body is not None
    assert "INFEASIBLE_BUDGET_CONSTRAINT" in body


def test_buy_listing_ask_not_blocked() -> None:
    du: dict = {}
    preprocess_adversarial_query("Is a 2015 Citation Latitude for $5M a good deal?", data_used=du)
    body = try_adversarial_buy_block(du["clean_normalized_query"]["normalized_query"], du)
    assert body is None


def test_buy_block_clarification_high_severity() -> None:
    du: dict = {}
    preprocess_adversarial_query(
        "buy and compare G650 vs Falcon 8X — what is it worth and is it a good deal?",
        data_used=du,
    )
    assert du["adversarial_preprocess"]["severity"] == "HIGH"
    body = try_adversarial_buy_block(du["clean_normalized_query"]["normalized_query"], du)
    assert body is not None
    assert "CLARIFICATION_REQUIRED" in body


def test_comparison_safety_ambiguous() -> None:
    du: dict = {}
    msg = check_comparison_safety("cheap gulfstream vs longitude", data_used=du)
    assert msg is None or "CLARIFICATION" in msg


# --- NO DOWNSTREAM MUTATION OF MARKET MATH ---


def test_deal_quality_unchanged_by_adversarial_preprocess() -> None:
    band = MarketBand(
        low=10_000_000.0,
        mid=11_800_000.0,
        high=13_000_000.0,
        confidence=BandConfidence.HIGH,
        listing_count=10,
    )
    before = evaluate_deal_quality(
        model="Citation Latitude", year=2018, ask_usd=9_500_000.0, band=band
    )
    preprocess_adversarial_query("Is a 2018 Citation Latitude for $9.5M a good deal?")
    after = evaluate_deal_quality(
        model="Citation Latitude", year=2018, ask_usd=9_500_000.0, band=band
    )
    assert before.verdict == after.verdict
    assert before.reason == after.reason


def test_buy_state_includes_adversarial_metadata() -> None:
    du: dict = {}
    preprocess_adversarial_query("Is a 2015 Citation Latitude for $5M a good deal?", data_used=du)
    parsed = {"model": "Citation Latitude", "year": 2015, "ask_usd": 5_000_000.0}
    state = prepare_buy_decision_state(
        query=du["clean_normalized_query"]["normalized_query"],
        parsed=parsed,
        db=None,
        data_used=du,
    )
    assert state.adversarial is not None
    assert state.adversarial.get("normalized_query")
    body = render_buy_decision_answer(state)
    assert "GOOD DEAL" in body.upper() or "FAIR DEAL" in body.upper()
