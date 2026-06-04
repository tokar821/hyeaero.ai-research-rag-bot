"""Phase 40 — broker reasoning layer tests."""

from __future__ import annotations

import pytest

from services.adversarial.adversarial_preprocessor import preprocess_adversarial_query, try_adversarial_buy_block
from services.broker_reasoning.broker_reasoning_layer import (
    apply_broker_reasoning_layer,
    get_broker_reasoning_compare_models,
    infer_buy_fields,
    is_acquisition_budget_query,
    render_acquisition_guidance,
)
from services.broker_reasoning.category_resolver import resolve_category
from services.broker_reasoning.comparison_soft_resolution import (
    AUTO_RESOLVE_SOFT,
    soft_resolve_comparison,
)
from services.broker_reasoning.intent_expander import expand_intent
from services.broker_reasoning.mission_interpreter import interpret_mission
from services.broker_reasoning.multi_intent_planner import plan_multi_intent
from services.routing.authority_dispatch import consult_authority_dispatch


def test_expand_cheap_gulfstream():
    exp = expand_intent("cheap gulfstream")
    assert exp.manufacturer == "Gulfstream"
    assert exp.acquisition_focus is True
    assert exp.price_sensitivity == "high"


def test_expand_like_longitude_cheaper():
    exp = expand_intent("something like a Longitude but cheaper")
    assert exp.alternative_search is True
    assert exp.reference_model == "Citation Longitude"
    assert exp.constraint == "lower_acquisition_cost"


def test_mission_interpret_15m():
    m = interpret_mission("best jet for 15m")
    assert m.acquisition_budget_musd == pytest.approx(15.0)
    assert m.follow_up_questions or "mission_priority" in m.missing_fields


def test_category_cheap_gulfstream_not_g700():
    cat = resolve_category("cheap gulfstream", manufacturer="Gulfstream", price_sensitive=True)
    assert cat.candidates
    assert cat.candidates[0] == "Gulfstream G280"
    assert "G700" not in cat.candidates


def test_soft_comparison_longitude_phenom():
    res = soft_resolve_comparison("longitude vs phenom")
    assert res is not None
    assert res.action in ("auto", "auto_with_note")
    assert res.models[0] == "Citation Longitude"
    assert min(res.confidences) >= AUTO_RESOLVE_SOFT


def test_soft_comparison_challenger_latitude():
    res = soft_resolve_comparison("challenger vs latitude")
    assert res is not None
    assert len(res.models) == 2
    assert "Challenger" in res.models[0]
    assert "Latitude" in res.models[1]


def test_infer_buy_under_budget():
    parsed = infer_buy_fields("buy challenger 350 under 8m")
    assert parsed is not None
    assert parsed["model"] == "Challenger 350"
    assert parsed["budget_musd"] == pytest.approx(8.0)


def test_acquisition_budget_query_detection():
    assert is_acquisition_budget_query("buy challenger 350 under 8m")
    assert is_acquisition_budget_query("best jet around 15m")
    assert is_acquisition_budget_query("what should I buy for 20m")


def test_multi_intent_compare_and_buy():
    plan = plan_multi_intent("compare g650 vs g700 and tell me which is the better buy")
    assert plan.primary_intent == "comparison"
    assert "buy_decision" in plan.secondary_intents
    assert "buy_read" in plan.overlays


def test_multi_intent_compare_and_trend():
    plan = plan_multi_intent("compare longitude vs phenom and include market trend")
    assert plan.primary_intent == "comparison"
    assert "temporal" in plan.overlays


def test_apply_layer_stamps_data_used():
    du: dict = {}
    apply_broker_reasoning_layer("cheap gulfstream", data_used=du)
    assert du.get("broker_reasoning_layer_applied") == 1
    assert du["broker_reasoning"]["category"]["candidates"]


def test_compare_models_for_dispatch():
    du: dict = {}
    apply_broker_reasoning_layer("longitude vs phenom", data_used=du)
    models = get_broker_reasoning_compare_models(du)
    assert models is not None
    assert len(models) == 2


def test_acquisition_guidance_budget_discovery():
    du: dict = {}
    apply_broker_reasoning_layer("buy challenger 350 under 8m", data_used=du)
    text = render_acquisition_guidance("buy challenger 350 under 8m", data_used=du)
    assert "Challenger 350" in text
    assert "8" in text


def test_acquisition_guidance_alternative():
    du: dict = {}
    apply_broker_reasoning_layer("something like a longitude but cheaper", data_used=du)
    text = render_acquisition_guidance("something like a longitude but cheaper", data_used=du)
    assert "Longitude" in text
    assert "Citation" in text


def test_adversarial_regression_cheap_g700_under_5m():
    du: dict = {}
    clean = preprocess_adversarial_query("cheap G700 under $5M", data_used=du)
    apply_broker_reasoning_layer(clean.normalized_query, data_used=du)
    blocked = try_adversarial_buy_block(clean.normalized_query, du)
    assert blocked is not None
    assert "INFEASIBLE_BUDGET_CONSTRAINT" in blocked


def test_dispatch_soft_comparison_integration():
    du: dict = {}
    apply_broker_reasoning_layer("longitude vs phenom", data_used=du)
    result = consult_authority_dispatch(
        "longitude vs phenom",
        qri=None,
        unified_route=None,
        context={"broker_reasoning": du["broker_reasoning"], "clean_normalized_query": {}},
    )
    assert result is not None
    assert result.dispatch_kind == "comparison"
    assert result.answer.strip()
    assert result.data_used.get("comparison_v2", {}).get("status") == "OK" or result.data_used.get(
        "authority_dispatch_models"
    )


def test_dispatch_acquisition_guidance():
    du: dict = {}
    apply_broker_reasoning_layer("buy challenger 350 under 8m", data_used=du)
    result = consult_authority_dispatch(
        "buy challenger 350 under 8m",
        qri=None,
        unified_route=None,
        context={"broker_reasoning": du["broker_reasoning"]},
    )
    assert result is not None
    assert result.dispatch_kind == "buy_decision"
    assert "Challenger 350" in result.answer


def test_manufacturer_discovery_dassault():
    du: dict = {}
    apply_broker_reasoning_layer("best dassault under 20m", data_used=du)
    cat = du["broker_reasoning"]["category"]
    assert cat["manufacturer"] == "Dassault"
    assert cat["candidates"]
