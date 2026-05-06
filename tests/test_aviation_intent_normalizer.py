"""Aviation intent normalizer — schema coercion + heuristic path."""

from rag.intent.aviation_intent_normalizer import (
    coerce_normalized_aviation_intent,
    default_normalized_aviation_intent,
    normalize_aviation_intent_heuristic,
)


def test_default_shape():
    d = default_normalized_aviation_intent()
    assert d["intent_type"] == "aircraft_lookup"
    assert d["aircraft"] is None
    assert d["category"] is None
    assert d["visual_focus"] is None
    assert set(d["constraints"].keys()) == {"budget", "style", "comparison_target"}


def test_coerce_accepts_valid():
    raw = {
        "intent_type": "interior_visual",
        "aircraft": "Challenger 350",
        "category": "midsize",
        "visual_focus": "interior",
        "constraints": {"budget": 1e7, "style": "modern", "comparison_target": "G650"},
    }
    out = coerce_normalized_aviation_intent(raw)
    assert out == raw


def test_coerce_rejects_bad_intent_type():
    out = coerce_normalized_aviation_intent({"intent_type": "banana"})
    assert out["intent_type"] == "aircraft_lookup"


def test_coerce_budget():
    out = coerce_normalized_aviation_intent(
        {"constraints": {"budget": "12500000", "style": None, "comparison_target": None}}
    )
    assert out["constraints"]["budget"] == 12500000.0


def test_heuristic_comparison():
    out = normalize_aviation_intent_heuristic("Compare Gulfstream G500 vs Falcon 7X for TATL", None)
    assert out["intent_type"] == "comparison"


def test_heuristic_interior_visual_show_cabin():
    out = normalize_aviation_intent_heuristic("Show me the cabin of the Challenger 350", None)
    assert out["intent_type"] == "interior_visual"
    assert out["visual_focus"] == "interior"


def test_heuristic_cabin_specs_not_gallery():
    out = normalize_aviation_intent_heuristic("What is the cabin height on the Citation X?", None)
    assert out["intent_type"] == "cabin_search"
    assert out["visual_focus"] == "interior"


def test_heuristic_cockpit():
    out = normalize_aviation_intent_heuristic("Global 7500 cockpit layout", None)
    assert out["intent_type"] == "cockpit"
    assert out["visual_focus"] == "cockpit"


def test_heuristic_budget_under_millions():
    out = normalize_aviation_intent_heuristic("Best cabin under $12M for US east-west", None)
    assert out["constraints"]["budget"] == 12_000_000.0


def test_heuristic_g650_comparison_target():
    out = normalize_aviation_intent_heuristic("Something like a G650 but cheaper", None)
    assert out["constraints"]["comparison_target"] == "Gulfstream G650"
    assert out["category"] == "ultra long range"
