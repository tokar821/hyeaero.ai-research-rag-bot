"""Budget + cabin shopping pivot."""

from __future__ import annotations

from services.intent_persistence.client_state import sanitize_client_state_for_shopping_pivot
from services.intent_persistence.pivot import (
    is_visual_budget_shopping_pivot,
    shopping_gallery_models,
    shopping_search_query,
)


def test_pivot_sanitizes_client_state():
    raw = {
        "current_aircraft_reference": "G650",
        "continuity": {"current_aircraft": "G650", "current_tail": "N123AB"},
        "conversation_memory": {"active_aircraft": "G650", "comparison_target": "G650 family"},
    }
    out = sanitize_client_state_for_shopping_pivot(raw)
    assert out["current_aircraft_reference"] is None
    assert out["continuity"]["current_aircraft"] is None
    assert out["conversation_memory"]["active_aircraft"] is None


def test_shopping_models_under_10m():
    models = shopping_gallery_models("Show me modern cabin under $10M.")
    assert "Challenger 350" in models
    assert "Praetor 500" in models
    assert not any("G650" in m for m in models)


def test_refinement_model_lists():
    from services.intent_persistence.pivot import (
        bigger_modern_cabin_models,
        less_corporate_interior_models,
        refinement_gallery_models,
    )

    assert "Praetor 600" in less_corporate_interior_models()
    assert "Global 6000" in bigger_modern_cabin_models()
    assert refinement_gallery_models("style_shift", "Something less corporate.")[0] == "Praetor 600"


def test_shopping_search_query_no_g650():
    q = shopping_search_query("Show me modern cabin under $10M.")
    assert "challenger" in q.lower() or "latitude" in q.lower()
    assert "g650" not in q.lower()
