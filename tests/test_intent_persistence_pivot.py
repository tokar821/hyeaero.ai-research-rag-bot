"""Visual + budget shopping pivot clears inherited aircraft."""

from __future__ import annotations

from services.intent_persistence.pivot import is_visual_budget_shopping_pivot
from services.intent_persistence.routing import resolve_routing
from services.intent_persistence.schemas import IntentResponseMode, PersistentIntentState, RoutingDecision


def test_pivot_detects_modern_cabin_budget():
    assert is_visual_budget_shopping_pivot("Show me modern cabin under $10M.")


def test_pivot_not_on_compare():
    assert not is_visual_budget_shopping_pivot("Compare G700 vs Global 7500.")


def test_routing_fresh_on_pivot():
    prev = PersistentIntentState(active_aircraft="G650", response_mode=IntentResponseMode.CONSULTANT_MODE)
    resolved = PersistentIntentState(active_aircraft="G650")
    decision, *_ = resolve_routing(
        "Show me modern cabin under $10M.",
        prev=prev,
        resolved=resolved,
        standalone_confidence=0.3,
        refinement_type="none",
    )
    assert decision == RoutingDecision.FRESH_RETRIEVAL
