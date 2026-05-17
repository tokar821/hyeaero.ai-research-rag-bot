"""Retrieval routing policy from resolved persistent intent."""

from __future__ import annotations

from typing import Tuple

from .inheritance import is_contextual_followup_query
from .schemas import IntentResponseMode, PersistentIntentState, RoutingDecision


def resolve_routing(
    query: str,
    *,
    prev: PersistentIntentState,
    resolved: PersistentIntentState,
    standalone_confidence: float,
    refinement_type: str,
) -> Tuple[RoutingDecision, bool, bool, bool]:
    """
    Returns ``routing_decision, restore_thread_history, suppress_faa_registry, suppress_generic_vector_rag``.
    """
    q = (query or "").strip()
    inherited = is_contextual_followup_query(q, prev) or standalone_confidence < 0.45
    has_anchor = bool(
        (resolved.active_aircraft or "").strip()
        or (resolved.active_tail or "").strip()
        or (prev.active_budget_usd or 0) > 0
        or bool(prev.aesthetic_preferences)
        or prev.current_conversation_goal.value == "visual_gallery"
    )

    showcase = resolved.response_mode in (
        IntentResponseMode.IMAGE_SHOWCASE,
        IntentResponseMode.VISUAL_ONLY,
        IntentResponseMode.SHORT_CAPTION,
    )
    prev_showcase = prev.response_mode in (
        IntentResponseMode.IMAGE_SHOWCASE,
        IntentResponseMode.VISUAL_ONLY,
        IntentResponseMode.SHORT_CAPTION,
    )

    if refinement_type == "explicit_reset":
        return RoutingDecision.FRESH_RETRIEVAL, False, False, False

    if inherited and has_anchor:
        if showcase or (prev_showcase and refinement_type in ("view_change", "ambiguous_followup", "none")):
            return RoutingDecision.IMAGE_SHOWCASE_CONTINUATION, True, True, True
        if refinement_type in (
            "size_upgrade",
            "style_shift",
            "size_or_budget_down",
            "budget_shift",
            "lifestyle_inference",
            "sleeping_configuration",
        ):
            return RoutingDecision.REFINEMENT_CONTINUATION, True, True, True
        return RoutingDecision.INHERIT_CONTEXT, True, True, bool(showcase or prev_showcase)

    if showcase and has_anchor:
        return RoutingDecision.IMAGE_SHOWCASE_CONTINUATION, True, True, True

    return RoutingDecision.FRESH_RETRIEVAL, False, False, False
