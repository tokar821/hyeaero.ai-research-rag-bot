"""
Intent Persistence Engine — multi-turn orchestration entry point.

Runs conversation continuity, maps to :class:`PersistentIntentState`, scores
standalone confidence, and emits retrieval routing policy + debug fields.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from services.conversation_continuity import run_continuity_turn
from services.conversation_continuity.schemas import ContinuityResponseMode

from .inheritance import inherited_field_names, is_contextual_followup_query, merge_prev_snapshot
from .client_state import sanitize_client_state_for_shopping_pivot
from .pivot import is_visual_budget_shopping_pivot
from .routing import resolve_routing
from .schemas import (
    ConversationGoal,
    IntentResponseMode,
    PersistentIntentState,
    RoutingDecision,
    intent_state_from_dict,
)
from .standalone_score import score_standalone_confidence

logger = logging.getLogger(__name__)


@dataclass
class IntentPersistenceBundle:
    effective_query: str
    previous_intent: Dict[str, Any]
    resolved_intent: Dict[str, Any]
    inherited_fields: List[str]
    routing_decision: RoutingDecision
    standalone_confidence: float
    restore_thread_history: bool
    suppress_faa_registry_lookup: bool
    suppress_generic_vector_rag: bool
    force_gallery_intent: bool
    continuity_serialized: Dict[str, Any]
    refinement_type: str
    prompt_block: str


def _continuity_mode_to_intent(mode: ContinuityResponseMode) -> IntentResponseMode:
    if mode == ContinuityResponseMode.VISUAL_ONLY:
        return IntentResponseMode.IMAGE_SHOWCASE
    if mode == ContinuityResponseMode.SHORT_CAPTION:
        return IntentResponseMode.SHORT_CAPTION
    if mode == ContinuityResponseMode.COMPARISON_MODE:
        return IntentResponseMode.COMPARISON_MODE
    if mode == ContinuityResponseMode.TECHNICAL_MODE:
        return IntentResponseMode.TECHNICAL_MODE
    return IntentResponseMode.CONSULTANT_MODE


def _state_from_continuity_bundle(
    continuity_serialized: Dict[str, Any],
    *,
    standalone_confidence: float,
    refinement_type: str,
) -> PersistentIntentState:
    from .inheritance import continuity_dict_to_intent

    base = continuity_dict_to_intent(continuity_serialized)
    base.standalone_confidence = standalone_confidence
    base.last_refinement_type = refinement_type or base.last_refinement_type

    if refinement_type == "comparison_anchor" and base.comparison_target:
        base.current_conversation_goal = ConversationGoal.COMPARE_MODELS
    elif base.active_visual_focus or base.response_mode in (
        IntentResponseMode.IMAGE_SHOWCASE,
        IntentResponseMode.SHORT_CAPTION,
    ):
        base.current_conversation_goal = ConversationGoal.VISUAL_GALLERY
    elif refinement_type in ("size_upgrade", "style_shift", "budget_shift", "size_or_budget_down"):
        base.current_conversation_goal = ConversationGoal.REFINEMENT
    elif base.active_aircraft:
        base.current_conversation_goal = ConversationGoal.EXPLORE_MODEL

    if continuity_serialized.get("response_mode") == "visual_only":
        base.response_mode = IntentResponseMode.IMAGE_SHOWCASE

    return base


def run_intent_persistence_turn(
    *,
    raw_user_query: str,
    isolated_query: str,
    history: Optional[List[Dict[str, Any]]],
    client_conversation_state: Optional[Dict[str, Any]],
    strict_tail_candidates: Optional[List[str]],
) -> IntentPersistenceBundle:
    _pivot = is_visual_budget_shopping_pivot(isolated_query or raw_user_query)
    _client_for_turn = (
        sanitize_client_state_for_shopping_pivot(client_conversation_state)
        if _pivot
        else client_conversation_state
    )

    prev = merge_prev_snapshot(_client_for_turn) or PersistentIntentState()
    if _pivot:
        prev = prev.model_copy(
            update={
                "active_aircraft": None,
                "active_tail": None,
                "comparison_target": None,
                "active_visual_focus": None,
            }
        )
    prev_snapshot = prev.model_dump(mode="json")

    continuity = run_continuity_turn(
        raw_user_query=raw_user_query,
        isolated_query=isolated_query,
        history=history,
        client_conversation_state=_client_for_turn,
        strict_tail_candidates=strict_tail_candidates,
    )

    refinement_type = continuity.refinement.type or "none"
    conf = score_standalone_confidence(isolated_query or raw_user_query, prev=prev)

    resolved = _state_from_continuity_bundle(
        continuity.serialized,
        standalone_confidence=conf,
        refinement_type=refinement_type,
    )
    if _pivot:
        try:
            from .pivot import _parse_budget_millions, shopping_gallery_models

            bm = _parse_budget_millions(isolated_query or raw_user_query)
            budget_usd = float(bm) * 1_000_000.0 if bm is not None else None
            sgm = shopping_gallery_models(isolated_query or raw_user_query)
            anchor = (sgm[0] if sgm else None) or (continuity.state.current_aircraft if continuity.state else None)
            resolved = resolved.model_copy(
                update={
                    "active_aircraft": anchor,
                    "active_tail": None,
                    "comparison_target": None,
                    "current_conversation_goal": ConversationGoal.VISUAL_GALLERY,
                    "response_mode": IntentResponseMode.IMAGE_SHOWCASE,
                    "active_budget_usd": budget_usd,
                    "active_visual_focus": "modern cabin",
                }
            )
        except Exception:
            resolved = resolved.model_copy(
                update={
                    "active_aircraft": None,
                    "active_tail": None,
                    "comparison_target": None,
                    "current_conversation_goal": ConversationGoal.VISUAL_GALLERY,
                    "response_mode": IntentResponseMode.IMAGE_SHOWCASE,
                }
            )

    if refinement_type in (
        "style_shift",
        "size_upgrade",
        "view_change",
        "ambiguous_followup",
        "sleeping_configuration",
    ):
        carry: Dict[str, Any] = {}
        if not (resolved.active_budget_usd or 0) and (prev.active_budget_usd or 0) > 0:
            carry["active_budget_usd"] = prev.active_budget_usd
        if not (resolved.active_aircraft or "").strip() and (prev.active_aircraft or "").strip():
            carry["active_aircraft"] = prev.active_aircraft
        if not resolved.aesthetic_preferences and prev.aesthetic_preferences:
            carry["aesthetic_preferences"] = list(prev.aesthetic_preferences)
        if not resolved.negative_preferences and prev.negative_preferences:
            carry["negative_preferences"] = list(prev.negative_preferences)
        if carry:
            resolved = resolved.model_copy(update=carry)

    # IMAGE_SHOWCASE sticks across vague visual follow-ups.
    if is_contextual_followup_query(isolated_query or raw_user_query, prev):
        if prev.response_mode == IntentResponseMode.IMAGE_SHOWCASE:
            resolved.response_mode = IntentResponseMode.IMAGE_SHOWCASE
        elif continuity.state.response_mode in (
            ContinuityResponseMode.VISUAL_ONLY,
            ContinuityResponseMode.SHORT_CAPTION,
        ):
            resolved.response_mode = _continuity_mode_to_intent(continuity.state.response_mode)

    routing, restore_hist, suppress_faa, suppress_rag = resolve_routing(
        isolated_query or raw_user_query,
        prev=prev,
        resolved=resolved,
        standalone_confidence=conf,
        refinement_type=refinement_type,
    )

    inherited = inherited_field_names(prev, resolved)
    force_gallery = routing in (
        RoutingDecision.IMAGE_SHOWCASE_CONTINUATION,
        RoutingDecision.REFINEMENT_CONTINUATION,
    ) or resolved.response_mode == IntentResponseMode.IMAGE_SHOWCASE

    if _pivot:
        try:
            from .pivot import shopping_search_query

            effective = shopping_search_query(isolated_query or raw_user_query).strip()
        except Exception:
            effective = (continuity.effective_query or isolated_query or raw_user_query).strip()
    elif refinement_type in ("style_shift", "size_upgrade", "view_change"):
        try:
            from .pivot import refinement_gallery_models, refinement_search_query

            rq = refinement_search_query(
                refinement_type, isolated_query or raw_user_query
            ).strip()
            if rq:
                effective = rq
            else:
                effective = (continuity.effective_query or isolated_query or raw_user_query).strip()
            rmodels = refinement_gallery_models(refinement_type, isolated_query or raw_user_query)
            if rmodels:
                resolved = resolved.model_copy(update={"active_aircraft": rmodels[0]})
        except Exception:
            effective = (continuity.effective_query or isolated_query or raw_user_query).strip()
    else:
        effective = (continuity.effective_query or isolated_query or raw_user_query).strip()

    logger.info(
        "intent_persistence previous_intent=%s resolved_intent=%s inherited_fields=%s "
        "routing_decision=%s standalone_confidence=%.3f",
        _brief(prev_snapshot),
        _brief(resolved.model_dump(mode="json")),
        inherited,
        routing.value,
        conf,
    )

    return IntentPersistenceBundle(
        effective_query=effective,
        previous_intent=prev_snapshot,
        resolved_intent=resolved.model_dump(mode="json"),
        inherited_fields=inherited,
        routing_decision=routing,
        standalone_confidence=conf,
        restore_thread_history=restore_hist,
        suppress_faa_registry_lookup=suppress_faa,
        suppress_generic_vector_rag=suppress_rag,
        force_gallery_intent=force_gallery,
        continuity_serialized=continuity.serialized,
        refinement_type=refinement_type,
        prompt_block=continuity.prompt_block,
    )


def _brief(d: Dict[str, Any]) -> str:
    keys = (
        "active_aircraft",
        "active_tail",
        "response_mode",
        "active_visual_focus",
        "current_conversation_goal",
    )
    return str({k: d.get(k) for k in keys if k in d})
