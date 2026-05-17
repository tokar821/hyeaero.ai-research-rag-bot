"""Deterministic state mutation rules for luxury aviation threads."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from .schemas import AircraftCategory, ConversationGoal, ConversationMemoryState, ResponseMode


def _merge_traits(base: List[str], add: List[str], remove: List[str], cap: int = 48) -> List[str]:
    out = [str(x).strip() for x in base if str(x).strip()]
    rem = {r.lower() for r in remove}
    out = [x for x in out if x.lower() not in rem]
    for a in add:
        s = str(a).strip()
        if s and all(s.lower() != o.lower() for o in out):
            out.append(s)
    return out[-cap:]


def _touch(state: ConversationMemoryState, field: str, reinforced: Set[str]) -> None:
    state.field_turns[field] = int(state.turn_index or 0)
    reinforced.add(field)


def _map_response_mode(raw: Optional[str]) -> ResponseMode:
    v = (raw or "").strip().lower()
    if v in ("visual_only", "image_showcase"):
        return ResponseMode.IMAGE_SHOWCASE
    if v == "short_caption":
        return ResponseMode.SHORT_CAPTION
    if v == "comparison_mode":
        return ResponseMode.COMPARISON
    if v == "technical_mode":
        return ResponseMode.TECHNICAL
    return ResponseMode.CONSULTANT


def _category_from_name(name: Optional[str]) -> AircraftCategory:
    try:
        from services.conversation_continuity.aircraft_ladder import categorize_model_hint

        return categorize_model_hint(name)
    except Exception:
        return AircraftCategory.UNKNOWN


def _upgrade_category(cur: AircraftCategory) -> AircraftCategory:
    order = [
        AircraftCategory.UNKNOWN,
        AircraftCategory.VLJ,
        AircraftCategory.LIGHT,
        AircraftCategory.MIDSIZE,
        AircraftCategory.SUPER_MID,
        AircraftCategory.LARGE,
        AircraftCategory.ULR,
    ]
    try:
        idx = order.index(cur)
    except ValueError:
        idx = 2
    return order[min(idx + 1, len(order) - 1)]


def apply_update_rules(
    state: ConversationMemoryState,
    *,
    query: str,
    refinement_type: str,
    continuity: Optional[Dict[str, Any]],
    intent_resolved: Optional[Dict[str, Any]],
    legacy_state: Optional[Dict[str, Any]],
    entity_models: Optional[List[str]],
    user_wants_gallery: bool,
    mission_hint: Optional[str],
) -> ConversationMemoryState:
    """Apply turn updates; mutates ``state`` in place."""
    q = (query or "").strip()
    ql = q.lower()
    reinforced: Set[str] = set()
    ref = (refinement_type or "none").strip().lower()

    cont = continuity if isinstance(continuity, dict) else {}
    intent = intent_resolved if isinstance(intent_resolved, dict) else {}
    leg = legacy_state if isinstance(legacy_state, dict) else {}

    models = [str(m).strip() for m in (entity_models or []) if str(m).strip()]
    if not models and cont.get("current_aircraft"):
        models = [str(cont["current_aircraft"]).strip()]
    if intent.get("active_aircraft"):
        models = models or [str(intent["active_aircraft"]).strip()]

    # --- Entity anchors (highest priority) ---
    explicit_air = (models[0] if models else None) or cont.get("current_aircraft") or intent.get("active_aircraft")
    tail = cont.get("current_tail") or intent.get("active_tail")
    locked = cont.get("locked_entity") if isinstance(cont.get("locked_entity"), dict) else {}
    if locked.get("type") == "tail" and locked.get("value"):
        tail = str(locked["value"]).strip().upper()

    inherit_entity = ref not in ("explicit_reset",) and ref != "comparison_anchor"
    if explicit_air and (not state.active_aircraft or not inherit_entity or explicit_air.lower() != (state.active_aircraft or "").lower()):
        if explicit_air:
            prev_a = state.active_aircraft
            state.active_aircraft = explicit_air.strip()
            _touch(state, "active_aircraft", reinforced)
            if prev_a and prev_a.lower() != state.active_aircraft.lower():
                evo = list(state.aircraft_evolution or [])
                if not evo or evo[-1].lower() != prev_a.lower():
                    evo.append(prev_a)
                if evo[-1].lower() != state.active_aircraft.lower():
                    evo.append(state.active_aircraft)
                state.aircraft_evolution = evo[-12:]
                _touch(state, "aircraft_evolution", reinforced)
            state.active_category = _category_from_name(state.active_aircraft)
            _touch(state, "active_category", reinforced)
    elif inherit_entity and state.active_aircraft:
        _touch(state, "active_aircraft", reinforced)
        _touch(state, "active_category", reinforced)

    if tail:
        state.active_tail = str(tail).strip().upper()
        _touch(state, "active_tail", reinforced)
    elif inherit_entity and state.active_tail:
        _touch(state, "active_tail", reinforced)

    # --- Refinement rules ---
    if ref == "size_upgrade":
        state.active_category = _upgrade_category(state.active_category)
        _touch(state, "active_category", reinforced)
        state.conversation_goal = ConversationGoal.REFINEMENT
        _touch(state, "conversation_goal", reinforced)
        state.active_topic = "size_upgrade"
        _touch(state, "active_topic", reinforced)
        if state.active_aircraft:
            _touch(state, "active_aircraft", reinforced)

    elif ref == "style_shift":
        add = list(cont.get("style_preferences") or intent.get("aesthetic_preferences") or [])
        neg = list(cont.get("negative_preferences") or intent.get("negative_preferences") or [])
        try:
            from services.conversation_continuity.visual_prefs import extract_preferences

            p, n = extract_preferences(q)
            add = add + p
            neg = neg + n
        except Exception:
            pass
        lr = cont.get("last_refinement") if isinstance(cont.get("last_refinement"), dict) else {}
        add = add + list(lr.get("add_traits") or [])
        neg = neg + list(lr.get("remove_traits") or [])
        state.aesthetic_preferences = _merge_traits(state.aesthetic_preferences, add, [])
        state.negative_preferences = _merge_traits(state.negative_preferences, neg, [])
        _touch(state, "aesthetic_preferences", reinforced)
        _touch(state, "negative_preferences", reinforced)
        state.conversation_goal = ConversationGoal.REFINEMENT
        _touch(state, "conversation_goal", reinforced)
        state.active_topic = "style_shift"
        _touch(state, "active_topic", reinforced)

    elif ref == "view_change":
        view = cont.get("last_requested_view") or intent.get("active_visual_focus") or "cockpit"
        if re.search(r"\bcockpit|flight\s+deck\b", ql):
            view = "cockpit"
        elif re.search(r"\b(interior|cabin)\b", ql):
            view = "interior"
        state.last_visual_context = str(view)
        _touch(state, "last_visual_context", reinforced)
        if state.active_aircraft:
            _touch(state, "active_aircraft", reinforced)
        if state.active_tail:
            _touch(state, "active_tail", reinforced)
        state.conversation_goal = ConversationGoal.VISUAL_GALLERY
        _touch(state, "conversation_goal", reinforced)

    elif ref == "comparison_anchor":
        comp = intent.get("comparison_target")
        lr = cont.get("last_refinement") if isinstance(cont.get("last_refinement"), dict) else {}
        comp = comp or lr.get("reference_aircraft")
        if comp:
            state.comparison_target = str(comp).strip()
            _touch(state, "comparison_target", reinforced)
        state.conversation_goal = ConversationGoal.COMPARE
        _touch(state, "conversation_goal", reinforced)

    elif ref in ("budget_shift", "size_or_budget_down"):
        state.conversation_goal = ConversationGoal.REFINEMENT
        _touch(state, "conversation_goal", reinforced)

    # --- Response mode & visual ---
    rm = _map_response_mode(
        str(cont.get("response_mode") or intent.get("response_mode") or leg.get("conversation_mode") or "")
    )
    if user_wants_gallery or rm in (ResponseMode.IMAGE_SHOWCASE, ResponseMode.SHORT_CAPTION):
        if ref == "view_change" or re.search(r"\b(show|photo|gallery|cabin|cockpit|interior)\b", ql):
            rm = ResponseMode.IMAGE_SHOWCASE if rm == ResponseMode.CONSULTANT else rm
    prev_rm = state.response_mode
    if rm != ResponseMode.CONSULTANT or prev_rm == ResponseMode.CONSULTANT:
        state.response_mode = rm
    elif ref in ("view_change", "ambiguous_followup", "none") and prev_rm in (
        ResponseMode.IMAGE_SHOWCASE,
        ResponseMode.SHORT_CAPTION,
        ResponseMode.VISUAL_ONLY,
    ):
        state.response_mode = prev_rm
        _touch(state, "response_mode", reinforced)
    else:
        state.response_mode = rm
    _touch(state, "response_mode", reinforced)

    vis = leg.get("current_visual_intent") or intent.get("active_visual_focus") or cont.get("last_requested_view")
    if vis:
        state.last_visual_context = str(vis)
        _touch(state, "last_visual_context", reinforced)
    elif user_wants_gallery and not state.last_visual_context:
        state.last_visual_context = "cabin interior"
        _touch(state, "last_visual_context", reinforced)

    # --- Budget / mission ---
    if intent.get("active_budget_usd"):
        try:
            state.active_budget_usd = float(intent["active_budget_usd"])
            _touch(state, "active_budget_usd", reinforced)
        except (TypeError, ValueError):
            pass
    bd = cont.get("buyer_direction") if isinstance(cont.get("buyer_direction"), dict) else {}
    if bd.get("budget_usd_approx"):
        try:
            state.active_budget_usd = float(bd["budget_usd_approx"])
            _touch(state, "active_budget_usd", reinforced)
        except (TypeError, ValueError):
            pass

    bl = leg.get("current_budget")
    if bl:
        state.active_budget_label = str(bl)
        _touch(state, "active_budget_usd", reinforced)

    mission = leg.get("current_mission") or (mission_hint or "").strip() or None
    if mission:
        state.active_mission = str(mission)[:240]
        _touch(state, "active_mission", reinforced)

    if state.response_mode in (ResponseMode.IMAGE_SHOWCASE, ResponseMode.SHORT_CAPTION) or user_wants_gallery:
        if state.conversation_goal == ConversationGoal.UNKNOWN:
            state.conversation_goal = ConversationGoal.VISUAL_GALLERY
            _touch(state, "conversation_goal", reinforced)

    if state.active_aircraft and state.conversation_goal == ConversationGoal.UNKNOWN:
        state.conversation_goal = ConversationGoal.EXPLORE
        _touch(state, "conversation_goal", reinforced)

    state.reinforced_fields = sorted(reinforced)
    return state
