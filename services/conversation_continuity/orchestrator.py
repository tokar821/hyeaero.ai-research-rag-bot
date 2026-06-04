"""
Conversation Continuity orchestrator — single entry for one user turn.

Runs **before** heavy retrieval inside :mod:`rag.consultant_retrieval` once the message
passed the aviation guardrail.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .aircraft_ladder import categorize_model_hint, evolution_hint_for_upgrade
from .contextual_signals import infer_contextual_tags
from .drift_prevention import continuity_drift_flags
from .entity_lock import explicit_aircraft_switch, merge_entity_lock
from .prompt_block import format_continuity_prompt_block
from .refinement import interpret_refinement, merge_traits, reinforce_query_with_context
from .response_mode import resolve_continuity_response_mode
from .schemas import (
    BuyerDirection,
    ConversationContinuityState,
    LockedEntityType,
    RefinementInterpretation,
    continuity_state_from_dict,
)
from .visual_prefs import extract_preferences


_BUDGET_RE = re.compile(
    r"\b(?:under|around|about|\~|<|<=)?\s*\$?\s*(\d+(?:[\.,]\d+)?)\s*(m(?:illion)?|mm|mil|k\b)\b",
    re.I,
)


def _history_user_blob(history: Optional[List[Dict[str, Any]]], max_pairs: int = 10) -> str:
    parts: List[str] = []
    for h in (history or [])[-max_pairs * 2 :]:
        if not isinstance(h, dict):
            continue
        if str(h.get("role") or "").lower() != "user":
            continue
        c = str(h.get("content") or "").strip()
        if c:
            parts.append(c)
    return " ".join(parts)


def _parse_budget_upper(query: str) -> Optional[float]:
    m = _BUDGET_RE.search(query or "")
    if not m:
        return None
    try:
        amt = float(m.group(1).replace(",", ""))
        unit = (m.group(2) or "").lower()
        if unit.startswith("m") or "mil" in unit:
            return amt * 1_000_000.0
        if unit == "k":
            return amt * 1_000.0
    except Exception:
        return None
    return None


@dataclass
class ContinuityTurnBundle:
    """Outputs from :func:`run_continuity_turn`."""

    effective_query: str
    state: ConversationContinuityState
    prompt_block: str
    refinement: RefinementInterpretation
    serialized: Dict[str, Any]


def _grow_evolution(
    prev_evolution: List[str],
    prev_air: Optional[str],
    focal_model: Optional[str],
    refinement_type: str,
) -> List[str]:
    if refinement_type == "comparison_anchor":
        return list(prev_evolution or [])[-12:]
    evolution = list(prev_evolution or [])
    pa = (prev_air or "").strip()
    fm = (focal_model or "").strip()
    if not fm:
        return evolution[-12:]
    if pa and fm.lower() != pa.lower():
        if not evolution or evolution[-1].lower() != pa.lower():
            evolution.append(pa)
        if not evolution or evolution[-1].lower() != fm.lower():
            evolution.append(fm)
    elif fm and not pa:
        if not evolution or evolution[-1].lower() != fm.lower():
            evolution.append(fm)
    return evolution[-12:]


def run_continuity_turn(
    *,
    raw_user_query: str,
    isolated_query: str,
    history: Optional[List[Dict[str, Any]]],
    client_conversation_state: Optional[Dict[str, Any]],
    strict_tail_candidates: Optional[List[str]],
) -> ContinuityTurnBundle:
    raw = (raw_user_query or "").strip()
    iso = (isolated_query or raw).strip()
    prev_bundle: Dict[str, Any] = {}
    if isinstance(client_conversation_state, dict):
        csub = client_conversation_state.get("continuity")
        if isinstance(csub, dict):
            prev_bundle = csub

    from services.intent_persistence.pivot import is_visual_budget_shopping_pivot

    _shopping_pivot = is_visual_budget_shopping_pivot(iso or raw)

    prev = continuity_state_from_dict({} if _shopping_pivot else prev_bundle)

    prev_air = (prev.current_aircraft or "").strip() or None
    prev_tail_val = (
        (prev.current_tail or "").strip()
        or (
            prev.locked_entity.value
            if prev.locked_entity and prev.locked_entity.type == LockedEntityType.TAIL
            else None
        )
    )

    refinement = interpret_refinement(raw, prev_aircraft=prev_air, prev_tail=prev_tail_val)

    if refinement.type == "explicit_reset":
        fresh = ConversationContinuityState()
        fresh.last_refinement = refinement
        fresh.drift_flags = continuity_drift_flags(fresh, not raw)
        return ContinuityTurnBundle(
            effective_query=iso or raw,
            state=fresh,
            prompt_block=format_continuity_prompt_block(fresh),
            refinement=refinement,
            serialized=fresh.model_dump(mode="json"),
        )

    from services.entity_scope.scope import history_allowed_for_tail_resolution

    _allow_history_tail = history_allowed_for_tail_resolution(iso or raw)

    explicit_switch = explicit_aircraft_switch(raw + " " + iso, prev_air)

    lock = merge_entity_lock(
        prev.locked_entity,
        query=(iso or raw),
        strict_tail_candidates=strict_tail_candidates,
        explicit_model=explicit_switch,
        prev_tail_aircraft=prev_air if prev_tail_val else None,
        allow_history_tail=_allow_history_tail,
    )

    try:
        from rag.consultant_query_expand import _detect_models

        hist_blob = _history_user_blob(history)
        model_blob = (raw + " " + iso) if _shopping_pivot else (raw + " " + iso + " " + hist_blob[-4000:])
        typed_models = _detect_models(model_blob) or []
    except Exception:
        typed_models = []

    focal_model = explicit_switch or (typed_models[0].strip() if typed_models else None)
    locked_tail_val = lock.value.strip().upper() if lock and lock.type == LockedEntityType.TAIL else None
    locked_serial = lock.value if lock and lock.type == LockedEntityType.SERIAL else None

    if refinement.inherit_entity is False:
        ct = locked_tail_val
    elif explicit_switch and not locked_tail_val:
        ct = None
    else:
        ct = locked_tail_val or (prev_tail_val if _allow_history_tail else None)

    if explicit_switch:
        focal_model = explicit_switch.strip()

    if not focal_model and refinement.inherit_entity:
        focal_model = prev_air

    # Prefer canonical client memory over stale continuity/history (prevents G650 bleed after budget pivot).
    if refinement.type in ("style_shift", "size_upgrade", "view_change", "ambiguous_followup"):
        if isinstance(client_conversation_state, dict):
            mem = client_conversation_state.get("conversation_memory")
            if isinstance(mem, dict) and (mem.get("active_aircraft") or "").strip():
                focal_model = str(mem["active_aircraft"]).strip()
            elif (client_conversation_state.get("current_aircraft_reference") or "").strip():
                focal_model = str(client_conversation_state["current_aircraft_reference"]).strip()

    if refinement.type == "comparison_anchor":
        focal_model = focal_model or prev_air

    if lock and lock.type == LockedEntityType.AIRCRAFT_MODEL:
        focal_model = lock.value.strip()

    if refinement.type != "comparison_anchor":
        fm_work = focal_model or prev_air
    else:
        fm_work = prev_air or focal_model

    if refinement.type in ("style_shift", "size_upgrade") and isinstance(client_conversation_state, dict):
        mem = client_conversation_state.get("conversation_memory")
        if isinstance(mem, dict) and (mem.get("active_aircraft") or "").strip():
            fm_work = str(mem["active_aircraft"]).strip()

    evolution = _grow_evolution(prev.aircraft_evolution, prev_air, fm_work or focal_model or prev_air, refinement.type)

    positive, negative_vis = extract_preferences(raw + " " + iso)
    pos_merged = merge_traits(prev.style_preferences, positive, refinement.remove_traits)
    neg_built = list(prev.negative_preferences or [])
    for n in negative_vis + (refinement.remove_traits or []):
        s = str(n).strip()
        if s and not any(x.lower() == s.lower() for x in neg_built):
            neg_built.append(s)
    neg_built = neg_built[-24:]

    buyer_direction = BuyerDirection.model_validate(prev.buyer_direction.model_dump())
    if refinement.type == "size_upgrade":
        buyer_direction.size = "larger"
    elif refinement.type in ("size_or_budget_down", "budget_shift"):
        buyer_direction.size = "smaller"
        buyer_direction.luxury = "lower"
    parsed_budget = _parse_budget_upper(raw + " " + iso)
    if parsed_budget is None and prev.buyer_direction.budget_usd_approx:
        parsed_budget = float(prev.buyer_direction.budget_usd_approx)
    if parsed_budget is None and isinstance(client_conversation_state, dict):
        mem = client_conversation_state.get("conversation_memory")
        if isinstance(mem, dict) and mem.get("active_budget_usd") is not None:
            try:
                parsed_budget = float(mem["active_budget_usd"])
            except (TypeError, ValueError):
                parsed_budget = None
        if parsed_budget is None:
            leg_b = str(client_conversation_state.get("current_budget") or "").strip()
            parsed_budget = _parse_budget_upper(leg_b)
    if parsed_budget:
        buyer_direction.budget_usd_approx = parsed_budget

    inferred = infer_contextual_tags(raw + " " + iso)
    inferred_tags = merge_traits([], list(refinement.inferred_style_tags) + inferred, [])

    last_view = prev.last_requested_view
    if refinement.requested_view:
        last_view = refinement.requested_view
    elif raw:
        ql = raw.lower()
        if re.search(r"\bcockpit|flight\s*deck\b", ql):
            last_view = "cockpit"
        elif re.search(r"\b(interior|cabin)\b", ql) or (
            re.search(r"\bshow\b", ql) and re.search(r"\b(photo|pic|gallery|image)\w*\b", ql)
        ):
            last_view = "interior"

    cat = categorize_model_hint(fm_work or focal_model)
    continuation_mode = resolve_continuity_response_mode(raw, inherited=prev.response_mode)

    state = ConversationContinuityState()
    state.locked_entity = lock
    state.current_aircraft = (fm_work or "").strip() or None
    state.current_tail = (ct or "").strip() or None
    if locked_serial and not state.current_tail:
        state.current_tail = str(locked_serial).strip()

    state.aircraft_evolution = evolution
    state.current_category = cat
    state.style_preferences = pos_merged
    state.negative_preferences = neg_built
    state.buyer_direction = buyer_direction
    state.last_requested_view = last_view
    state.response_mode = continuation_mode
    state.contextual_intent_tags = inferred_tags[-20:]
    state.last_refinement = refinement

    _budget_cap = float(buyer_direction.budget_usd_approx or 0) or None
    size_frag = evolution_hint_for_upgrade(
        prev_air if refinement.type == "size_upgrade" else None,
        max_budget_usd=_budget_cap if _budget_cap else None,
        refinement_query=raw,
    )
    augmented = bool(refinement.type == "size_upgrade" and prev_air)

    effective = reinforce_query_with_context(
        iso,
        interpretation=refinement,
        locked_tail=locked_tail_val,
        locked_model=state.current_aircraft,
        augment_size=bool(augmented and bool(size_frag.strip())),
        size_augment_fragment=size_frag,
    )

    if refinement.type in ("style_shift", "size_upgrade", "view_change", "ambiguous_followup"):
        extras: List[str] = []
        if state.current_aircraft and (state.current_aircraft or "").lower() not in effective.lower():
            extras.append(str(state.current_aircraft))
        if _budget_cap and _budget_cap <= 25_000_000:
            cap_m = int(round(_budget_cap / 1_000_000))
            if str(cap_m) not in effective and f"${cap_m}" not in effective:
                extras.append(f"under ~${cap_m}M budget")
        if refinement.type == "style_shift" and state.negative_preferences:
            for pref in state.negative_preferences[:2]:
                if pref.lower() not in effective.lower():
                    extras.append(pref)
        if extras:
            effective = (effective + " " + " ".join(extras)).strip()

    state.drift_flags = continuity_drift_flags(state, not raw)

    return ContinuityTurnBundle(
        effective_query=effective,
        state=state,
        prompt_block=format_continuity_prompt_block(state),
        refinement=refinement,
        serialized=state.model_dump(mode="json"),
    )
