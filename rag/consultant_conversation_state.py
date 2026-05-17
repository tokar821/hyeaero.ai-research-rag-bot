"""
Live consultant conversation state (server-computed, client-echoable).

Updated after each user turn. Clients may send back ``conversation_state`` from the
prior response's ``data_used`` so deictic follow-ups stay anchored without re-deriving
everything from raw history alone.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional

_STATE_KEYS = (
    "user_style",
    "current_aircraft_reference",
    "current_visual_intent",
    "current_budget",
    "current_mission",
    "current_passenger_count",
    "current_cabin_preference",
    "conversation_mode",
)

_DEFAULT_STATE: Dict[str, Any] = {k: None for k in _STATE_KEYS}

_RESET_RE = re.compile(
    r"\b("
    r"new\s+question|different\s+topic|unrelated|forget\s+(?:that|the)|start\s+over|"
    r"switching\s+gears|ignore\s+(?:that|the)\s+above|never\s+mind\s+that"
    r")\b",
    re.I,
)

_DEICTIC_FOLLOWUP_RE = re.compile(
    r"\b("
    r"something\s+nicer|something\s+better|bigger|more\s+modern|not\s+that|"
    r"similar\s+to\s+that|like\s+that|the\s+same\s+one|same\s+jet|that\s+one|"
    r"like\s+a\s+hotel|hotel\s+vibe|luxury\s+hotel|premium\s+feel|that\s+interior|"
    r"again|same\s+cabin|more\s+like\s+that"
    r")\b",
    re.I,
)

_BUDGET_RE = re.compile(
    r"(?:under|around|about|~|<=?)\s*\$?\s*(\d+(?:\.\d+)?)\s*(m|mm|million|mil)\b",
    re.I,
)
_PAX_RE = re.compile(
    r"\b(\d{1,2})\s*(?:pax|passengers?|people|seats?|souls)\b",
    re.I,
)
_MISSION_ROUTE_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*(?:→|->|to)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
)


def default_consultant_conversation_state() -> Dict[str, Any]:
    return copy.deepcopy(_DEFAULT_STATE)


def _sanitize_client_state(raw: Any) -> Dict[str, Any]:
    out = default_consultant_conversation_state()
    if not isinstance(raw, dict):
        return out
    for k in _STATE_KEYS:
        v = raw.get(k)
        if v is None:
            continue
        if isinstance(v, (str, int, float)):
            s = str(v).strip()
            out[k] = s if s else None
        elif isinstance(v, bool) and k not in out:
            continue
    if isinstance(raw.get("continuity"), dict):
        out["continuity"] = raw["continuity"]
    if isinstance(raw.get("conversation_memory"), dict):
        out["conversation_memory"] = raw["conversation_memory"]
    if isinstance(raw.get("intent_persistence"), dict):
        out["intent_persistence"] = raw["intent_persistence"]
    return out


def _thread_user_blob(history: Optional[List[Dict[str, str]]], max_turns: int = 14) -> str:
    parts: List[str] = []
    for h in (history or [])[-max_turns:]:
        if not isinstance(h, dict):
            continue
        if str(h.get("role") or "").strip().lower() != "user":
            continue
        c = str(h.get("content") or "").strip()
        if c:
            parts.append(c)
    return " ".join(parts).strip()


def _infer_models_from_history(history: Optional[List[Dict[str, str]]]) -> List[str]:
    try:
        from rag.consultant_query_expand import _detect_models
    except Exception:
        return []
    blob = _thread_user_blob(history, max_turns=20)
    if not blob:
        return []
    try:
        found = _detect_models(blob) or []
    except Exception:
        return []
    out: List[str] = []
    for m in found:
        s = str(m or "").strip()
        if s and s not in out:
            out.append(s)
    return out[:6]


def _infer_style_and_cabin(ql: str) -> tuple[Optional[str], Optional[str]]:
    style: Optional[str] = None
    cabin: Optional[str] = None
    if re.search(r"\b(ceo|board|executive|c-suite)\b", ql):
        style = "CEO vibe"
    elif re.search(r"\b(first[- ]time|new\s+to\s+jets?|never\s+owned)\b", ql):
        style = "first-time buyer"
    elif re.search(r"\b(value|cheap|budget|deal|cap\b|under\s+\$)\b", ql):
        style = "value-focused"
    elif re.search(r"\b(luxury|premium|flagship|hotel|boutique|bespoke)\b", ql):
        style = "luxury-focused"
    elif re.search(r"\b(minimal|clean\s+lines|sparse)\b", ql):
        style = "minimalist"

    if re.search(r"\b(stand[- ]?up|flat\s+floor|bed\b|berth|divan)\b", ql):
        cabin = "flat floor / sleeping"
    elif re.search(r"\b(large\s+cabin|wide\s+cabin|tall\s+cabin)\b", ql):
        cabin = "large cabin volume"
    return style, cabin


def _infer_visual_intent(q: str) -> Optional[str]:
    ql = q.lower().strip()
    if not ql:
        return None
    if re.search(r"\b(cockpit|flight\s+deck)\b", ql):
        return "cockpit"
    if re.search(r"\b(bedroom|berth|divan|sleep)\b", ql):
        return "bedroom setup"
    if re.search(r"\b(ambient|lighting|mood\s+light)\b", ql):
        return "ambient lighting"
    if re.search(r"\b(hotel|suite|spa\s+bath)\b", ql):
        return "luxury hotel vibe"
    if re.search(r"\b(modern|contemporary|white\s+interior|huge\s+windows)\b", ql):
        return "modern cabin"
    if re.search(r"\b(cabin|interior|inside|show\s+me|photos?|pictures?)\b", ql):
        return "cabin interior"
    return None


def _infer_budget(q: str) -> Optional[str]:
    m = _BUDGET_RE.search(q or "")
    if not m:
        return None
    amt, unit = m.group(1), (m.group(2) or "").lower()
    u = "M" if unit.startswith("m") else unit
    return f"${amt}{u}"


def _infer_pax(q: str) -> Optional[str]:
    m = _PAX_RE.search(q or "")
    if not m:
        return None
    return m.group(1)


def _infer_mission(q: str, mission_hint: Optional[str]) -> Optional[str]:
    if (mission_hint or "").strip():
        return (mission_hint or "").strip()[:240]
    m = _MISSION_ROUTE_RE.search(q or "")
    if m:
        return f"{m.group(1)} → {m.group(2)}"
    m2 = re.search(r"\b(transatlantic|transcon|coast[- ]to[- ]coast|regional)\b", (q or "").lower())
    if m2:
        return m2.group(0)
    return None


def _map_conversation_mode(
    *,
    fine_intent: Optional[str],
    hybrid_kind: Optional[str],
    response_mode: Optional[str],
    ql: str,
) -> str:
    rm = (response_mode or "").strip().lower()
    fi = (fine_intent or "").strip().lower()
    hk = (hybrid_kind or "").strip().lower()

    if rm == "deal_analysis_mode" or hk == "aircraft_price_lookup" or re.search(
        r"\b(good\s+deal|overpriced|market\s+value|worth\s+it)\b", ql
    ):
        return "deal-analysis"
    if fi == "aircraft_comparison" or re.search(r"\bvs\.?\b|\bversus\b|\bcompare\b", ql):
        return "comparing"
    if hk == "aircraft_listing_query" or fi in ("aircraft_recommendation",):
        return "shopping"
    if re.search(r"\b(someday|dream|aspir|bucket\s+list|if\s+i\s+could)\b", ql):
        return "aspirational"
    return "browsing"


def _primary_aircraft_reference_fixed(
    entity_models: Optional[List[str]],
    query: str,
    history: Optional[List[Dict[str, str]]],
    prev_air: Optional[str],
    deictic: bool,
) -> Optional[str]:
    models = [str(x).strip() for x in (entity_models or []) if str(x).strip()]
    if models:
        return models[0]
    try:
        from rag.consultant_query_expand import _detect_models

        q_models = _detect_models(query or "") or []
    except Exception:
        q_models = []
    if q_models:
        return str(q_models[0]).strip()
    if deictic and (prev_air or "").strip():
        return (prev_air or "").strip()
    hist = _infer_models_from_history(history)
    if hist:
        return hist[-1]
    return (prev_air or "").strip() or None


def merge_consultant_conversation_state(
    prev_in: Optional[Dict[str, Any]],
    *,
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    entity_models: Optional[List[str]] = None,
    hybrid_kind: Optional[str] = None,
    fine_intent: Optional[str] = None,
    response_mode: Optional[str] = None,
    user_wants_gallery: bool = False,
    mission_hint: Optional[str] = None,
    conversation_guard_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Produce the next ``consultant_conversation_state`` snapshot (JSON-friendly).

    ``prev_in`` is typically the client's last ``data_used["consultant_conversation_state"]``,
    optionally merged with history inference when absent.
    """
    q = (query or "").strip()
    ql = q.lower()

    if prev_in is None:
        prev = default_consultant_conversation_state()
        hist_infer = _infer_models_from_history(history if isinstance(history, list) else None)
        if hist_infer and not prev.get("current_aircraft_reference"):
            prev["current_aircraft_reference"] = hist_infer[-1]
    else:
        prev = _sanitize_client_state(prev_in)

    if _RESET_RE.search(q):
        prev = default_consultant_conversation_state()

    deictic = bool(_DEICTIC_FOLLOWUP_RE.search(q)) and len(q) < 140
    prev_air = prev.get("current_aircraft_reference")

    air = _primary_aircraft_reference_fixed(
        entity_models,
        q,
        history if isinstance(history, list) else None,
        str(prev_air) if prev_air else None,
        deictic,
    )

    st_new, cab_new = _infer_style_and_cabin(ql)
    user_style = st_new or prev.get("user_style")
    cabin_pref = cab_new or prev.get("current_cabin_preference")

    vis = _infer_visual_intent(q)
    if user_wants_gallery and not vis:
        vis = "cabin interior"
    if deictic and not vis and prev.get("current_visual_intent"):
        vis = prev.get("current_visual_intent")
    elif vis is None:
        vis = prev.get("current_visual_intent")

    budget = _infer_budget(q) or prev.get("current_budget")
    pax = _infer_pax(q) or prev.get("current_passenger_count")
    mission = _infer_mission(q, mission_hint) or prev.get("current_mission")

    conv_mode = _map_conversation_mode(
        fine_intent=fine_intent,
        hybrid_kind=hybrid_kind,
        response_mode=response_mode,
        ql=ql,
    )
    if (conversation_guard_type or "").strip().lower() in (
        "greeting",
        "small_talk",
        "identity_question",
        "non_aviation_general",
    ):
        conv_mode = "browsing"

    return {
        "user_style": user_style,
        "current_aircraft_reference": air,
        "current_visual_intent": vis,
        "current_budget": budget,
        "current_mission": mission,
        "current_passenger_count": pax,
        "current_cabin_preference": cabin_pref,
        "conversation_mode": conv_mode,
    }


def format_conversation_state_for_system_prompt(state: Dict[str, Any]) -> str:
    """Internal-only block for the consultant system prompt."""
    blocks: List[str] = []
    mem = state.get("conversation_memory") if isinstance(state, dict) else None
    if isinstance(mem, dict) and mem:
        try:
            from services.conversation_state_engine.prompt_block import format_memory_prompt_block
            from services.conversation_state_engine.schemas import memory_from_dict

            pb = format_memory_prompt_block(memory_from_dict(mem))
            if pb:
                blocks.append(pb)
        except Exception:
            pass
    if isinstance(state, dict) and any(state.get(k) for k in _STATE_KEYS):
        lines = [f"- **{k}:** {state.get(k)}" for k in _STATE_KEYS if state.get(k)]
        if lines:
            blocks.append(
                "\n\n**Live conversation state (legacy cues — internal):**\n"
                + "\n".join(lines)
            )
    if not blocks:
        return ""
    return "".join(blocks) + (
        "\n- Treat this thread as **continuous** unless the user clearly pivots; "
        "do not restart as if there were no prior context.\n"
    )


def finalize_consultant_conversation_state(
    data_used: Dict[str, Any],
    client_state: Optional[Dict[str, Any]],
    *,
    query: str,
    history: Optional[List[Dict[str, str]]],
    entity_models: Optional[List[str]] = None,
    hybrid_kind: Optional[str] = None,
    fine_intent: Optional[str] = None,
    response_mode: Optional[str] = None,
    user_wants_gallery: bool = False,
    mission_hint: Optional[str] = None,
    conversation_guard_type: Optional[str] = None,
    continuity_state: Optional[Dict[str, Any]] = None,
    intent_persistence_state: Optional[Dict[str, Any]] = None,
    refinement_type: str = "none",
    routing_hint: str = "",
) -> Dict[str, Any]:
    """Mutates ``data_used`` in place and returns the new state dict."""
    merged = merge_consultant_conversation_state(
        client_state,
        query=query,
        history=history,
        entity_models=entity_models,
        hybrid_kind=hybrid_kind,
        fine_intent=fine_intent,
        response_mode=response_mode,
        user_wants_gallery=user_wants_gallery,
        mission_hint=mission_hint,
        conversation_guard_type=conversation_guard_type,
    )
    if continuity_state is not None:
        merged["continuity"] = continuity_state
    elif isinstance(client_state, dict) and isinstance(client_state.get("continuity"), dict):
        merged["continuity"] = client_state["continuity"]
    if intent_persistence_state is not None:
        merged["intent_persistence"] = intent_persistence_state
    elif isinstance(client_state, dict) and isinstance(client_state.get("intent_persistence"), dict):
        merged["intent_persistence"] = client_state["intent_persistence"]

    try:
        from services.conversation_state_engine import run_conversation_state_turn, sync_legacy_flat_fields

        mem_bundle = run_conversation_state_turn(
            query=query or "",
            client_conversation_state=merged,
            continuity_serialized=merged.get("continuity") if isinstance(merged.get("continuity"), dict) else continuity_state,
            intent_resolved=intent_persistence_state,
            refinement_type=refinement_type,
            entity_models=entity_models,
            user_wants_gallery=user_wants_gallery,
            mission_hint=mission_hint,
            routing_hint=routing_hint,
        )
        merged["conversation_memory"] = mem_bundle.serialized
        for k, v in sync_legacy_flat_fields(mem_bundle.state).items():
            if v is not None:
                merged[k] = v
        data_used["conversation_memory"] = mem_bundle.serialized
        data_used["conversation_state_engine"] = {
            "previous_state": mem_bundle.previous_snapshot,
            "resolved_state": mem_bundle.serialized,
            "inherited_fields": mem_bundle.inherited_fields,
            "decayed_fields": mem_bundle.decayed_fields,
            "memory_stack": mem_bundle.state.memory_stack,
        }
    except Exception:
        pass

    data_used["consultant_conversation_state"] = merged
    return merged
