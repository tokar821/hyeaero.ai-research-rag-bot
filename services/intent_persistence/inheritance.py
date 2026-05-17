"""Detect short follow-ups that must inherit prior conversational intent."""

from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

from .schemas import ConversationGoal, IntentResponseMode, PersistentIntentState


# Short lines that rarely carry standalone retrieval meaning.
_CONTEXTUAL_FOLLOWUP_RE = re.compile(
    r"(?is)\b("
    r"actually\b.*\b(bigger|larger)|"
    r"(?:something\s+)?bigger|larger|more\s+space|step\s*up|"
    r"more\s+modern|less\s+corporate|younger\s+feeling|"
    r"cheaper|less\s+expensive|tighter\s+budget|"
    r"show\s+cockpit|cockpit\s+too|and\s+cockpit|flight\s+deck|"
    r"bedroom\s+setup|box\s+spring|divan|berth|"
    r"show\s+me\s+(?:that|this|it|more)?|just\s+show|"
    r"same\s+(?:jet|plane|aircraft|cabin)|that\s+one|"
    r"interior\s+too|cabin\s+too|exterior\s+too|"
    r"more\s+like\s+that|like\s+that|not\s+that\s+old"
    r")\b",
)

_DEICTIC_ONLY_RE = re.compile(
    r"(?is)^\s*("
    r"(?:show\s+me\s+)?(?:interior|cabin|cockpit|exterior)\s*\??|"
    r"cockpit\s+too(?:\s+please)?|"
    r"(?:actually\s+)?bigger|more\s+modern|cheaper|younger\s+feeling|"
    r"less\s+corporate|bedroom\s+setup|show\s+cockpit"
    r")\s*[\.\!]?\s*$",
)


def query_lacks_standalone_entity(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return True
    try:
        from rag.query_isolation_engine import _detect_aircraft_in_text

        if _detect_aircraft_in_text(q):
            return False
    except Exception:
        pass
    return True


def is_contextual_followup_query(query: str, prev: Optional[PersistentIntentState]) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    has_anchor = bool(
        prev
        and (
            (prev.active_aircraft or "").strip()
            or (prev.active_tail or "").strip()
            or prev.response_mode == IntentResponseMode.IMAGE_SHOWCASE
            or prev.current_conversation_goal == ConversationGoal.VISUAL_GALLERY
            or (prev.active_budget_usd or 0) > 0
            or bool(prev.aesthetic_preferences)
        )
    )
    if not has_anchor:
        return False
    if _DEICTIC_ONLY_RE.match(q):
        return True
    if len(q) < 120 and _CONTEXTUAL_FOLLOWUP_RE.search(q):
        return True
    if len(q) < 80 and re.search(r"\b(too|also|same|that|this|it)\b", q, re.I):
        return bool(re.search(r"\b(cockpit|cabin|interior|jet|plane|aircraft|photo|gallery)\b", q, re.I))
    return False


def inherited_field_names(
    prev: PersistentIntentState,
    resolved: PersistentIntentState,
) -> List[str]:
    out: List[str] = []
    pairs = (
        ("active_aircraft", prev.active_aircraft, resolved.active_aircraft),
        ("active_tail", prev.active_tail, resolved.active_tail),
        ("active_topic", prev.active_topic, resolved.active_topic),
        ("active_visual_focus", prev.active_visual_focus, resolved.active_visual_focus),
        ("active_budget_usd", prev.active_budget_usd, resolved.active_budget_usd),
        ("response_mode", prev.response_mode.value, resolved.response_mode.value),
        ("comparison_target", prev.comparison_target, resolved.comparison_target),
        ("current_conversation_goal", prev.current_conversation_goal.value, resolved.current_conversation_goal.value),
    )
    for name, old, new in pairs:
        if old is None and new is not None:
            out.append(name)
        elif old is not None and new is not None and str(old) != str(new):
            if name not in ("response_mode", "current_conversation_goal"):
                out.append(name)
    if resolved.aesthetic_preferences and resolved.aesthetic_preferences != prev.aesthetic_preferences:
        out.append("aesthetic_preferences")
    return list(dict.fromkeys(out))[:24]


def merge_prev_snapshot(raw: Optional[dict]) -> Optional[PersistentIntentState]:
    if not isinstance(raw, dict):
        return None
    direct = raw.get("intent_persistence")
    if isinstance(direct, dict):
        return intent_state_from_dict_safe(direct)
    cont = raw.get("continuity")
    if isinstance(cont, dict):
        return continuity_dict_to_intent(cont)
    return None


def intent_state_from_dict_safe(raw: dict) -> PersistentIntentState:
    from .schemas import intent_state_from_dict

    return intent_state_from_dict(raw)


def continuity_dict_to_intent(cont: dict) -> PersistentIntentState:
    from services.conversation_continuity.schemas import continuity_state_from_dict

    c = continuity_state_from_dict(cont)
    mode = IntentResponseMode.CONSULTANT_MODE
    rm = (c.response_mode.value if c.response_mode else "") or ""
    if rm == "visual_only":
        mode = IntentResponseMode.IMAGE_SHOWCASE
    elif rm == "short_caption":
        mode = IntentResponseMode.SHORT_CAPTION
    elif rm == "comparison_mode":
        mode = IntentResponseMode.COMPARISON_MODE
    elif rm == "technical_mode":
        mode = IntentResponseMode.TECHNICAL_MODE

    goal = ConversationGoal.UNKNOWN
    if c.last_requested_view or mode in (IntentResponseMode.IMAGE_SHOWCASE, IntentResponseMode.SHORT_CAPTION):
        goal = ConversationGoal.VISUAL_GALLERY
    elif c.last_refinement and c.last_refinement.type == "comparison_anchor":
        goal = ConversationGoal.COMPARE_MODELS
    elif c.current_aircraft:
        goal = ConversationGoal.EXPLORE_MODEL

    comp = None
    if c.last_refinement and c.last_refinement.type == "comparison_anchor":
        comp = c.last_refinement.reference_aircraft

    return PersistentIntentState(
        active_aircraft=(c.current_aircraft or "").strip() or None,
        active_tail=(c.current_tail or "").strip() or None,
        active_topic=(c.last_refinement.type if c.last_refinement else None),
        active_visual_focus=c.last_requested_view,
        active_budget_usd=c.buyer_direction.budget_usd_approx if c.buyer_direction else None,
        response_mode=mode,
        aesthetic_preferences=list(c.style_preferences or [])[:48],
        negative_preferences=list(c.negative_preferences or [])[:24],
        comparison_target=comp,
        current_conversation_goal=goal,
        last_refinement_type=c.last_refinement.type if c.last_refinement else None,
    )

