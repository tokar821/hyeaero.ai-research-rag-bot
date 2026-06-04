"""Current-turn entity scope for retrieval isolation (Phase 10)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

_DEICTIC_TAIL_FOLLOWUP = re.compile(
    r"(?is)\b("
    r"what(?:'s|\s+is)\s+(?:the\s+)?(?:ask(?:ing)?\s+price|asking\s+price|status|registration)|"
    r"how\s+many\s+hours|"
    r"who\s+owns\s+(?:it|this|that|the\s+aircraft)|"
    r"show\s+(?:me\s+)?(?:photos?|pictures?|images?|gallery)|"
    r"(?:what(?:'s|\s+is)|how\s+much\s+is)\s+(?:the\s+)?(?:price|worth|value)|"
    r"what(?:'s|\s+is)\s+(?:the\s+)?(?:serial|tail|n[-\s]?number)|"
    r"(?:biggest\s+)?acquisition\s+risks?|"
    r"(?:what\s+are\s+)?(?:the\s+)?risks?\s+(?:on|for)|"
    r"engine\s+program(?:ming)?|"
    r"apu\s+program|"
    r"enrolled\s+on|"
    r"compare\s+.+\s+(?:against|to|vs)|"
    r"everything\s+about|"
    r"tell\s+me\s+(?:everything|all)"
    r")\b"
)

_COMPARISON_RE = re.compile(r"(?is)\b(?:vs\.?|versus|compare)\b")


@dataclass(frozen=True)
class EntityScope:
    scope_type: str  # aircraft_model, tail, comparison, deictic, none
    scope_value: Optional[str]
    scope_source: str  # current_turn, history, memory

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def is_deictic_tail_followup(query: str) -> bool:
    """True when the user likely refers to a prior tail/listing without naming a new aircraft."""
    q = (query or "").strip()
    if not q:
        return False
    if _DEICTIC_TAIL_FOLLOWUP.search(q):
        return True
    if len(q) < 100 and re.search(
        r"(?is)^\s*(?:and\s+)?(?:the\s+)?(?:hours|status|price|photos?|pictures?|registration)\s*\??\s*$",
        q,
    ):
        return True
    return False


def _resolve_current_turn_model(user_message: str) -> Optional[str]:
    raw = (user_message or "").strip()
    if not raw:
        return None
    try:
        from rag.consultant_query_expand import _detect_manufacturers, _detect_models
        from services.searchapi_aircraft_images import (
            compose_manufacturer_model_phrase,
            normalize_aircraft_name,
        )

        mans = _detect_manufacturers(raw.lower())
        mdls = _detect_models(raw) or []
        mm = compose_manufacturer_model_phrase(mans[0] if mans else "", mdls[0] if mdls else "").strip()
        mm = normalize_aircraft_name(mm) if mm else ""
        if not mm and mdls:
            mm = normalize_aircraft_name(mdls[0]) if mdls[0] else ""
        mm = (mm or "").strip()
        return mm if len(mm) >= 3 else (str(mdls[0]).strip() if mdls else None)
    except Exception:
        return None


def resolve_entity_scope(
    user_message: str,
    *,
    isolated_message: Optional[str] = None,
    history_allowed: bool = True,
) -> EntityScope:
    """
    Resolve authoritative entity scope for the current turn.

    Explicit aircraft or tail in the latest user message always wins over history.
    """
    del history_allowed  # reserved for callers; scope is current-turn first
    msg = (isolated_message or user_message or "").strip()

    try:
        from rag.aviation_tail import find_strict_tail_candidates_in_text

        tails = find_strict_tail_candidates_in_text(msg)
        if tails:
            return EntityScope(scope_type="tail", scope_value=tails[0], scope_source="current_turn")
    except Exception:
        pass

    if _COMPARISON_RE.search(msg):
        model = _resolve_current_turn_model(msg)
        if model:
            return EntityScope(scope_type="comparison", scope_value=model, scope_source="current_turn")
        return EntityScope(scope_type="comparison", scope_value=None, scope_source="current_turn")

    try:
        from rag.consultant_query_anchor import latest_message_anchors_aircraft_identity

        if latest_message_anchors_aircraft_identity(msg):
            model = _resolve_current_turn_model(msg)
            if model:
                return EntityScope(
                    scope_type="aircraft_model",
                    scope_value=model,
                    scope_source="current_turn",
                )
    except Exception:
        pass

    if is_deictic_tail_followup(msg):
        return EntityScope(scope_type="deictic", scope_value=None, scope_source="current_turn")

    return EntityScope(scope_type="none", scope_value=None, scope_source="current_turn")


def normalize_aircraft_label(label: Optional[str]) -> str:
    text = (label or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def aircraft_identities_conflict(model_a: Optional[str], model_b: Optional[str]) -> bool:
    """True when two canonical model labels refer to different aircraft types."""
    a = normalize_aircraft_label(model_a)
    b = normalize_aircraft_label(model_b)
    if not a or not b:
        return False
    if a == b:
        return False
    if a in b or b in a:
        return False
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if a_tokens & b_tokens:
        return False
    return True


def should_release_tail_on_model_switch(
    explicit_model: str,
    prev_tail_aircraft: Optional[str],
) -> bool:
    """Release a stale tail lock when the user names a different aircraft this turn."""
    model = (explicit_model or "").strip()
    if not model:
        return False
    if not (prev_tail_aircraft or "").strip():
        return True
    return aircraft_identities_conflict(model, prev_tail_aircraft)


def history_allowed_for_tail_resolution(user_message: str) -> bool:
    """False when the latest user line anchors a new aircraft identity (no history tails)."""
    try:
        from rag.consultant_query_anchor import latest_message_anchors_aircraft_identity

        return not latest_message_anchors_aircraft_identity(user_message or "")
    except Exception:
        return True
