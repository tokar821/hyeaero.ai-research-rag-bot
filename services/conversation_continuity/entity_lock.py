"""Entity locking: tails, serial patterns, listing cues."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .schemas import LockedEntity, LockedEntityType


_TAIL_RE = re.compile(r"\b([Nn])\s*(\d{1,6})(?:\s*([A-Za-z]{1,3}))?\b")
_SERIAL_RE = re.compile(r"\b(?:serial|msn|s\/n)\s*:?\s*(\d{3,6})\b", re.I)


def extract_tail_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = _TAIL_RE.search(text)
    if not m:
        return None
    suf = (m.group(3) or "").strip().upper()
    raw = f"{m.group(1).upper()}{m.group(2).strip()}{suf}"
    if raw.startswith("N") and len(raw) >= 2:
        return raw
    return None


def extract_serial(text: str) -> Optional[str]:
    m = _SERIAL_RE.search(text or "")
    return m.group(1) if m else None


def merge_entity_lock(
    prev: Optional[LockedEntity],
    *,
    query: str,
    strict_tail_candidates: Optional[List[str]] = None,
    explicit_model: Optional[str] = None,
) -> Optional[LockedEntity]:
    """Choose strongest lock for this turn (tail wins)."""
    q = (query or "").strip()
    tails = [t for t in (strict_tail_candidates or []) if str(t).strip()]
    tail_from_q = extract_tail_from_text(q)
    chosen_tail = tail_from_q or (tails[0] if tails else None)
    if chosen_tail:
        return LockedEntity(type=LockedEntityType.TAIL, value=chosen_tail.strip().upper(), locked_at_turn_hint=q[:120])
    ser = extract_serial(q)
    if ser:
        return LockedEntity(type=LockedEntityType.SERIAL, value=ser, locked_at_turn_hint=q[:120])
    if explicit_model and not prev:
        return LockedEntity(
            type=LockedEntityType.AIRCRAFT_MODEL,
            value=str(explicit_model).strip(),
            locked_at_turn_hint=q[:120],
        )
    if prev:
        # Keep lock unless superseded above
        if tail_from_q and prev.type != LockedEntityType.TAIL:
            return LockedEntity(type=LockedEntityType.TAIL, value=tail_from_q.strip().upper(), locked_at_turn_hint=q[:120])
        return prev
    return None


def explicit_aircraft_switch(query: str, prev_model: Optional[str]) -> Optional[str]:
    """Detect explicit new model in line (releases lock context)."""
    try:
        from rag.consultant_query_expand import _detect_models

        found = _detect_models(query or "") or []
        if found:
            new_m = str(found[0]).strip()
            if not prev_model or new_m.lower() != (prev_model or "").strip().lower():
                return new_m
    except Exception:
        pass
    return None
