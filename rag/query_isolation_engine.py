"""
Query Isolation Engine (deterministic).

Goal: decide whether the latest user message should be interpreted as a NEW independent request
or a CONTEXTUAL follow-up that inherits the last known aircraft/entity from the thread.

Output shape (JSON-friendly dict):
{
  "mode": "NEW" | "CONTEXTUAL",
  "resolved_entity": "<aircraft>" | null
}

When mode == NEW: downstream logic should ignore conversation history (use only this message).
When mode == CONTEXTUAL: downstream logic should reuse the last known aircraft/entity.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


_DEICTIC_CONTEXTUAL = re.compile(
    r"(?is)(?:"
    r"^\s*(?:interior|cabin|cockpit|exterior)\s*\??\s*$"
    r"|^\s*(?:interior\?|cabin\?|cockpit\?|exterior\?)\s*$"
    r"|^\s*show\s+me\s*(?:that|this|it|them|the\s+same|that\s+one)\b"
    r"|^\s*show\s+me\s*$"
    r"|^\s*show\s+me\s+inside\s*\??\s*$"
    r"|^\s*(?:inside)\s*\??\s*$"
    r"|^\s*let\s+me\s+see\s*$"
    r"|^\s*let\s+me\s+see\s+(?:it|that|this|them|that\s+one)\b"
    r"|^\s*(?:can|could)\s+i\s+see\s+(?:it|that|this|them|that\s+one)\b"
    r"|^\s*compare\s+(?:it|that|this|them|those)\b"
    r"|^\s*same\s+(?:one|jet|aircraft|cabin|interior|cockpit)\b"
    r"|^\s*that\s+one\s*$"
    r"|^\s*(?:show\s+me\s+)?again\s*\??\s*$"
    r"|^\s*(?:show|see)\s+both\s+(?:cabins|interiors|cockpits)\b"
    r"|^\s*both\s+(?:cabins|interiors|cockpits)\b"
    r"|^\s*(?:show\s+me\s+inside|inside)\s+best\s+(?:option|one)\b"
    r"|^\s*best\s+(?:option|one)\s*$"
    r")"
)

_NEW_CONCEPT = re.compile(
    r"(?is)\b(?:"
    r"best\s+cabin|"
    r"best\s+private\s+jet|"
    r"cheap\s+jet|"
    r"budget\s+jet|"
    r"luxury|premium|hotel\s+feel|"
    r"compare\b(?!\s+(?:it|that|this|them|those)\b)"
    r")\b"
)


def _detect_aircraft_in_text(text: str) -> Optional[str]:
    """
    Return a normalized aircraft identity string from **this text only**.

    Preference order:
    - strict tail token (e.g. N807JS)
    - normalized marketing model phrase (e.g. Gulfstream G650)
    """
    raw = (text or "").strip()
    if not raw:
        return None

    try:
        from rag.aviation_tail import find_strict_tail_candidates_in_text, normalize_tail_token

        tails = find_strict_tail_candidates_in_text(raw)
        if tails:
            return normalize_tail_token(tails[0])
    except Exception:
        pass

    try:
        from rag.consultant_query_expand import _detect_manufacturers, _detect_models
        from services.searchapi_aircraft_images import (
            compose_manufacturer_model_phrase,
            normalize_aircraft_name,
        )

        mans = _detect_manufacturers(raw.lower())
        mdls = _detect_models(raw)
        mm = compose_manufacturer_model_phrase(mans[0] if mans else "", mdls[0] if mdls else "").strip()
        mm = normalize_aircraft_name(mm) if mm else ""
        if not mm and mdls:
            mm = normalize_aircraft_name(mdls[0]) if mdls[0] else ""
        mm = (mm or "").strip()
        if len(mm) >= 3:
            # Heuristic validation: reject non-aircraft English tokens accidentally detected as "models"
            # (e.g. "nonstop", "best", "option") so we don't poison contextual resolution.
            mm_l = mm.lower()
            if mm_l in {"nonstop", "stop", "best", "option", "inside"}:
                return None
            looks_real = bool(
                re.search(r"\d", mm)
                or re.search(
                    r"\b(challenger|citation|gulfstream|falcon|global|embraer|legacy|praetor|phenom|"
                    r"learjet|hawker|king\s*air|pilatus|pc-?12|pc-?24|cessna|bombardier|dassault)\b",
                    mm_l,
                    re.I,
                )
            )
            if looks_real:
                return mm
    except Exception:
        pass

    return None


def _last_thread_entity(history: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """Find the most recent aircraft identity mentioned in the thread."""
    if not history:
        return None
    # Most recent first.
    for h in reversed(history[-18:]):
        c = (h.get("content") or "").strip()
        if not c:
            continue
        ent = _detect_aircraft_in_text(c)
        if ent:
            return ent
    return None


def isolate_query_mode(
    user_query: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Optional[str]]:
    """
    Determine whether this query is NEW or CONTEXTUAL and resolve entity accordingly.
    """
    q = (user_query or "").strip()
    low = q.lower()

    explicit = _detect_aircraft_in_text(q)
    if explicit:
        return {"mode": "NEW", "resolved_entity": explicit}

    # If the user used deictic visual / pronoun style, this is contextual only if the thread has an entity.
    if _DEICTIC_CONTEXTUAL.search(q):
        # Special case: after "like G650 but cheaper" advisory, "show me inside" should anchor to the
        # first large-cabin alternative (G500) if it was presented in the thread.
        if re.fullmatch(r"(?is)\s*(?:show\s+me\s+inside|inside)\s*\??\s*$", q):
            try:
                thread_blob = " ".join((h.get("content") or "") for h in (history or [])[-24:] if isinstance(h, dict))
                tb = thread_blob.lower()
                if ("g650" in tb or "g 650" in tb) and ("cheaper" in tb or "less expensive" in tb):
                    if "g500" in tb:
                        return {"mode": "CONTEXTUAL", "resolved_entity": "Gulfstream G500"}
                    if "falcon 7x" in tb:
                        return {"mode": "CONTEXTUAL", "resolved_entity": "Falcon 7X"}
                    if "challenger 650" in tb:
                        return {"mode": "CONTEXTUAL", "resolved_entity": "Challenger 650"}
            except Exception:
                pass
        # Special case: "best option" should resolve to the last explicitly stated best-pick in thread,
        # not merely the last mentioned aircraft in a shortlist.
        if re.search(r"(?i)\bbest\s+(?:option|one)\b", q):
            try:
                for h in reversed((history or [])[-24:]):
                    c = (h.get("content") or "").strip()
                    if not c:
                        continue
                    if re.search(r"(?i)\bbest\s+option\b", c):
                        # Prefer the aircraft mentioned immediately after "Best option" phrasing.
                        m_best = re.search(r"(?is)\bbest\s+option[^:\n]*:\s*(.+)$", c, re.I)
                        slice_c = (m_best.group(1).strip() if m_best else c).strip()
                        slice_c = re.sub(r"[*_`]", "", slice_c).strip()
                        ent2 = _detect_aircraft_in_text(slice_c) or _detect_aircraft_in_text(c)
                        if ent2:
                            return {"mode": "CONTEXTUAL", "resolved_entity": ent2}
            except Exception:
                pass
        ent = _last_thread_entity(history)
        if ent:
            return {"mode": "CONTEXTUAL", "resolved_entity": ent}
        return {"mode": "NEW", "resolved_entity": None}

    # If a new concept is introduced and there's no explicit entity in this message, treat as NEW.
    if _NEW_CONCEPT.search(low):
        return {"mode": "NEW", "resolved_entity": None}

    # Default: if the query doesn't reference a previous entity and doesn't explicitly ask deictically,
    # treat as NEW. (This is intentionally conservative; contextual mode is only for clear follow-ups.)
    return {"mode": "NEW", "resolved_entity": None}

