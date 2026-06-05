"""
Decide when the **latest user message alone** pins aircraft identity for retrieval.

When true, gallery / photo Tavily / tail-candidate code must **not** inherit prior turns or stale
Phly rows — fixes wrong images after unrelated thread context (e.g. Challenger thread then "G650 but cheaper").
"""

from __future__ import annotations

import re
from typing import List, Optional

from rag.aviation_tail import find_strict_tail_candidates_in_text, normalize_tail_token


def latest_message_anchors_aircraft_identity(user_message: str) -> bool:
    """
    True when ``user_message`` by itself contains a credible tail/mark **or** recognized make/model tokens.

    Follow-ups like "show me that" / "same cockpit" (no aircraft in the latest line) return False so
    history-backed resolution still runs.
    """
    raw = (user_message or "").strip()
    if not raw:
        return False
    if find_strict_tail_candidates_in_text(raw):
        return True
    low = raw.lower()
    # Short deictic-only visual follow-ups — latest line does not anchor a type; keep history.
    if _DEICTIC_ONLY_VISUAL_FOLLOWUP.search(raw) and not _has_inline_aircraft_tokens(raw, low):
        return False
    try:
        from rag.consultant_query_expand import _detect_manufacturers, _detect_models

        if _detect_models(raw) or _detect_manufacturers(low):
            return True
    except Exception:
        pass
    # Shorthand marks not always in _MODEL_REGEX (defensive).
    if re.search(
        r"(?i)\b(?:g\s*[-.]?\s*6\d{2,3}|gulfstream\s+g\s*[-.]?\s*6\d{2}|global\s*\d{4}|"
        r"falcon\s*\d{1,4}|challenger\s*\d{3}|citation\s+|phenom\s*|learjet\s*|pc[\s-]?12|king\s*air)\b",
        raw,
    ):
        return True
    return False


_DEICTIC_ONLY_VISUAL_FOLLOWUP = re.compile(
    r"(?is)^\s*(?:(?:so|well|ok)\s*,?\s+)?(?:"
    r"\b(?:show|showing)\s+me\s+(?:that|this|it|them|the\s+same)\b"
    r"|\b(?:can|could)\s+i\s+see\s+(?:that|this|it|them)\b"
    r"|\blet\s+me\s+see\s+(?:that|this|it|them)\b"
    r"|\b(?:same|that)\s+(?:one|cockpit|cabin|interior|jet)\b"
    r"|\bmore\s+(?:photos?|pictures?|images?)\b"
    r")\s*\.?\s*$"
)


def _has_inline_aircraft_tokens(raw: str, low: str) -> bool:
    if find_strict_tail_candidates_in_text(raw):
        return True
    try:
        from rag.consultant_query_expand import _detect_manufacturers, _detect_models

        if _detect_models(raw) or _detect_manufacturers(low):
            return True
    except Exception:
        pass
    return bool(
        re.search(
            r"(?i)\b(?:g\s*[-.]?\s*6\d{2,3}|gulfstream|falcon|challenger|citation|phenom|learjet|global\s*\d)\b",
            raw,
        )
    )


_GALLERY_FULL_RE = re.compile(
    r"(?is)\b(?:every\s+(?:verified\s+)?image|all\s+(?:photos?|pictures?|images?)|"
    r"full\s+gallery|show\s+(?:me\s+)?everything|all\s+cabin\s+photos?)\b"
)


def user_wants_full_gallery(query: str) -> bool:
    return bool(_GALLERY_FULL_RE.search(query or ""))


def gallery_user_query_for_image_pipeline(raw_query: str, *, resolved_tail: Optional[str]) -> str:
    """
    When gallery routing resolved a **tail** from thread context but the latest line is deictic
    (\"can I see that\"), append the tail so SearchAPI / image-query layers see an explicit mark.

    When the user asks for photos of a specific registration, prefer a tail-led search string.
    """
    q = (raw_query or "").strip()
    t = normalize_tail_token(resolved_tail or "")
    if not t or len(t) < 3:
        return q
    compact_q = re.sub(r"\s+", "", (q or "").upper())
    tail_in_line = t.replace(" ", "") in compact_q
    if tail_in_line and re.search(r"(?is)\b(?:show|see|photos?|pictures?|images?|gallery)\b", q):
        low = q.lower()
        if re.search(r"(?is)\b(?:cabin|cabine|interior|salon|layout)\b", low):
            return f"{t} cabin interior photos"
        if re.search(r"(?is)\bcockpit\b", low):
            return f"{t} cockpit photos"
        if re.search(r"(?is)\b(?:exterior|outside)\b", low):
            return f"{t} exterior aircraft photos"
        return f"{t} aircraft photos"
    if tail_in_line:
        return q
    low = q.lower()
    if re.search(r"(?is)\b(?:interior|cabin|cabine)\s+too\b", low):
        return f"{t} cabin interior"
    if re.search(r"(?is)\bmore\s+(?:interior|cabin|cabine)\b", low):
        return f"{t} cabin interior"
    if re.search(r"(?is)\b(?:exterior|photos?|pictures?|images?)\s+too\b", low):
        return f"{t} aircraft photos"
    return f"{q} {t}".strip()


def effective_history_for_gallery_tail(
    user_message: str,
    history: Optional[List[dict]],
) -> Optional[List[dict]]:
    """
    Return ``history`` for tail discovery, or ``None`` to scan **only** the latest user line
    (no stale tails from earlier turns when the user named a model explicitly in this message).
    """
    if latest_message_anchors_aircraft_identity(user_message):
        return None
    return history
