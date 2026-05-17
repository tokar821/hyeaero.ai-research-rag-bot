"""Post-generation enforcement for response modes."""

from __future__ import annotations

import re
from typing import Any, Dict

from .schemas import ResponseMode

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_DISCLAIMER_RE = re.compile(
    r"(?i)\b(closest\s+reference|unable\s+to\s+(?:find|locate)|can't\s+find\s+reliable|"
    r"may\s+not\s+be\s+exact|best\s+match\s+available|for\s+illustration\s+only)\b",
)


def _truncate_sentences(text: str, max_sentences: int) -> str:
    if max_sentences <= 0:
        return text
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    kept = [p for p in parts if p.strip()][:max_sentences]
    return " ".join(kept).strip()


def enforce_mode_on_answer(
    answer: str,
    *,
    mode: str,
    has_gallery: bool = False,
    max_sentences_hint: int = 8,
) -> str:
    """Deterministic last-mile shaping for routed modes."""
    s = (answer or "").strip()
    if not s:
        return s

    m = (mode or "").strip().lower()
    if m not in (ResponseMode.IMAGE_SHOWCASE.value, ResponseMode.TAIL_SPECIFIC.value):
        if m == ResponseMode.FOLLOWUP_CONTINUATION.value and has_gallery:
            s = _URL_RE.sub("", s)
            s = re.sub(r"\s{2,}", " ", s).strip()
        return s

    s = _URL_RE.sub("", s)
    s = _DISCLAIMER_RE.sub("", s)
    s = re.sub(r"\s{2,}", " ", s).strip()

    if has_gallery or m == ResponseMode.IMAGE_SHOWCASE.value:
        cap = min(max_sentences_hint, 2) if m == ResponseMode.IMAGE_SHOWCASE.value else max_sentences_hint
        s = _truncate_sentences(s, cap)

    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def enforce_from_data_used(answer: str, data_used: Dict[str, Any]) -> str:
    mode = str(
        (data_used or {}).get("consultant_response_mode_canonical")
        or (data_used or {}).get("consultant_response_mode")
        or ""
    )
    router = (data_used or {}).get("consultant_response_router")
    max_sent = 8
    if isinstance(router, dict):
        max_sent = int(router.get("max_sentences_hint") or max_sent)
    gallery = bool((data_used or {}).get("tavily_results")) or bool(
        (data_used or {}).get("consultant_tavily_gallery_forced")
    )
    imgs = (data_used or {}).get("aircraft_images")
    if isinstance(imgs, list) and imgs:
        gallery = True
    return enforce_mode_on_answer(
        answer,
        mode=mode,
        has_gallery=gallery,
        max_sentences_hint=max_sent,
    )
