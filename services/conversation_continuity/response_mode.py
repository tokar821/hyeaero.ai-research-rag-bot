"""Continuity-aware response shaping (orthogonal to ConsultantResponseMode elsewhere)."""

from __future__ import annotations

import re

from .schemas import ContinuityResponseMode


def resolve_continuity_response_mode(query: str, *, inherited: ContinuityResponseMode) -> ContinuityResponseMode:
    ql = (query or "").strip().lower()
    if not ql:
        return inherited

    if re.search(r"\bdon'?t\s+explain\b|\bno\s+walls?\s+of\s+text\b|\bjust\s+show\b|\bvisuals\s+only\b|\bminimal\s+(?:text|words)\b", ql):
        return ContinuityResponseMode.VISUAL_ONLY

    if re.search(r"\bcompare\b|\bversus\b|\bvs\.?\b", ql):
        return ContinuityResponseMode.COMPARISON_MODE

    if re.search(r"\b(specs?\b|ranges?\b|cruise\s|mtow|nacelle|faa)\b|\btechnical\b", ql):
        return ContinuityResponseMode.TECHNICAL_MODE

    if re.search(
        r"\b(show\s+me|show\s+us|photos?|pictures?|images?|cockpit|cabin\b|interior|gallery)\b",
        ql,
    ) and len(ql) < 180:
        return ContinuityResponseMode.SHORT_CAPTION

    return inherited
