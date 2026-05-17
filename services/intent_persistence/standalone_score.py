"""Confidence that the latest user line has standalone retrieval meaning."""

from __future__ import annotations

import re
from typing import Optional

from .inheritance import is_contextual_followup_query, query_lacks_standalone_entity
from .schemas import PersistentIntentState


def score_standalone_confidence(
    query: str,
    *,
    prev: Optional[PersistentIntentState],
) -> float:
    q = (query or "").strip()
    if not q:
        return 0.0

    score = 0.55

    if not query_lacks_standalone_entity(q):
        score = 0.92
    elif len(q) > 160:
        score = 0.78
    elif len(q) > 80:
        score = 0.65

    if prev and is_contextual_followup_query(q, prev):
        score = min(score, 0.28)

    if re.search(
        r"\b(show\s+me|photos?|pictures?|interior|cabin|cockpit|gallery|versus|vs\.?|compare)\b",
        q,
        re.I,
    ):
        if query_lacks_standalone_entity(q) and prev and (prev.active_aircraft or prev.active_tail):
            score = min(score, 0.35)

    if re.search(r"\b(faa|n-number|registration\s+lookup|serial\s+number)\b", q, re.I):
        score = max(score, 0.85)

    return max(0.0, min(1.0, round(score, 3)))
