"""Visual / emotional preference extraction (deterministic)."""

from __future__ import annotations

import re
from typing import List, Tuple


_NEG_HINTS = (
    ("less corporate", "corporate beige"),
    ("banker vibe", "corporate beige"),
    ("old-money", "corporate beige"),
    ("not tacky", "tacky trim"),
    ("not old", "dated interior"),
    ("not flashy", "loud finishes"),
)


def extract_preferences(query: str) -> Tuple[List[str], List[str]]:
    """Returns (positive_style_tokens, negative_style_tokens)."""
    ql = (query or "").lower()
    pos: List[str] = []
    neg: List[str] = []

    patterns = (
        (r"\b(?:luxury|five[- ]star)\s+hotel\b", "luxury hotel"),
        (r"\bfour\s+seasons\b", "luxury hotel"),
        (r"\bmore\s+like\s+four\s+seasons\b", "luxury hotel"),
        (r"\b(?:young\s+|\b|^)tech\s+ceo\b|\byoung\s+(?:professional|money)\b|\bminimalist\b", "young tech minimalist"),
        (r"\bambient\s+(?:light|lighting)\b", "ambient lighting"),
        (r"\bhuge\s+windows\b|\bbig\s+windows\b|\blots\s+of\s+glass\b", "large windows"),
        (r"\bwhite\s+interior\b|\bbeige\s+fleece\b|\bcream\b.*\bcabin\b", "light-neutral interior palette"),
        (r"\bmodern\b|\bcontemporary\b", "modern"),
        (r"\bprestige\b|\bimpact\b|\b(?:wow|dramatic)\b", "visual impact prestige"),
        (r"\bprivate\s+airline\b|\bcharter\s+aesthetic\b", "charter-program aesthetic"),
    )
    for pat, label in patterns:
        if re.search(pat, ql):
            pos.append(label)

    for phrase, canon in _NEG_HINTS:
        if phrase in ql:
            neg.append(canon)

    return pos, neg
