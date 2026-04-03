"""
Strict civil registration detection — avoids false positives from aircraft model numbers (601, 604, 2000).

Valid examples: N123AB, N12345, G-ABCD, PR-CCA, C-GUGU, TC-KEA, FL-1185, VH-ABC, etc.
"""

from __future__ import annotations

import re
from typing import List, Optional

# U.S. civil: N + at least two characters; avoids bare "N" or "N1". Model numbers like "601" have no N prefix.
_US_N_NUMBER = re.compile(r"\bN[1-9A-Z][A-Z0-9]{1,5}\b", re.IGNORECASE)

# International: hyphenated civil marks (not digit-leading MSN like 525-0444).
# Includes Brazil (PR/PP/PT), Liechtenstein-style FL-, etc.; TC- (Turkey) and many others.
_INTL_MARK = re.compile(
    r"\b(?:G|D|F|I|OO|LX|TC|CN|HK|SX|9V|VH|XA|V|ZK|JA|ZS|HA|OE|YR|B|CF|PR|PP|PT|FL)-[A-Z0-9]{2,5}\b",
    re.IGNORECASE,
)

# Canadian civil marks C-F…, C-G…, C-I… — avoids treating "C-130" as a tail (digit after C-).
_CANADIAN_CIVIL_MARK = re.compile(r"\bC-[FGI][A-Z0-9]{2,4}\b", re.IGNORECASE)


def normalize_tail_token(raw: str) -> str:
    return (raw or "").strip().upper().replace(" ", "")


def find_strict_tail_candidates_in_text(blob: str) -> List[str]:
    """Return unique registration strings found in ``blob``."""
    if not (blob or "").strip():
        return []
    seen: set[str] = set()
    out: List[str] = []

    def add(t: str) -> None:
        u = normalize_tail_token(t)
        if len(u) < 3 or len(u) > 10:
            return
        if u in seen:
            return
        seen.add(u)
        out.append(u)

    for m in _US_N_NUMBER.finditer(blob):
        add(m.group(0).upper())

    for m in _INTL_MARK.finditer(blob):
        add(m.group(0))

    for m in _CANADIAN_CIVIL_MARK.finditer(blob):
        add(m.group(0))

    return out


def find_strict_tail_candidates(
    query: str,
    history: Optional[List[dict]] = None,
    *,
    max_history_messages: int = 16,
) -> List[str]:
    """Scan latest message and recent chat for valid registration patterns only."""
    seen: set[str] = set()
    ordered: List[str] = []

    def consume(blob: str) -> None:
        for t in find_strict_tail_candidates_in_text(blob or ""):
            if t not in seen:
                seen.add(t)
                ordered.append(t)

    consume(query or "")
    if history:
        for h in history[-max_history_messages:]:
            role = (h.get("role") or "").strip().lower()
            if role not in ("user", "assistant"):
                continue
            consume(h.get("content") or "")
    return ordered[:24]
