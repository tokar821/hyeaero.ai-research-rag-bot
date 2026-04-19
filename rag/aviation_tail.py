"""
Strict civil registration detection — avoids false positives from aircraft model numbers (601, 604, 2000).

Valid examples: N123AB, N12345, G-ABCD, PR-CCA, C-GUGU, TC-KEA, FL-1185, VH-ABC, etc.
"""

from __future__ import annotations

import re
from typing import List, Optional

# U.S. civil: N + at least two characters; avoids bare "N" or "N1". Model numbers like "601" have no N prefix.
_US_N_NUMBER = re.compile(r"\bN[1-9A-Z][A-Z0-9]{1,5}\b", re.IGNORECASE)
# Same, glued to a preceding lowercase letter (e.g. "showN140NE") — \b does not sit between "w" and "N".
# Require at least one digit in the mark (avoids "thanks" → nks, "France" → nce).
_US_N_NUMBER_AFTER_LOWER = re.compile(
    r"(?<=(?-i:[a-z]))N(?=[A-Z0-9]*\d)[A-Z0-9]{2,6}\b",
    re.IGNORECASE,
)

# International: hyphenated civil marks (not digit-leading MSN like 525-0444).
# Includes Brazil (PR/PP/PT), Liechtenstein-style FL-, etc.; TC- (Turkey) and many others.
_INTL_MARK = re.compile(
    r"\b(?:G|D|F|I|OO|LX|TC|CN|HK|SX|9V|VH|XA|V|ZK|JA|ZS|HA|OE|YR|B|CF|PR|PP|PT|FL)-[A-Z0-9]{2,5}\b",
    re.IGNORECASE,
)

# Canadian civil marks C-F…, C-G…, C-I… — avoids treating "C-130" as a tail (digit after C-).
_CANADIAN_CIVIL_MARK = re.compile(r"\bC-[FGI][A-Z0-9]{2,4}\b", re.IGNORECASE)

# Gallery-only: any plausible U.S. ``N`` mark with ≥1 digit (includes marks that fail stricter airborne rules).
_US_N_GALLERY_LOOSE = re.compile(r"\bN(?=[A-Z0-9]*\d)[A-Z0-9]{2,6}\b", re.IGNORECASE)


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

    for m in _US_N_NUMBER_AFTER_LOWER.finditer(blob):
        add(m.group(0).upper())

    for m in _INTL_MARK.finditer(blob):
        add(m.group(0))

    for m in _CANADIAN_CIVIL_MARK.finditer(blob):
        add(m.group(0))

    return out


def find_loose_us_n_tail_tokens_in_text(blob: str) -> List[str]:
    """
    U.S. ``N`` marks for **visual / gallery** routing only — requires at least one digit in the mark
    so hostnames like ``planespotters.net`` do not yield a fake ``NET`` tail.
    """
    if not (blob or "").strip():
        return []
    seen: set[str] = set()
    out: List[str] = []

    def add(raw: str) -> None:
        u = normalize_tail_token(raw)
        if len(u) < 3 or len(u) > 8:
            return
        if u in seen:
            return
        seen.add(u)
        out.append(u)

    for m in _US_N_GALLERY_LOOSE.finditer(blob):
        add(m.group(0))
    return out


def find_visual_gallery_tail_candidates(
    query: str,
    history: Optional[List[dict]] = None,
    *,
    max_history_messages: int = 16,
) -> List[str]:
    """
    Tails that should trigger **strict gallery** image matching: strict civil marks first; else any
    plausible ``N``+digit mark on the latest **user** text (then recent user history).
    """
    strict = find_strict_tail_candidates(query, history)
    if strict:
        return strict
    q = (query or "").strip()
    loose = find_loose_us_n_tail_tokens_in_text(q)
    if loose:
        return loose
    if history:
        for h in reversed(history[-max_history_messages:]):
            if (h.get("role") or "").strip().lower() != "user":
                continue
            loose_h = find_loose_us_n_tail_tokens_in_text(h.get("content") or "")
            if loose_h:
                return loose_h
    return []


def is_invalid_placeholder_us_n_tail(tail: str) -> bool:
    """
    True for U.S. marks that look like placeholders (e.g. ``N00000`` / ``N0000``) — not credible for retrieval.

    Real marks can contain zeros; this only flags **all-zero** suffixes after ``N``.
    """
    t = normalize_tail_token(tail)
    if not t.startswith("N") or len(t) < 4:
        return False
    suf = t[1:]
    if not suf.isdigit():
        return False
    return set(suf) == {"0"}


_US_N_STRICT_FULL = re.compile(r"^N[1-9A-Z][A-Z0-9]{1,5}$", re.IGNORECASE)


def registration_format_kind(mark: str) -> str:
    """
    Structural class of a single registration token (does **not** assert the mark exists on a registry).

    Returns one of:
    ``US_N_STRICT``, ``US_N_LOOSE``, ``ICAO_STYLE``, ``CANADIAN``, ``US_N_PLACEHOLDER``, ``EMPTY``, ``UNKNOWN``.
    """
    m = normalize_tail_token(mark or "")
    if not m:
        return "EMPTY"
    if is_invalid_placeholder_us_n_tail(m):
        return "US_N_PLACEHOLDER"
    if _CANADIAN_CIVIL_MARK.fullmatch(m):
        return "CANADIAN"
    if _INTL_MARK.fullmatch(m):
        return "ICAO_STYLE"
    if _US_N_STRICT_FULL.match(m):
        return "US_N_STRICT"
    if _US_N_GALLERY_LOOSE.fullmatch(m):
        return "US_N_LOOSE"
    return "UNKNOWN"


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
