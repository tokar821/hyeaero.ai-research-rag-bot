"""Filter Phly / lookup tokens that are aircraft model numbers, not serials or tails."""

from __future__ import annotations

import re
from typing import List

# Manufacturer / family token before a 3–4 digit model suffix (Challenger 601, Citation 650, …)
_MODEL_FAMILY_BEFORE_NUMBER = re.compile(
    r"\b(?:challenger|citation|gulfstream|falcon|global\s*\d+|g\s*\d+|phenom|"
    r"praetor|learjet|haw(?:ker)?|pilatus|embraer|legacy\s*\d+|"
    r"cessna|king\s*air|beechcraft|westwind|sovereign|latitude|longitude|"
    r"mustang|denali|tbm\b|pc-?\s*\d+|cj\d+|bravo|excel|jet|ultra|x+\b|"
    r"bombardier|dassault)"
    r"\s+(\d{3,4}[A-Z]{0,3})\b",
    re.IGNORECASE,
)


def is_likely_aircraft_model_number_token(token: str, query: str) -> bool:
    """True if ``token`` is probably a model series number (601, 604, 350), not an MSN/tail fragment."""
    t = (token or "").strip()
    if not re.match(r"^\d{3,4}[A-Z]{0,2}$", t, re.I):
        return False
    q = query or ""
    if _MODEL_FAMILY_BEFORE_NUMBER.search(q):
        for m in _MODEL_FAMILY_BEFORE_NUMBER.finditer(q):
            if m.group(1).upper().startswith(t.upper()) or t.upper() == m.group(1).upper():
                return True
    # Standalone "601" after manufacturer name on same line (slack spacing)
    if re.search(
        rf"\b(?:challenger|citation|gulfstream|falcon|global|phenom|learjet|g\s*\d+|cj\d+)\s+{re.escape(t)}\b",
        q,
        re.I,
    ):
        return True
    return False


def filter_phly_lookup_tokens(tokens: List[str], query: str) -> List[str]:
    """Drop tokens that would confuse registry/serial lookup with model numbers."""
    q = query or ""
    out: List[str] = []
    for raw in tokens or []:
        t = (raw or "").strip()
        if not t:
            continue
        if is_likely_aircraft_model_number_token(t, q):
            continue
        out.append(raw)
    return out
