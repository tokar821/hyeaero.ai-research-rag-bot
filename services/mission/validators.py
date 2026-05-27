"""
Schema validators for typed mission extraction.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

from services.mission.models import MissionProfile, Route
from services.mission.normalization import normalize_passenger_count, normalize_place

# Tokens that must never appear in route endpoints
_ENDPOINT_STOPWORDS: Set[str] = frozenset(
    """
    what would you like work full higher alternatives explore compared chartering
    efficiency ownership gulfstream and the for with from this that your our
    recommend best option help please show tell about into onto they them
    consultant insight bottom line assuming passengers typical business
    aircraft jet planes plane flights flying operate operating efficiency
    pax passengers seats executives people travel with
    """.split()
)

_GENERIC_NOUNS: Set[str] = frozenset(
    "mission route trip leg segment market data information advice help".split()
)

_UI_PHRASE_RE = re.compile(
    r"\b(what\s+would\s+you\s+like|click\s+here|learn\s+more|sign\s+up|"
    r"full\s+ownership\s*!|alternatives\s*!|explore\s*,)\b",
    re.I,
)

_CORRUPT_JOIN_RE = re.compile(r"(?:^|, )[A-Za-z], [a-z], ")


def _tokenize_endpoint(label: str) -> List[str]:
    return re.findall(r"[a-z]{2,}", (label or "").lower())


def is_valid_endpoint(label: str) -> bool:
    raw = (label or "").strip()
    if not raw or len(raw) < 3:
        return False
    if _CORRUPT_JOIN_RE.search(raw):
        return False
    if _UI_PHRASE_RE.search(raw):
        return False
    tokens = _tokenize_endpoint(raw)
    if not tokens:
        return False
    if any(t in _ENDPOINT_STOPWORDS for t in tokens):
        return False
    if all(t in _GENERIC_NOUNS for t in tokens):
        return False
    if len(tokens) == 1 and tokens[0] in _GENERIC_NOUNS:
        return False
    return True


def validate_route_candidate(origin: str, destination: str) -> Optional[Route]:
    """Return a typed Route or None if endpoints fail semantic validation."""
    o = normalize_place(origin)
    d = normalize_place(destination)
    if not is_valid_endpoint(o) or not is_valid_endpoint(d):
        return None
    if o.lower() == d.lower():
        return None
    try:
        return Route(origin=o, destination=d)
    except ValueError:
        return None


def validate_route_label(label: str) -> Optional[Route]:
    s = (label or "").strip().replace("→", "->")
    if "->" not in s or _CORRUPT_JOIN_RE.search(s):
        return None
    left, right = s.split("->", 1)
    return validate_route_candidate(left, right)


def dedupe_routes(routes: List[Route]) -> List[Route]:
    seen: Set[Tuple[str, str]] = set()
    out: List[Route] = []
    for r in routes:
        key = (r.origin.lower(), r.destination.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out[:6]


def validate_passengers(value: Optional[int]) -> Optional[int]:
    return normalize_passenger_count(value)


def validate_profile(profile: MissionProfile) -> MissionProfile:
    """
    Final pass: normalize passengers, dedupe routes, drop any invalid legs.
    """
    profile.passengers = validate_passengers(profile.passengers)
    clean_routes: List[Route] = []
    for r in profile.routes:
        v = validate_route_candidate(r.origin, r.destination)
        if v:
            clean_routes.append(v)
    profile.routes = dedupe_routes(clean_routes)

    profile.regions = list(
        dict.fromkeys(r.strip() for r in profile.regions if (r or "").strip())
    )[:8]

    profile.preferred_airports = list(
        dict.fromkeys(
            c.upper()
            for c in profile.preferred_airports
            if isinstance(c, str) and re.fullmatch(r"[A-Z]{3,4}", c.strip())
        )
    )[:12]
    return profile
