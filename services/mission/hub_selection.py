"""
Hub selection — mission-origin local hub first; foreign hubs only when explicit.

Used by geographic route intelligence and field-access spokes.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple, Union

from services.mission.models import MissionProfile, Route
from services.mission.route_extractor import resolve_place

# Continuation / ULR destinations — never used as domestic-utilization origins
_ME_CONTINUATION_HUBS = frozenset(
    {
        "dubai",
        "abu dhabi",
        "doha",
        "riyadh",
        "jeddah",
    }
)

_EUROPEAN_HUBS = frozenset(
    {
        "london",
        "paris",
        "geneva",
        "zurich",
        "frankfurt",
        "madrid",
        "rome",
        "milan",
        "barcelona",
        "berlin",
        "munich",
    }
)

_REGIONAL_BIAS_PATTERNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (r"\btexas\b.*\b(?:energy|desert|drilling|oil)\b", ("Houston", "Dallas")),
    (r"\b(?:texas|houston|dallas)\b", ("Houston", "Dallas")),
    (r"\b(?:california|nevada|los\s+angeles|san\s+francisco)\b", ("Los Angeles", "San Francisco", "Las Vegas")),
    (r"\b(?:northeast|east\s+coast|new\s+york|boston)\b", ("New York", "Boston", "Washington", "Teterboro")),
    (r"\b(?:south\s+florida|miami|palm\s+beach)\b", ("Miami", "Palm Beach")),
    (r"\b(?:perth|australia|australian)\b", ("Perth", "Singapore")),
    (r"\b(?:houston|lagos|west\s+africa)\b", ("Houston",)),
    (r"\b(?:middle\s+east|uae|saudi)\b", ("Dubai", "Abu Dhabi", "Riyadh")),
    (
        r"\b(?:yellowknife|nunavut|northern\s+canada|northern\s+alberta|calgary\s+oil)\b",
        ("Yellowknife", "Calgary"),
    ),
)


def hub_priority_score(
    candidate: str,
    text: str,
    *,
    profile: Optional[MissionProfile] = None,
    mission_type: str = "executive",
    regional_bias: Optional[Sequence[str]] = None,
) -> float:
    """
    Higher score = better hub choice. Local mission-origin bias dominates.
    """
    tl = (text or "").lower()
    c_lower = candidate.lower()
    score = 0.5

    place, conf = resolve_place(candidate)
    if not place or conf < 0.72:
        return 0.2

    # Penalize ME hubs as general-purpose origins
    if c_lower in _ME_CONTINUATION_HUBS and mission_type in (
        "executive",
        "domestic_utilization",
        "industrial",
    ):
        score -= 0.45

    # Boost explicit mention in query
    if re.search(rf"\b{re.escape(c_lower)}\b", tl):
        score += 0.25

    # Home / base explicit
    if re.search(
        rf"\b(?:based\s+in|headquartered\s+in|from\s+){re.escape(c_lower)}\b", tl
    ):
        score += 0.35

    # Regional bias from mission text
    if regional_bias:
        for rb in regional_bias:
            if rb.lower() == c_lower:
                score += 0.4

    for pat, hubs in _REGIONAL_BIAS_PATTERNS:
        if re.search(pat, tl, re.I) and c_lower in {h.lower() for h in hubs}:
            score += 0.35

    # US locality for US-mission language
    if place.country == "US" and re.search(
        r"\b(?:domestic|u\.?s\.?|united\s+states|corridor|texas|california|nevada|florida|northeast)\b",
        tl,
        re.I,
    ):
        score += 0.2

    # Industrial missions: penalize EU cities unless EU industrial context only
    if mission_type == "industrial":
        if c_lower in _EUROPEAN_HUBS and not re.search(
            r"\b(?:london|paris|geneva|zurich|frankfurt)\b.*\b(?:desert|drilling|mining|field)\b",
            tl,
            re.I,
        ):
            if re.search(r"\btexas\b|\bhouston\b|\bdallas\b", tl, re.I):
                score -= 0.5

    # Already used as origin on existing routes (mission anchor evidence)
    routes_list = _routes_from_profile(profile)
    if routes_list:
        for r in routes_list:
            if r.origin.lower() == c_lower:
                score += 0.15
            if r.destination.lower() == c_lower and mission_type != "continuation_dest":
                score += 0.05

    return min(1.0, max(0.0, score))


def _routes_from_profile(
    profile: Optional[Union[MissionProfile, List[Route]]],
) -> List[Route]:
    if profile is None:
        return []
    if isinstance(profile, list):
        return profile
    return list(profile.routes)


def select_local_hub(
    profile: Union[MissionProfile, List[Route], None],
    text: str,
    candidates: Sequence[str],
    *,
    mission_type: str = "executive",
    regional_bias: Optional[Sequence[str]] = None,
    default: str = "New York",
) -> str:
    """Pick highest-scoring hub from candidates — local mission origin first."""
    if not candidates:
        return default

    inferred_bias: List[str] = list(regional_bias or [])
    for pat, hubs in _REGIONAL_BIAS_PATTERNS:
        if re.search(pat, text or "", re.I):
            for h in hubs:
                if h not in inferred_bias:
                    inferred_bias.append(h)

    best = default
    best_score = -1.0
    for hub in candidates:
        s = hub_priority_score(
            hub,
            text,
            profile=profile,
            mission_type=mission_type,
            regional_bias=inferred_bias,
        )
        if s > best_score:
            best_score = s
            best = hub
    return best


def is_me_continuation_hub(name: str) -> bool:
    return (name or "").lower() in _ME_CONTINUATION_HUBS


def is_european_hub(name: str) -> bool:
    return (name or "").lower() in _EUROPEAN_HUBS


__all__ = [
    "hub_priority_score",
    "is_european_hub",
    "is_me_continuation_hub",
    "select_local_hub",
]
