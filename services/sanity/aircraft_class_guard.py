"""
Aircraft class sanity guard — block invalid class substitutions in broker output.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence

# Models that must never appear as substitutes for executive / ULR missions.
_BANNED_SUBSTITUTES = frozenset(
    {
        "Citation CJ2",
        "Citation CJ4",
        "Learjet 75",
        "Pilatus PC-24",
        "Pilatus PC-12",
        "King Air",
        "Caravan",
    }
)

_BANNED_PATTERNS = re.compile(
    r"\b(?:citation\s+cj[24]|cj2|cj4|learjet\s*75|pc-?24|pc-?12|king\s+air|caravan)\b",
    re.I,
)

_ULR_CORRIDOR_RE = re.compile(
    r"\b(?:london|tokyo|dubai|singapore|hong\s+kong|hkg|sydney|johannesburg|"
    r"transatlantic|westbound|nonstop)\b",
    re.I,
)

_HEAVY_CABIN_RE = re.compile(
    r"\b(?:boardroom|heavy[- ]cabin|g650|global\s+7500|ultra[- ]long|ulr|"
    r"similar\s+cabin|executive\s+replacement)\b",
    re.I,
)

_WINTER_NBAA_RE = re.compile(
    r"\b(?:winter|nbaa|ifr\s+reserves?|westbound)\b",
    re.I,
)

_LIGHT_JET_PROFILES = frozenset(
    {
        "Citation CJ2",
        "Citation CJ4",
        "Learjet 75",
        "Pilatus PC-24",
        "Pilatus PC-12",
    }
)

_UTILITY_PROFILES = frozenset({"Pilatus PC-12"})


def _mission_signals(mission: Any, query: str = "") -> dict:
    ql = (query or "").lower()
    routes = list(getattr(mission, "routes", None) or [])
    route_blob = " ".join(routes).lower()
    blob = f"{ql} {route_blob}"
    pax = int(getattr(mission, "passenger_count", None) or 0)
    return {
        "ulr_corridor": bool(_ULR_CORRIDOR_RE.search(blob)),
        "heavy_cabin": bool(_HEAVY_CABIN_RE.search(blob)),
        "winter_nbaa": bool(_WINTER_NBAA_RE.search(blob)),
        "high_pax": pax >= 10,
        "long_stage": any(
            x in blob for x in ("tokyo", "london", "dubai", "singapore", "sydney", "hong kong")
        ),
    }


def violates_class_sanity(mission: Any, aircraft: str, *, query: str = "") -> bool:
    """
    True when an aircraft class is structurally inappropriate for the stated mission.
    """
    model = (aircraft or "").strip()
    if not model:
        return True
    if model in _BANNED_SUBSTITUTES:
        sig = _mission_signals(mission, query)
        if sig["ulr_corridor"] or sig["heavy_cabin"] or sig["high_pax"] or (
            sig["winter_nbaa"] and sig["long_stage"]
        ):
            return True
    if model in _LIGHT_JET_PROFILES:
        sig = _mission_signals(mission, query)
        if sig["ulr_corridor"] and (sig["winter_nbaa"] or sig["high_pax"]):
            return True
    if model in _UTILITY_PROFILES:
        sig = _mission_signals(mission, query)
        if sig["heavy_cabin"] or sig["ulr_corridor"]:
            return True
    return False


def filter_models_by_class_sanity(
    models: Sequence[str],
    mission: Any,
    *,
    query: str = "",
) -> List[str]:
    """Drop models that violate class sanity for this mission."""
    out: List[str] = []
    for m in models or []:
        name = (m or "").strip()
        if not name:
            continue
        if violates_class_sanity(mission, name, query=query):
            continue
        if _BANNED_PATTERNS.search(name) and _mission_signals(mission, query)["ulr_corridor"]:
            continue
        out.append(name)
    return out


__all__ = [
    "violates_class_sanity",
    "filter_models_by_class_sanity",
]
