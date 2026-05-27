"""
Route directionality enforcement — preserve textual flow; reject inverted continuations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from services.consultant.mission_state import MissionState, normalize_routes
from services.mission.hub_selection import is_me_continuation_hub
from services.mission.models import MissionProfile, Route
from services.mission.route_extractor import resolve_place

ROUTE_DIRECTIONALITY_KEY = "route_directionality"

_PACIFIC_ULR_CITIES = frozenset(
    {"tokyo", "singapore", "hong kong", "seoul", "beijing", "shanghai", "sydney"}
)
_MOUNTAIN_CITIES = frozenset(
    {
        "aspen",
        "telluride",
        "jackson hole",
        "jackson",
        "vail",
        "eagle",
        "sun valley",
    }
)
_EU_EXEC_CITIES = frozenset(
    {
        "paris",
        "geneva",
        "zurich",
        "frankfurt",
        "london",
        "madrid",
        "rome",
        "milan",
        "berlin",
        "munich",
    }
)

_OPERATIONAL_HUBS = frozenset(
    {
        "houston",
        "dallas",
        "new york",
        "los angeles",
        "miami",
        "perth",
        "calgary",
        "chicago",
        "teterboro",
    }
)
_US_EXECUTIVE_HUBS = frozenset(
    {
        "new york",
        "los angeles",
        "san francisco",
        "miami",
        "houston",
        "dallas",
        "chicago",
        "boston",
        "denver",
        "washington",
        "teterboro",
        "perth",
        "las vegas",
        "salt lake city",
    }
)

# Extractor leak patterns — cross-domain without itinerary authority
_EXTRACTOR_LEAK_PAIRS = (
    (r"\bsingapore\b", r"\baspen\b"),
    (r"\baspen\b", r"\btokyo\b"),
    (r"\baspen\b", r"\bsingapore\b"),
    (r"\bcaribbean\b", r"\b(?:dubai|riyadh|abu\s+dhabi)\b"),
    (r"\baspen\b", r"\b(?:dubai|riyadh)\b"),
)

_FIELD_REGION_NAMES = frozenset(
    {
        "remote drilling sites",
        "remote gravel strips",
        "northern alberta oil fields",
        "nunavut field ops",
        "arctic industrial access",
        "west africa",
        "permian basin",
        "desert energy corridor",
        "australian extraction strips",
    }
)

_CARIBBEAN_ANCHOR_HUBS = frozenset(
    {"miami", "palm beach", "fort lauderdale", "west palm"}
)
_NON_FLORIDA_CARIBBEAN_ORIGINS = _PACIFIC_ULR_CITIES | frozenset(
    {"dubai", "abu dhabi", "doha", "riyadh", "jeddah", "los angeles", "new york", "chicago"}
)


@dataclass
class DirectionalityReport:
    corrected: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    swapped: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corrected": list(self.corrected),
            "removed": list(self.removed),
            "swapped": list(self.swapped),
        }


def literal_direction_in_text(origin: str, destination: str, text: str) -> Optional[bool]:
    """
    Return True if origin->dest stated, False if dest->origin stated, None if ambiguous.
    """
    tl = (text or "").lower()
    o, d = origin.lower(), destination.lower()
    forward = bool(
        re.search(rf"\b{re.escape(o)}\s*(?:->|→|—|–|to|-)\s*{re.escape(d)}\b", tl)
        or re.search(rf"\bfrom\s+{re.escape(o)}\s+to\s+{re.escape(d)}\b", tl)
        or re.search(rf"\b{re.escape(o)}\s*[-–]\s*{re.escape(d)}\b", tl)
    )
    backward = bool(
        re.search(rf"\b{re.escape(d)}\s*(?:->|→|—|–|to|-)\s*{re.escape(o)}\b", tl)
        or re.search(rf"\bfrom\s+{re.escape(d)}\s+to\s+{re.escape(o)}\b", tl)
    )
    if forward and not backward:
        return True
    if backward and not forward:
        return False
    return None


def _is_us_executive_city(name: str) -> bool:
    place, conf = resolve_place(name)
    if place and conf >= 0.72:
        if place.country == "US":
            return True
        if place.canonical.lower() in _US_EXECUTIVE_HUBS:
            return True
    return (name or "").lower() in _US_EXECUTIVE_HUBS


def _is_extractor_leak(origin: str, dest: str, text: str) -> bool:
    if literal_direction_in_text(origin, dest, text) is True:
        return False
    o, d = origin.lower(), dest.lower()
    blob = f"{text or ''} {o} {d}".lower()
    for o_pat, d_pat in _EXTRACTOR_LEAK_PAIRS:
        if re.search(o_pat, blob) and re.search(d_pat, blob):
            if (re.search(o_pat, o) or o in blob) and (re.search(d_pat, d) or d in blob):
                if any(m in o for m in _MOUNTAIN_CITIES) or any(
                    m in d for m in _MOUNTAIN_CITIES
                ):
                    if any(p in o for p in _PACIFIC_ULR_CITIES) or any(
                        p in d for p in _PACIFIC_ULR_CITIES
                    ):
                        return True
    # Mountain <-> Pacific without literal edge
    o_mountain = any(m in o for m in _MOUNTAIN_CITIES)
    d_mountain = any(m in d for m in _MOUNTAIN_CITIES)
    o_pacific = any(p in o for p in _PACIFIC_ULR_CITIES) or o in (
        "tokyo",
        "singapore",
        "hong kong",
        "seoul",
    )
    d_pacific = any(p in d for p in _PACIFIC_ULR_CITIES) or d in (
        "tokyo",
        "singapore",
        "hong kong",
        "seoul",
    )
    if (o_mountain and d_pacific) or (o_pacific and d_mountain):
        return True
    # Pacific / ME / non-Florida exec → Caribbean without literal edge
    if d == "caribbean" and o not in _CARIBBEAN_ANCHOR_HUBS:
        if o in _NON_FLORIDA_CARIBBEAN_ORIGINS or is_me_continuation_hub(origin):
            return True
    # ME continuation hub -> US executive city without literal edge (remove, never swap)
    if is_me_continuation_hub(origin) and _is_us_executive_city(dest):
        if literal_direction_in_text(origin, dest, text) is not True:
            return True
    # EU executive -> operational hub inversion without literal edge (remove, never swap)
    if o in _EU_EXEC_CITIES and d in _OPERATIONAL_HUBS:
        if literal_direction_in_text(origin, dest, text) is not True:
            return True
    # Florida hub -> Pacific/ULR cities without literal edge (avoid Miami absorbing LA/Tokyo)
    if o in _CARIBBEAN_ANCHOR_HUBS and d in _PACIFIC_ULR_CITIES:
        if literal_direction_in_text(origin, dest, text) is not True:
            return True
    # Florida corridor hub → transatlantic EU without literal edge
    if o in _CARIBBEAN_ANCHOR_HUBS and d in _EU_EXEC_CITIES:
        if literal_direction_in_text(origin, dest, text) is not True:
            return True
    # Florida hub → industrial/field regions (wrong anchor for arctic/industrial spokes)
    if o in _CARIBBEAN_ANCHOR_HUBS and d in _FIELD_REGION_NAMES:
        if literal_direction_in_text(origin, dest, text) is not True:
            return True
    return False


def should_swap_continuation(origin: str, destination: str, text: str) -> bool:
    """ME hub as origin to US executive city without textual reverse authority."""
    # Geographic stabilization rule: never swap/flip routes here.
    return False


def should_swap_eu_to_operational_hub(origin: str, destination: str, text: str) -> bool:
    """EU city → operational hub should be hub → EU unless explicitly stated backward."""
    # Geographic stabilization rule: never swap/flip routes here.
    return False


def validate_route_direction(
    route: Route,
    text: str,
    *,
    locked: bool = False,
) -> Tuple[Optional[Route], str]:
    """
    Return (corrected_route_or_none, action) where action is keep|swap|remove.
    """
    if locked:
        return route, "keep"

    o, d = route.origin, route.destination
    if _is_extractor_leak(o, d, text):
        return None, "remove"

    stated = literal_direction_in_text(o, d, text)
    if stated is False:
        return None, "remove"

    return route, "keep"


def enforce_route_directionality(
    query: str,
    profile: MissionProfile,
    mission: MissionState,
    *,
    explicit_lock=None,
) -> DirectionalityReport:
    """Correct inverted continuations; strip extractor ghost edges."""
    from services.mission.explicit_route_lock import extract_explicit_routes

    lock = explicit_lock or extract_explicit_routes(query)
    report = DirectionalityReport()
    kept: List[Route] = []
    existing: Set[Tuple[str, str]] = set()

    for route in list(profile.routes):
        corrected, action = validate_route_direction(
            route, query, locked=lock.is_locked_route(route)
        )
        if action == "remove":
            report.removed.append(route.label())
            continue
        if action == "swap" and corrected:
            report.swapped.append(f"{route.label()} -> {corrected.label()}")
            route = corrected
        if (route.origin.lower(), route.destination.lower()) in existing:
            continue
        existing.add((route.origin.lower(), route.destination.lower()))
        kept.append(route)

    profile.routes = kept
    mission.routes = normalize_routes([r.label() for r in kept])
    report.corrected = [r.label() for r in kept]
    return report


def apply_route_directionality(
    query: str,
    profile: MissionProfile,
    mission: MissionState,
    *,
    data_used: Optional[Dict[str, Any]] = None,
) -> DirectionalityReport:
    from services.mission.explicit_route_lock import extract_explicit_routes

    du = data_used if isinstance(data_used, dict) else {}
    lock = extract_explicit_routes(query)
    report = enforce_route_directionality(query, profile, mission, explicit_lock=lock)
    if isinstance(data_used, dict):
        data_used[ROUTE_DIRECTIONALITY_KEY] = report.to_dict()
    return report


__all__ = [
    "ROUTE_DIRECTIONALITY_KEY",
    "DirectionalityReport",
    "apply_route_directionality",
    "enforce_route_directionality",
    "literal_direction_in_text",
    "validate_route_direction",
]
