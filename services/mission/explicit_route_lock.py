"""
Explicit route lock — HARD guarantee for user-stated city pairs.

Explicit routes override hub inference, re-anchoring, directionality swaps, and topology removal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from services.mission.models import Route
from services.mission.route_extractor import (
    RouteExtraction,
    _build_route_extraction,
    _extract_arrow_segments,
    _extract_known_phrases,
    _extract_to_segments,
    dedupe_extractions,
    resolve_place,
    sanitize_user_text_for_routes,
)

EXPLICIT_ROUTE_LOCK_KEY = "explicit_route_lock"

_CITY_TOKEN = (
    r"(?:los\s+angeles|san\s+francisco|new\s+york|nyc|\bla\b|\bsf\b|miami|houston|dallas|"
    r"chicago|boston|london|paris|geneva|zurich|frankfurt|tokyo|singapore|dubai|"
    r"abu\s+dhabi|riyadh|doha|calgary|yellowknife|perth|lagos|aspen|denver|madrid|seoul|"
    r"hong\s+kong|sao\s+paulo)"
)
_EXPLICIT_EDGE_RE = re.compile(
    rf"\b(?P<o>{_CITY_TOKEN})\s*(?:->|→|—|–|to|-)\s*(?P<d>{_CITY_TOKEN})\b",
    re.I,
)
_FROM_TO_RE = re.compile(
    rf"\bfrom\s+(?P<o>{_CITY_TOKEN})\s+to\s+(?P<d>{_CITY_TOKEN})\b",
    re.I,
)

_ALIAS_NORMALIZE = {
    "nyc": "New York",
    "la": "Los Angeles",
    "sf": "San Francisco",
}


_FIELD_REGION_ORIGINS = frozenset(
    {
        "remote drilling sites",
        "desert energy corridor",
        "arctic industrial access",
        "arctic oil platforms",
        "west africa",
        "remote gravel strips",
        "northern alberta oil fields",
        "nunavut field ops",
        "australian extraction strips",
        "permian basin",
        "nigerian energy corridor",
        "northern africa",
        "offshore rigs",
        "pilbara",
        "regional",
    }
)


def _is_valid_explicit_route(route: Route) -> bool:
    o_l = route.origin.lower()
    d_l = route.destination.lower()
    if o_l in _FIELD_REGION_ORIGINS:
        return False
    if d_l in _FIELD_REGION_ORIGINS:
        return False
    op, oc = resolve_place(route.origin)
    dp, dc = resolve_place(route.destination)
    if not op or oc < 0.72 or not dp or dc < 0.72:
        return False
    if op.kind == "region" and dp.kind == "region":
        return False
    return True


def _normalize_city(raw: str) -> Optional[str]:
    text = (raw or "").strip()
    key = text.lower()
    if key in _ALIAS_NORMALIZE:
        text = _ALIAS_NORMALIZE[key]
    place, conf = resolve_place(text)
    if place and conf >= 0.72:
        return place.canonical
    return None


@dataclass
class ExplicitRouteLock:
    routes: List[Route] = field(default_factory=list)
    locked_pairs: FrozenSet[Tuple[str, str]] = field(default_factory=frozenset)
    locked_labels: List[str] = field(default_factory=list)

    def is_locked(self, origin: str, destination: str) -> bool:
        return (origin.lower(), destination.lower()) in self.locked_pairs

    def is_locked_route(self, route: Route) -> bool:
        return self.is_locked(route.origin, route.destination)

    def to_dict(self) -> Dict:
        return {
            "locked_labels": list(self.locked_labels),
            "locked_count": len(self.locked_pairs),
        }


def extract_explicit_routes(query: str) -> ExplicitRouteLock:
    """Collect all explicitly stated city→city pairs from user text."""
    text = sanitize_user_text_for_routes(query)
    if not text:
        return ExplicitRouteLock()

    extractions: List[RouteExtraction] = []
    for fn in (_extract_known_phrases, _extract_arrow_segments, _extract_to_segments):
        extractions.extend(fn(text))

    for pat in (_EXPLICIT_EDGE_RE, _FROM_TO_RE):
        for m in pat.finditer(text):
            o = _normalize_city(m.group("o"))
            d = _normalize_city(m.group("d"))
            if not o or not d or o.lower() == d.lower():
                continue
            ext = _build_route_extraction(o, d, pattern_boost=0.12)
            if ext:
                extractions.append(
                    RouteExtraction(route=ext.route, confidence=max(ext.confidence, 0.92))
                )

    deduped = dedupe_extractions(extractions)
    routes = [e.route for e in deduped if e.confidence >= 0.72 and _is_valid_explicit_route(e.route)]
    pairs = frozenset((r.origin.lower(), r.destination.lower()) for r in routes)
    return ExplicitRouteLock(
        routes=routes,
        locked_pairs=pairs,
        locked_labels=[r.label() for r in routes],
    )


def merge_explicit_routes_into_profile(lock: ExplicitRouteLock, profile) -> List[str]:
    """Ensure locked explicit routes exist in profile."""
    added: List[str] = []
    existing = {(r.origin.lower(), r.destination.lower()) for r in profile.routes}
    for route in lock.routes:
        key = (route.origin.lower(), route.destination.lower())
        if key not in existing:
            profile.routes.append(route)
            existing.add(key)
            added.append(route.label())
    return added


def strip_conflicting_inferred_routes(lock: ExplicitRouteLock, profile) -> List[str]:
    """Drop inferred re-anchors that duplicate explicit route destinations under wrong origin."""
    removed: List[str] = []
    if not lock.locked_pairs:
        return removed

    explicit_dests: Dict[str, Set[str]] = {}
    for o, d in lock.locked_pairs:
        if o in _FIELD_REGION_ORIGINS:
            continue
        explicit_dests.setdefault(o, set()).add(d)

    kept: List[Route] = []
    for r in profile.routes:
        if lock.is_locked_route(r):
            kept.append(r)
            continue
        o_l, d_l = r.origin.lower(), r.destination.lower()
        conflict = False
        for ex_o, dests in explicit_dests.items():
            if d_l in dests and o_l != ex_o:
                conflict = True
                break
        if conflict:
            removed.append(r.label())
        else:
            kept.append(r)
    profile.routes = kept
    return removed


__all__ = [
    "EXPLICIT_ROUTE_LOCK_KEY",
    "ExplicitRouteLock",
    "extract_explicit_routes",
    "merge_explicit_routes_into_profile",
    "strip_conflicting_inferred_routes",
]
