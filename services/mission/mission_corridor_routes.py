"""
Multi-hub corridor route construction from catalog-resolved geography.

Builds validated legs for "operate between A, B, C, and …" missions without
collapsing to a single hub→region stub when multiple international cities appear.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Set, Tuple

from services.mission.aviation_places import ALIAS_TO_PLACE, AviationPlace
from services.mission.models import MissionProfile, Route
from services.mission.route_extractor import (
    MIN_CONFIDENCE,
    RouteExtraction,
    _build_route_extraction,
    dedupe_extractions,
    resolve_place,
)

_BETWEEN_CLAUSE_RE = re.compile(
    r"\b(?:operate|operating|fly|flying|travel|traveling|move|moving)\s+between\s+(.+?)(?:\.|$|\n)",
    re.I,
)
_MULTI_DOMAIN_PORTFOLIO_RE = re.compile(r"\boperate\s+across\b", re.I)

_PRIMARY_HUBS = (
    "Miami",
    "Houston",
    "New York",
    "Dallas",
    "Los Angeles",
    "Chicago",
    "Teterboro",
)

_FIELD_ACCESS_RE = re.compile(
    r"\b(?:unpaved|northern\s+canada|oil\s+sites?|remote\s+(?:oil|field)|"
    r"short\s+and\s+unpaved|unpaved\s+runways?)\b",
    re.I,
)


def detect_field_access_posture(text: str) -> bool:
    """Operational field-access evidence (runway over cabin), not response templating."""
    return bool(_FIELD_ACCESS_RE.search(text or ""))


def enumerate_ordered_places(text: str) -> List[AviationPlace]:
    """Catalog-ordered place mentions in user text (longest alias wins at each offset)."""
    tl = (text or "").lower()
    if not tl:
        return []
    spans: List[Tuple[int, int, AviationPlace]] = []
    seen_alias: Set[str] = set()
    for alias, place in sorted(ALIAS_TO_PLACE.items(), key=lambda x: -len(x[0])):
        if len(alias) < 3 or alias in seen_alias:
            continue
        if place.kind == "region" and alias in ("us", "eu", "la"):
            continue
        seen_alias.add(alias)
        for m in re.finditer(rf"\b{re.escape(alias)}\b", tl):
            spans.append((m.start(), m.end(), place))

    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    ordered: List[AviationPlace] = []
    seen_canon: Set[str] = set()
    occupied: List[Tuple[int, int]] = []
    for start, end, place in spans:
        if any(start < oe and end > os for os, oe in occupied):
            continue
        key = place.canonical.lower()
        if key in seen_canon:
            continue
        seen_canon.add(key)
        occupied.append((start, end))
        ordered.append(place)
    return ordered


def count_resolved_city_hubs_in_text(text: str) -> int:
    return sum(1 for p in enumerate_ordered_places(text) if p.kind == "city")


def _is_multi_domain_portfolio(text: str) -> bool:
    """Portfolio-style geography — defer to geographic intelligence, not star-hub corridor."""
    if not _MULTI_DOMAIN_PORTFOLIO_RE.search(text or ""):
        return False
    return len(enumerate_ordered_places(text)) >= 4


def _pick_primary_hub(
    cities: Sequence[AviationPlace],
    *,
    text: str = "",
    profile: Optional[MissionProfile] = None,
) -> Optional[AviationPlace]:
    if not cities:
        return None
    from services.mission.hub_selection import select_local_hub

    canon = {c.canonical for c in cities}
    candidates = [h for h in _PRIMARY_HUBS if h in canon]
    if not candidates:
        return cities[0]
    chosen = select_local_hub(
        profile,
        text,
        candidates,
        mission_type="executive",
        default=candidates[0],
    )
    return next((c for c in cities if c.canonical == chosen), cities[0])


def build_corridor_extractions_from_places(
    places: Sequence[AviationPlace],
    *,
    text: str = "",
    profile: Optional[MissionProfile] = None,
    pattern_boost: float = 0.09,
) -> List[RouteExtraction]:
    """Star hub + longest intercontinental pair for multi-city portfolios."""
    if len(places) < 2:
        return []

    cities = [p for p in places if p.kind == "city"]
    regions = [p for p in places if p.kind == "region"]
    found: List[RouteExtraction] = []

    hub = _pick_primary_hub(cities, text=text, profile=profile) if cities else None
    if hub:
        for dest in cities:
            if dest.canonical == hub.canonical:
                continue
            ext = _build_route_extraction(
                hub.canonical, dest.canonical, pattern_boost=pattern_boost
            )
            if ext:
                found.append(ext)
        for reg in regions:
            ext = _build_route_extraction(
                hub.canonical, reg.canonical, pattern_boost=pattern_boost - 0.01
            )
            if ext:
                found.append(ext)

    if len(cities) >= 3:
        try:
            from services.consultant.route_feasibility import estimate_route_distance_nm

            best_ext: Optional[RouteExtraction] = None
            best_dist = 0.0
            for i in range(len(cities)):
                for j in range(i + 1, len(cities)):
                    dist = estimate_route_distance_nm(
                        f"{cities[i].canonical} -> {cities[j].canonical}"
                    )
                    ext = _build_route_extraction(
                        cities[i].canonical,
                        cities[j].canonical,
                        pattern_boost=pattern_boost + 0.02,
                    )
                    if ext and dist >= best_dist:
                        best_dist = dist
                        best_ext = ext
            if best_ext:
                found.append(best_ext)
        except Exception:
            pass

    return dedupe_extractions(found)


def extract_between_corridor(text: str) -> List[RouteExtraction]:
    """Parse ``operate between X, Y, Z and …`` into catalog-validated legs."""
    m = _BETWEEN_CLAUSE_RE.search(text or "")
    if not m:
        return []
    clause = m.group(1).strip()
    clause = re.sub(
        r"\b(?:some|small|short)\s+(?:caribbean\s+)?islands?\b",
        "Caribbean",
        clause,
        flags=re.I,
    )
    parts = re.split(r",|\band\b", clause, flags=re.I)
    places: List[AviationPlace] = []
    for part in parts:
        fragment = part.strip()
        if not fragment:
            continue
        place, conf = resolve_place(fragment)
        if not place or conf < MIN_CONFIDENCE:
            trimmed = fragment
            for n in range(len(fragment.split()), 0, -1):
                gram = " ".join(fragment.split()[:n])
                place, conf = resolve_place(gram)
                if place and conf >= MIN_CONFIDENCE:
                    break
        if place and conf >= MIN_CONFIDENCE:
            if not any(p.canonical == place.canonical for p in places):
                places.append(place)
    if len(places) < 2:
        return []
    return build_corridor_extractions_from_places(places, text=text)


def _is_hub_region_collapse(profile: MissionProfile, text: str) -> bool:
    labels = profile.route_labels()
    if len(labels) != 1:
        return False
    if "caribbean" not in labels[0].lower():
        return False
    return count_resolved_city_hubs_in_text(text) >= 2


def enrich_profile_routes_from_corridor(text: str, profile: MissionProfile) -> bool:
    """
    Merge corridor legs into profile when extraction under-represents multi-city intent.

    Returns True when routes were added or replaced.
    """
    if not (text or "").strip():
        return False

    if _is_multi_domain_portfolio(text):
        return False

    candidates: List[RouteExtraction] = []
    candidates.extend(extract_between_corridor(text))
    places = enumerate_ordered_places(text)
    if len(places) >= 2 and not candidates:
        candidates.extend(build_corridor_extractions_from_places(places, text=text, profile=profile))

    if not candidates:
        return False

    existing = list(profile.routes)
    if existing and not _is_hub_region_collapse(profile, text):
        if len(existing) >= len(candidates):
            return False

    new_routes = [e.route for e in candidates]
    if not new_routes:
        return False

    profile.routes = new_routes
    profile.international_ops = True
    for r in new_routes:
        for name in (r.origin, r.destination):
            from services.mission.normalization import infer_regions_from_places

            for reg in infer_regions_from_places(name):
                if reg not in profile.regions:
                    profile.regions.append(reg)
    return True


__all__ = [
    "build_corridor_extractions_from_places",
    "count_resolved_city_hubs_in_text",
    "detect_field_access_posture",
    "enrich_profile_routes_from_corridor",
    "enumerate_ordered_places",
    "extract_between_corridor",
]
