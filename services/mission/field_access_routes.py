"""
Field-access spokes — hub cities to operational field regions (drilling, mining, arctic).

These are first-class route-graph nodes, not narrative mentions.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from services.mission.hub_selection import select_local_hub
from services.mission.models import MissionProfile, Route
from services.mission.route_extractor import _build_route_extraction

_FIELD_REGION_SPECS: Tuple[Tuple[str, str], ...] = (
    (r"\bremote\s+(?:drilling\s+)?sites?\b", "Remote Drilling Sites"),
    (r"\barctic\s+oil\s+platforms?\b", "Arctic Oil Platforms"),
    (r"\b(?:unpaved|mining)\s+strips?\s+in\s+west\s+africa\b", "West Africa"),
    (r"\bwest\s+africa\b.*\b(?:mining|unpaved)\b", "West Africa"),
    (r"\boil\s+(?:fields?|platforms?)\b", "Remote Drilling Sites"),
    (r"\bremote\s+oil\b", "Remote Drilling Sites"),
)

# Northern Canada gravel / arctic logistics — arctic_industrial_layer owns these spokes
_ARCTIC_FIELD_DEFER_RE = re.compile(
    r"\b(?:northern\s+canada|yellowknife|nunavut|northern\s+alberta|"
    r"gravel\s+strips?\s+in\s+northern\s+canada)\b",
    re.I,
)

_HUB_PRIORITY = (
    "Houston",
    "Calgary",
    "Anchorage",
    "Denver",
    "Dallas",
    "New York",
)


def _field_regions_in_text(text: str) -> List[str]:
    regions: List[str] = []
    tl = text or ""
    for pat, canonical in _FIELD_REGION_SPECS:
        if re.search(pat, tl, re.I) and canonical not in regions:
            regions.append(canonical)
    return regions


def _pick_field_hub(profile: MissionProfile, text: str) -> str:
    return select_local_hub(
        profile,
        text,
        _HUB_PRIORITY,
        mission_type="industrial",
        default="Houston",
    )


def infer_field_access_spokes(text: str, profile: MissionProfile) -> List[Route]:
    """Hub → field region legs when operational sites are named without city pairs."""
    tl = text or ""
    if _ARCTIC_FIELD_DEFER_RE.search(tl):
        return []
    regions = _field_regions_in_text(tl)
    if not regions:
        return []
    hub = _pick_field_hub(profile, text)
    existing = {(r.origin.lower(), r.destination.lower()) for r in profile.routes}
    spokes: List[Route] = []
    for region in regions:
        ext = _build_route_extraction(hub, region, pattern_boost=0.1)
        if not ext:
            continue
        key = (ext.route.origin.lower(), ext.route.destination.lower())
        if key not in existing:
            spokes.append(ext.route)
            existing.add(key)
    return spokes
