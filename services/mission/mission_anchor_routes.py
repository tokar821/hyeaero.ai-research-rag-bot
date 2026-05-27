"""
Mission anchor routes — bind operational domains to validated legs when extraction is thin.

Used when mountain, ULR, or ski domains are stated without explicit city pairs.
"""

from __future__ import annotations

import re
from typing import List, Set, Tuple

from services.mission.models import MissionProfile, Route
from services.mission.route_extractor import _build_route_extraction

# (trigger pattern, list of (origin, destination) anchors)
_ANCHOR_RULES: Tuple[Tuple[str, Tuple[Tuple[str, str], ...]], ...] = (
    (
        r"\b(?:ski\s+access|aspen|jackson\s+hole)\b.*\b(?:asia|tokyo|singapore)\b"
        r"|\b(?:asia|tokyo|singapore)\b.*\b(?:ski|aspen|jackson)\b",
        (
            ("Los Angeles", "Tokyo"),
            ("Los Angeles", "Singapore"),
            ("Los Angeles", "Aspen"),
            ("Los Angeles", "Jackson Hole"),
        ),
    ),
    (
        r"\b(?:ski\s+access|aspen|jackson)\b.*\b(?:europe|winter)\b"
        r"|\beuropean\s+winter\b.*\b(?:aspen|jackson)\b",
        (
            ("Los Angeles", "Aspen"),
            ("Los Angeles", "Jackson Hole"),
            ("New York", "London"),
        ),
    ),
    (
        r"\bdispatch\s+reliability\b.*\b(?:aspen|jackson|ski)\b"
        r"|\b(?:aspen|jackson\s+hole)\b.*\b(?:dispatch|failures?)\b",
        (
            ("Los Angeles", "Aspen"),
            ("Los Angeles", "Jackson Hole"),
            ("Los Angeles", "Tokyo"),
        ),
    ),
    (
        r"\b(?:need|want|require).*\b(?:asia|tokyo|singapore)\b.*\b(?:ski|aspen)\b"
        r"|\b(?:ski|aspen).*\b(?:asia|tokyo|singapore)\s+capability\b",
        (
            ("Los Angeles", "Tokyo"),
            ("Los Angeles", "Aspen"),
            ("Los Angeles", "Jackson Hole"),
        ),
    ),
)


def infer_mission_anchor_routes(text: str, profile: MissionProfile) -> List[Route]:
    """Add missing anchor legs when operational domain mix is stated in text."""
    tl = text or ""
    existing: Set[Tuple[str, str]] = {
        (r.origin.lower(), r.destination.lower()) for r in profile.routes
    }
    found: List[Route] = []
    for pat, pairs in _ANCHOR_RULES:
        if not re.search(pat, tl, re.I):
            continue
        for o, d in pairs:
            ext = _build_route_extraction(o, d, pattern_boost=0.09)
            if not ext:
                continue
            key = (ext.route.origin.lower(), ext.route.destination.lower())
            if key in existing:
                continue
            existing.add(key)
            found.append(ext.route)
    return found
