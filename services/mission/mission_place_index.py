"""
Canonical place index — accent-insensitive capture for route graph and constraints.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Set

from services.mission.aviation_places import ALIAS_TO_PLACE, AviationPlace
from services.mission.models import MissionProfile
from services.mission.route_extractor import resolve_place


def normalize_place_key(text: str) -> str:
    """Fold accents and punctuation for place matching."""
    s = unicodedata.normalize("NFKD", (text or "").strip().lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip()


def place_keys_for_label(label: str) -> Set[str]:
    keys: Set[str] = set()
    for part in re.split(r"\s*->\s*", label or ""):
        place, conf = resolve_place(part.strip())
        if place and conf >= 0.72:
            keys.add(normalize_place_key(place.canonical))
            keys.add(normalize_place_key(part))
    return keys


def places_captured_from_mission(
    profile: MissionProfile,
    text: str = "",
) -> List[str]:
    """All catalog city/region canonicals present in routes or user text."""
    found: Set[str] = set()
    for r in profile.routes:
        for part in (r.origin, r.destination):
            place, conf = resolve_place(part)
            if place and conf >= 0.72:
                found.add(place.canonical)

    tl = normalize_place_key(text)
    for alias, place in ALIAS_TO_PLACE.items():
        if len(alias) < 3:
            continue
        if re.search(rf"\b{re.escape(normalize_place_key(alias))}\b", tl):
            found.add(place.canonical)
    return sorted(found)


def city_captured(city_needle: str, profile: MissionProfile, text: str = "") -> bool:
    """True if needle matches any captured place (accent-insensitive)."""
    needle = normalize_place_key(city_needle)
    keys = place_keys_for_label(" -> ".join(profile.route_labels()))
    for k in keys:
        if needle in k or k in needle:
            return True
    for canon in places_captured_from_mission(profile, text):
        if needle in normalize_place_key(canon):
            return True
    return False
