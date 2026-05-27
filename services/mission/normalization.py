"""
Deterministic normalization for mission extraction inputs.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from services.mission.models import MissionCategory, OwnershipMode, PriorityLevel

# City / region canonical names
_PLACE_ALIASES: dict[str, str] = {
    "nyc": "New York",
    "new york city": "New York",
    "la": "Los Angeles",
    "los angeles": "Los Angeles",
    "l.a.": "Los Angeles",
    "sf": "San Francisco",
    "sfo": "San Francisco",
    "san fran": "San Francisco",
    "mia": "Miami",
    "bos": "Boston",
    "ord": "Chicago",
    "chi": "Chicago",
    "dfw": "Dallas",
    "dallas fort worth": "Dallas",
    "pbi": "Palm Beach",
    "teb": "Teterboro",
    "lon": "London",
    "lhr": "London",
    "par": "Paris",
    "cdg": "Paris",
    "tok": "Tokyo",
    "hnd": "Tokyo",
    "nrt": "Tokyo",
    "dxb": "Dubai",
    "gva": "Geneva",
    "west coast": "West Coast",
    "east coast": "East Coast",
    "europe": "Europe",
    "caribbean": "Caribbean",
    "transcon": "US Transcontinental",
}

_REGION_FROM_PLACE: dict[str, str] = {
    "miami": "South Florida / Caribbean gateway",
    "caribbean": "Caribbean",
    "europe": "Europe",
    "london": "Europe",
    "paris": "Europe",
    "tokyo": "Asia-Pacific",
    "seoul": "Asia-Pacific",
    "west coast": "US West Coast",
    "los angeles": "US West Coast",
    "san francisco": "US West Coast",
    "new york": "US Northeast",
    "aspen": "US Mountain",
    "dallas": "US South Central",
}


def normalize_place(name: str) -> str:
    """Canonical city/region label for route endpoints."""
    raw = re.sub(r"\s+", " ", (name or "").strip())
    if not raw:
        return ""
    key = raw.lower()
    if key in _PLACE_ALIASES:
        return _PLACE_ALIASES[key]
    # Title-case multi-word unless known acronym
    if len(raw) <= 4 and raw.isupper():
        return raw.upper()
    return " ".join(w.capitalize() for w in raw.split())


def infer_regions_from_places(*places: str) -> list[str]:
    regions: list[str] = []
    for p in places:
        key = p.strip().lower()
        if key in _REGION_FROM_PLACE:
            label = _REGION_FROM_PLACE[key]
            if label not in regions:
                regions.append(label)
    return regions


def normalize_passenger_count(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= n <= 19:
        return n
    return None


def normalize_ownership(text: str) -> Optional[OwnershipMode]:
    tl = (text or "").lower()
    if re.search(r"\bfractional|netjets|wheels\s+up|flexjet\b", tl):
        return OwnershipMode.FRACTIONAL
    if re.search(r"\bcharter\s+only|charter\s+vs\b", tl):
        return OwnershipMode.CHARTER
    if re.search(r"\b(?:full\s+)?own(?:ership)?|buy\s+outright|acquire|purchase\b", tl):
        return OwnershipMode.FULL_OWNERSHIP
    return None


def normalize_mission_category(text: str) -> Optional[MissionCategory]:
    tl = (text or "").lower()
    if re.search(r"\bcompare|versus|vs\.?\b", tl):
        return MissionCategory.COMPARISON
    if re.search(r"\brecommend|best\s+(?:jet|aircraft|option)|what\s+(?:jet|aircraft)\b", tl):
        return MissionCategory.ACQUISITION_ADVISORY
    if re.search(r"\bfractional|charter\s+vs\s+own\b", tl):
        return MissionCategory.OWNERSHIP_STRUCTURE
    if re.search(r"\brange\s+map|payload|feasibility|can\s+it\s+make\b", tl):
        return MissionCategory.ROUTE_FEASIBILITY
    if re.search(r"\bresale|sell|exit\b", tl):
        return MissionCategory.DISPOSITION
    if re.search(r"\bhow\s+many\s+seats|what(?:'s| is)\s+the\s+range\b", tl):
        return MissionCategory.SPECS
    if re.search(r"\bnon[- ]?stop|nonstop\b", tl):
        return MissionCategory.POINT_TO_POINT
    return None


def priority_from_text(text: str, *needles: str) -> PriorityLevel:
    tl = (text or "").lower()
    if any(n in tl for n in needles):
        if re.search(r"\b(critical|must|required|priority|high)\b", tl):
            return PriorityLevel.HIGH
        return PriorityLevel.MEDIUM
    return PriorityLevel.NONE


def format_budget_range(usd_mid: Optional[float]) -> Optional[str]:
    if usd_mid is None or usd_mid <= 0:
        return None
    millions = usd_mid / 1_000_000.0
    if millions >= 1:
        return f"~${millions:.1f}M"
    return f"~${int(usd_mid):,}"


def parse_budget_usd_mid(text: str) -> Optional[float]:
    best: Optional[float] = None
    pat = re.compile(
        r"\b(?:budget|around|under|~)\s*\$?\s*(\d{1,3})(?:\.\d+)?\s*(m|million|mil|k)?\b",
        re.I,
    )
    for m in pat.finditer(text or ""):
        try:
            val = float(m.group(1))
        except (TypeError, ValueError):
            continue
        suf = (m.group(2) or "").lower()
        if suf in ("m", "million", "mil"):
            val *= 1_000_000.0
        elif suf == "k":
            val *= 1_000.0
        else:
            val *= 1_000_000.0
        if best is None or val > best:
            best = val
    return best
