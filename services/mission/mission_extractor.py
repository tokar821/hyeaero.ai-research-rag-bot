"""
Strict per-turn mission extraction — current user message ONLY.

Returns typed :class:`MissionProfile` with validated :class:`Route` objects.

For strict JSON operational requirements (no aircraft recommendations), use
:class:`services.mission_extraction.MissionExtractionResult` via
:func:`services.mission_extraction.extract_mission_requirements_json`.
"""

from __future__ import annotations

import re
from typing import List, Optional

from services.mission.models import (
    MissionCategory,
    MissionProfile,
    PriorityLevel,
    Route,
)
from services.mission.normalization import (
    format_budget_range,
    infer_regions_from_places,
    normalize_mission_category,
    normalize_ownership,
    normalize_passenger_count,
    parse_budget_usd_mid,
    priority_from_text,
)
from services.mission.route_extractor import extract_routes, routes_from_extractions
from services.mission.validators import validate_profile

_PAX_RE = re.compile(
    r"\b(\d{1,2})\s*(?:passengers?|pax|people|executives?|seats?)\b"
    r"|\b(?:for\s+)?(\d{1,2})\s+pax\b"
    r"|\btravel\s+with\s+(\d{1,2})\b",
    re.I,
)
_PAX_RANGE_RE = re.compile(
    r"\b(\d{1,2})\s*(?:-|–|—|to)\s*(\d{1,2})\s*(?:passengers?|pax|people|executives?|seats?)\b",
    re.I,
)
_WESTBOUND_RE = re.compile(r"\bwestbound|west\s*bound\b", re.I)
_EASTBOUND_RE = re.compile(r"\beastbound|east\s*bound\b", re.I)
_WINTER_RE = re.compile(r"\bwinter|december|january|february|polar|headwind\b", re.I)
_MOUNTAIN_RE = re.compile(
    r"\b(aspen|telluride|jackson|sun\s+valley|high\s+elevation|short\s+runway|hot\s+and\s+high|hot/high)\b",
    re.I,
)
_NONSTOP_RE = re.compile(r"\bnon[- ]?stop|nonstop|direct\s+only\b", re.I)
_CABIN_RE = re.compile(
    r"\b(cabin|interior|stand[- ]?up|lav|galley|quiet|spacious|hotel|premium\s+feel|luxury)\b",
    re.I,
)
_BAGGAGE_RE = re.compile(r"\b(baggage|golf|ski|luggage|bulky)\b", re.I)
_RUNWAY_RE = re.compile(r"\b(\d{3,4})\s*(?:ft|foot|feet)\s+runway|short\s+field|runway\s+flex", re.I)
_OPERATING_COST_RE = re.compile(r"\boperating\s+cost|fuel\s+burn|direct\s+operating|lowest\s+cost\b", re.I)
_SHORT_FIELD_RE = re.compile(
    r"\bshort[- ]?field|short\s+runway|runway\s+flex|under\s+4[,.]?000\s*ft|island\s+hopping\b",
    re.I,
)
_INTERNATIONAL_RE = re.compile(
    r"\b(europe|asia|tokyo|london|paris|geneva|caribbean|south\s+america|international|transatlantic)\b",
    re.I,
)
_FREQ_RE = re.compile(r"\b(\d{2,4})\s+hours?\s+(?:per\s+)?(?:year|annually)\b", re.I)
_AIRPORT_CODE_RE = re.compile(r"\b([A-Z]{3,4})\b")


def _sanitize_user_message(user_message: str) -> str:
    from services.mission.route_extractor import sanitize_user_text_for_routes

    return sanitize_user_text_for_routes(user_message)


def _extract_passengers(text: str) -> Optional[int]:
    # Range forms: "8-10 executives" → take upper bound as planning load.
    m_rng = _PAX_RANGE_RE.search(text or "")
    if m_rng:
        try:
            lo = int(m_rng.group(1))
            hi = int(m_rng.group(2))
            if 0 < lo <= hi <= 80:
                n = normalize_passenger_count(hi)
                if n is not None:
                    return n
        except (TypeError, ValueError):
            pass
    for m in _PAX_RE.finditer(text):
        raw = m.group(1) or m.group(2) or m.group(3)
        if raw is None:
            continue
        try:
            n = normalize_passenger_count(int(raw))
            if n is not None:
                return n
        except (TypeError, ValueError):
            continue
    return None


def extract_mission(user_message: str) -> MissionProfile:
    """
    Build a fresh typed mission profile from the current user turn only.
    """
    text = _sanitize_user_message(user_message)
    profile = MissionProfile()

    if not text:
        return profile

    from services.mission.passenger_distribution import (
        apply_passenger_distribution_to_profile,
        extract_passenger_distribution,
    )

    dist = extract_passenger_distribution(text)
    apply_passenger_distribution_to_profile(profile, dist)
    if profile.passengers is None:
        profile.passengers = _extract_passengers(text)
    profile.routes = routes_from_extractions(extract_routes(user_message))
    try:
        from services.mission.mission_corridor_routes import (
            detect_field_access_posture,
            enrich_profile_routes_from_corridor,
        )

        enrich_profile_routes_from_corridor(text, profile)
        if detect_field_access_posture(text):
            profile.short_field_priority = PriorityLevel.HIGH
            if profile.runway_priority == PriorityLevel.NONE:
                profile.runway_priority = PriorityLevel.HIGH
    except Exception:
        pass

    usd = parse_budget_usd_mid(text)
    if usd is not None:
        profile.budget_usd_mid = usd
        profile.budget_range = format_budget_range(usd)

    profile.mission_category = normalize_mission_category(text)
    profile.ownership_interest = normalize_ownership(text)

    if _NONSTOP_RE.search(text):
        profile.nonstop_required = True
    if _WESTBOUND_RE.search(text):
        profile.westbound_sensitive = True
    if _EASTBOUND_RE.search(text):
        profile.eastbound_sensitive = True
    if _WINTER_RE.search(text):
        profile.seasonal_note = "winter_headwinds"
    if _MOUNTAIN_RE.search(text):
        profile.mountain_airports = True
        profile.mountain_airport_priority = True

    if _SHORT_FIELD_RE.search(text):
        profile.short_field_priority = PriorityLevel.HIGH
        if profile.runway_priority == PriorityLevel.NONE:
            profile.runway_priority = PriorityLevel.HIGH

    if _INTERNATIONAL_RE.search(text):
        profile.international_ops = True

    freq_m = _FREQ_RE.search(text)
    if freq_m:
        profile.mission_frequency = f"{freq_m.group(1)} hours/year"

    if _CABIN_RE.search(text):
        profile.cabin_priority = (
            PriorityLevel.HIGH
            if re.search(r"\bluxury|premium|hotel|expensive\b", text, re.I)
            else PriorityLevel.MEDIUM
        )
    if _OPERATING_COST_RE.search(text):
        profile.operating_cost_priority = PriorityLevel.HIGH
    if _RUNWAY_RE.search(text) or _MOUNTAIN_RE.search(text):
        profile.runway_priority = PriorityLevel.HIGH
    if _BAGGAGE_RE.search(text):
        profile.baggage_priority = PriorityLevel.HIGH

    codes = [c for c in _AIRPORT_CODE_RE.findall(text) if len(c) in (3, 4)]
    profile.preferred_airports = list(dict.fromkeys(codes))[:12]

    if re.search(r"\bnbaa|reserve|contingency\b", text, re.I):
        profile.reserves_requirement = "nbaa_standard"
        profile.nbaa_reserve_required = True

    profile.ownership_posture = profile.ownership_interest

    places: List[str] = []
    for r in profile.routes:
        places.extend([r.origin, r.destination])
    profile.regions = infer_regions_from_places(*places)

    return validate_profile(profile)
