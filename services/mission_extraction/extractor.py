"""
Mission Extraction Layer — natural language → strict operational JSON.

Does NOT recommend aircraft, ask follow-ups, or emit prose.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Set, Tuple

from services.mission.normalization import (
    normalize_mission_category,
    normalize_ownership,
    normalize_passenger_count,
    normalize_place,
    parse_budget_usd_mid,
)
from services.mission.route_extractor import extract_routes, routes_from_extractions
from services.mission_extraction.schema import (
    AircraftCategory,
    MissionExtractionResult,
    MissionType,
    PriorityLevel,
)

_PAX_RE = re.compile(
    r"\b(\d{1,2})\s*(?:passengers?|pax|people|executives?|seats?)\b"
    r"|\b(?:for\s+)?(\d{1,2})\s+pax\b"
    r"|\btravel\s+with\s+(\d{1,2})\b",
    re.I,
)
_EXECUTIVES_RE = re.compile(r"\bexecutives?\b", re.I)
_FROM_TO_AND_RE = re.compile(
    r"\bfrom\s+([a-z][a-z\s.-]{1,40}?)\s+to\s+([a-z][a-z\s.-]{1,40}?)"
    r"(?:\s+and\s+([a-z][a-z\s.-]{1,40}?))?(?:\s*[,;.]|$)",
    re.I,
)
_WESTBOUND_RE = re.compile(r"\bwestbound|west\s*bound\b", re.I)
_WINTER_RE = re.compile(r"\b(?:winter|december|january|february|polar|headwind)\b", re.I)
_NONSTOP_RE = re.compile(r"\bnon[- ]?stop|nonstop|direct\s+only\b", re.I)
_CABIN_RE = re.compile(
    r"\b(cabin|interior|stand[- ]?up|lav|galley|quiet|spacious|hotel|premium\s+feel|luxury)\b",
    re.I,
)
_BAGGAGE_RE = re.compile(r"\b(baggage|golf|ski|skis|luggage|bulky)\b", re.I)
_OPERATING_COST_RE = re.compile(
    r"\boperating\s+cost|fuel\s+burn|direct\s+operating|lowest\s+cost|"
    r"operating\s+economics|doc\b",
    re.I,
)
_RUNWAY_RE = re.compile(
    r"\b(\d{3,4})\s*(?:ft|foot|feet)\s+runway|short\s+field|runway\s+flex|runway\s+priority\b",
    re.I,
)
_SHORT_RUNWAY_RE = re.compile(
    r"\bshort[- ]?field|short\s+runway|runway\s+flex|under\s+4[,.]?000\s*ft|island\s+hopping\b",
    re.I,
)
_HOT_HIGH_RE = re.compile(r"\bhot\s*/?\s*high|hot\s+and\s+high\b", re.I)
_MOUNTAIN_RE = re.compile(
    r"\b(aspen|telluride|jackson|sun\s+valley|mountain\s+airport|ski\s+trip)\b",
    re.I,
)
_COMPARE_RE = re.compile(r"\bcompare|versus|vs\.?\b|head[- ]to[- ]head\b", re.I)
_FEASIBILITY_RE = re.compile(
    r"\bcan\s+(?:it|we|this)\s+(?:fly|make|do)\s+nonstop|feasib|nonstop\s+possible\b",
    re.I,
)
_ACQUISITION_RE = re.compile(
    r"\b(?:buy|purchase|acquire|shopping\s+for|in\s+the\s+market)\b", re.I
)
_HOURS_RE = re.compile(r"\b(\d{2,4})\s+hours?\s+(?:a|per)\s+year\b", re.I)

_EUROPE_RE = re.compile(
    r"\b(?:europe|london|paris|geneva|zurich|nice|milan|rome|madrid|amsterdam|frankfurt)\b",
    re.I,
)
_ASIA_RE = re.compile(
    r"\b(?:asia|tokyo|seoul|beijing|hong\s+kong|singapore|shanghai|mumbai|delhi)\b",
    re.I,
)
_CARIBBEAN_RE = re.compile(
    r"\b(?:caribbean|bahamas|nassau|st\.?\s*barths|turks|caicos|st\s+maarten)\b",
    re.I,
)
_SOUTH_AMERICA_RE = re.compile(
    r"\b(?:south\s+america|são\s+paulo|sao\s+paulo|buenos\s+aires|bogot[aá]|lima)\b",
    re.I,
)
_TRANSPACIFIC_RE = re.compile(r"\btranspacific|trans[- ]pacific\b", re.I)
_TRANSATLANTIC_RE = re.compile(r"\btransatlantic|trans[- ]atlantic\b", re.I)
_US_WEST_RE = re.compile(
    r"\b(?:san\s+francisco|los\s+angeles|seattle|sfo|lax|west\s+coast)\b",
    re.I,
)
_US_EAST_RE = re.compile(r"\b(?:new\s+york|nyc|boston|miami|teterboro|jfk)\b", re.I)


def _priority_from_match(text: str, pattern: re.Pattern[str]) -> Optional[PriorityLevel]:
    if not pattern.search(text):
        return None
    if re.search(
        r"\b(?:high|critical|priority|prioritize|must|non[- ]?negotiable)\b",
        text,
        re.I,
    ):
        return "high"
    if re.search(r"\b(?:low|minimize|ego)\b", text, re.I):
        return "low"
    return "medium"


def _extract_passengers(text: str) -> Optional[int]:
    for m in _PAX_RE.finditer(text):
        raw = m.group(1) or m.group(2) or m.group(3)
        if raw is None:
            continue
        try:
            return normalize_passenger_count(int(raw))
        except (TypeError, ValueError):
            continue
    if _EXECUTIVES_RE.search(text):
        return 6
    return None


def _places_from_routes(route_labels: List[str]) -> Tuple[Optional[str], List[str]]:
    """Derive origin + destination list from validated route legs."""
    if not route_labels:
        return None, []

    origins: List[str] = []
    dests: List[str] = []
    for label in route_labels:
        if "->" not in label:
            continue
        left, right = label.split("->", 1)
        o, d = normalize_place(left.strip()), normalize_place(right.strip())
        if o:
            origins.append(o)
        if d:
            dests.append(d)

    if not origins and not dests:
        return None, []

    origin = origins[0] if origins else None
    unique_dests: List[str] = []
    seen: Set[str] = set()
    for d in dests:
        key = d.lower()
        if key not in seen:
            seen.add(key)
            unique_dests.append(d)
    if origin and unique_dests and unique_dests[0].lower() == origin.lower():
        unique_dests = unique_dests[1:]
    return origin, unique_dests or None


def _extract_from_to_and(text: str) -> Tuple[Optional[str], Optional[List[str]]]:
    m = _FROM_TO_AND_RE.search(text)
    if not m:
        return None, None
    origin = normalize_place(m.group(1).strip())
    first_dest = normalize_place(m.group(2).strip())
    second = m.group(3)
    dests: List[str] = []
    if first_dest:
        dests.append(first_dest)
    if second:
        extra = normalize_place(second.strip())
        if extra and extra.lower() not in {d.lower() for d in dests}:
            dests.append(extra)
    return origin or None, dests or None


def _extract_origin_destinations(
    text: str,
    route_labels: List[str],
) -> Tuple[Optional[str], Optional[List[str]]]:
    origin, dests = _extract_from_to_and(text)
    if origin or dests:
        return origin, dests

    origin_r, dests_r = _places_from_routes(route_labels)
    if origin_r or dests_r:
        return origin_r, dests_r

    # Comma list without explicit "to": first place origin, rest destinations
    if "," in text and not re.search(r"\bto\b|\b->\b", text, re.I):
        working = re.sub(
            r",?\s*\d{1,2}\s*(?:passengers?|pax|people|executives?).*$",
            "",
            text,
            flags=re.I,
        ).strip()
        segments = [
            normalize_place(s.strip()) for s in working.split(",") if s.strip()
        ]
        segments = [s for s in segments if s]
        if len(segments) >= 2:
            return segments[0], segments[1:]

    return None, None


def _infer_mission_type(text: str) -> Optional[MissionType]:
    ownership = normalize_ownership(text)
    if ownership is not None:
        return "ownership"
    if _COMPARE_RE.search(text):
        return "comparison"
    if _FEASIBILITY_RE.search(text):
        return "feasibility"
    if _ACQUISITION_RE.search(text):
        return "acquisition"
    if re.search(r"\bto\b|\b->\b|→", text, re.I) or _FROM_TO_AND_RE.search(text):
        return "point_to_point"
    if "," in text and re.search(
        r"\b(?:dallas|new\s+york|chicago|miami|boston|los\s+angeles)\b", text, re.I
    ):
        return "multi_city"
    cat = normalize_mission_category(text)
    if cat is not None:
        mapping = {
            "comparison": "comparison",
            "ownership_structure": "ownership",
            "route_feasibility": "feasibility",
            "point_to_point": "point_to_point",
            "acquisition_advisory": "acquisition",
        }
        return mapping.get(cat.value, "general")
    return None


def _region_flags(text: str, destinations: Optional[List[str]]) -> dict[str, bool]:
    blob = text.lower()
    if destinations:
        blob += " " + " ".join(d.lower() for d in destinations)

    europe = bool(_EUROPE_RE.search(blob))
    asia = bool(_ASIA_RE.search(blob))
    caribbean = bool(_CARIBBEAN_RE.search(blob))
    south_america = bool(_SOUTH_AMERICA_RE.search(blob))
    transatlantic = bool(_TRANSATLANTIC_RE.search(blob)) or (
        bool(_US_EAST_RE.search(blob) or _US_WEST_RE.search(blob)) and europe
    )
    transpacific = bool(_TRANSPACIFIC_RE.search(blob)) or (
        bool(_US_WEST_RE.search(blob)) and asia
    )
    international = (
        europe
        or asia
        or caribbean
        or south_america
        or transatlantic
        or transpacific
        or bool(re.search(r"\binternational\b", blob))
    )
    return {
        "europe": europe,
        "asia": asia,
        "caribbean": caribbean,
        "south_america": south_america,
        "transatlantic": transatlantic,
        "transpacific": transpacific,
        "international_ops": international,
    }


def _estimate_max_leg_nm(route_labels: List[str]) -> float:
    if not route_labels:
        return 0.0
    try:
        from services.consultant.route_feasibility import estimate_route_distance_nm

        return max(estimate_route_distance_nm(r) for r in route_labels)
    except Exception:
        return 0.0


def _infer_aircraft_category(
    *,
    text: str,
    regions: dict[str, bool],
    max_leg_nm: float,
    mountain: bool,
    short_runway: bool,
    nonstop: bool,
    pax: Optional[int],
) -> Optional[AircraftCategory]:
    if regions.get("transpacific") and (nonstop or max_leg_nm >= 3500):
        return "ultra_long_range"
    if max_leg_nm >= 4800 or (regions.get("transatlantic") and nonstop and max_leg_nm >= 3000):
        return "ultra_long_range"
    if regions.get("transatlantic") or max_leg_nm >= 2600:
        return "large_cabin"
    if mountain or short_runway:
        return "midsize"
    if regions.get("caribbean") and max_leg_nm < 1500:
        return "midsize"
    if max_leg_nm > 0 and max_leg_nm < 900:
        return "light_jet"
    if max_leg_nm > 0 and max_leg_nm < 1700:
        return "super_midsize"
    if pax is not None and pax >= 12:
        return "large_cabin"
    if re.search(r"\bturboprop|pc-12|king\s+air|pilatus\b", text, re.I):
        return "turboprop"
    if regions.get("international_ops"):
        return "super_midsize"
    if max_leg_nm > 0:
        return "regional_utility"
    return None


def extract_mission_requirements(user_message: str) -> MissionExtractionResult:
    """
    Extract operational mission requirements from a single user message.

    Returns a validated :class:`MissionExtractionResult`. Never recommends aircraft.
    """
    raw = (user_message or "").strip()
    if not raw:
        return MissionExtractionResult()

    from services.mission.route_extractor import sanitize_user_text_for_routes

    text = sanitize_user_text_for_routes(raw)
    route_labels = [r.label() for r in routes_from_extractions(extract_routes(raw))]

    passengers = _extract_passengers(text)
    origin, destinations = _extract_origin_destinations(text, route_labels)
    mission_type = _infer_mission_type(text)
    regions = _region_flags(text, destinations)
    max_leg = _estimate_max_leg_nm(route_labels)

    ownership = normalize_ownership(text)
    ownership_str: Optional[str] = ownership.value if ownership else None

    hours_m = _HOURS_RE.search(text)
    annual_hours = int(hours_m.group(1)) if hours_m else None

    budget = parse_budget_usd_mid(text)
    nonstop = bool(_NONSTOP_RE.search(text)) or None
    westbound = bool(_WESTBOUND_RE.search(text)) or None
    winter = bool(_WINTER_RE.search(text)) or None
    mountain = bool(_MOUNTAIN_RE.search(text)) or None
    hot_high = bool(_HOT_HIGH_RE.search(text) or _MOUNTAIN_RE.search(text)) or None
    short_rw = bool(_SHORT_RUNWAY_RE.search(text) or _RUNWAY_RE.search(text)) or None

    runway_pri = _priority_from_match(text, _RUNWAY_RE) or (
        "high" if short_rw else None
    )
    op_cost_pri = _priority_from_match(text, _OPERATING_COST_RE)
    cabin_pri = _priority_from_match(text, _CABIN_RE)
    baggage_pri = _priority_from_match(text, _BAGGAGE_RE)

    category = _infer_aircraft_category(
        text=text,
        regions=regions,
        max_leg_nm=max_leg,
        mountain=bool(mountain),
        short_runway=bool(short_rw),
        nonstop=bool(nonstop),
        pax=passengers,
    )

    return MissionExtractionResult(
        passengers=passengers,
        origin=origin,
        destination=destinations,
        mission_type=mission_type,
        nonstop_required=nonstop,
        westbound_sensitive=westbound,
        winter_ops=winter,
        runway_priority=runway_pri,
        operating_cost_priority=op_cost_pri,
        cabin_priority=cabin_pri,
        baggage_priority=baggage_pri,
        ownership_interest=ownership_str,  # type: ignore[arg-type]
        annual_hours=annual_hours,
        budget=budget,
        hot_high_ops=hot_high,
        mountain_airports=mountain,
        short_runway_ops=short_rw,
        international_ops=regions["international_ops"] or None,
        transatlantic=regions["transatlantic"] or None,
        transpacific=regions["transpacific"] or None,
        south_america=regions["south_america"] or None,
        caribbean=regions["caribbean"] or None,
        europe=regions["europe"] or None,
        asia=regions["asia"] or None,
        inferred_aircraft_category=category,
    )


def extract_mission_requirements_json(user_message: str) -> str:
    """Return strict JSON string only — no markdown, no commentary."""
    result = extract_mission_requirements(user_message)
    return json.dumps(
        result.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )
