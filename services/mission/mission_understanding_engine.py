"""
Mission Understanding Engine v2 — operational reasoning BEFORE aircraft recommendation.

Decomposes the user turn into explicit + latent constraints, operational environment,
and corridor posture. Ranking and broker copy consume :class:`MissionUnderstandingPacket`,
not raw field extraction alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from services.consultant.mission_state import MissionState
from services.mission.models import MissionProfile, PriorityLevel
from services.mission.mission_profile_inference import (
    InferredMissionProfile,
    UtilizationStyle,
    apply_inference_to_profile,
    infer_mission_profile,
)
from services.recommendation.mission_ranker import classify_mission_category

MISSION_UNDERSTANDING_KEY = "mission_understanding_packet"

# Evidence collectors — composable signals (not canned response templates).
_EMPLOYEES_RE = re.compile(
    r"\b(\d{3,5})\s+(?:employees?|staff|people\s+at\s+the\s+company)\b", re.I
)
_REVENUE_RE = re.compile(
    r"\b(\d{1,4})\s*\+?\s*(?:million|m\b|bn|billion)\s+(?:revenue|sales|arr)\b"
    r"|\b(?:\$|usd\s*)?(\d{2,4})\s*m(?:illion)?\s+(?:revenue|company)\b",
    re.I,
)
_EUROPE_FREQ_RE = re.compile(
    r"\b(\d{1,3})\s+(?:europe|european|transatlantic)\s+trips?\b"
    r"|\beurope\s+(?:several|multiple|\d+)\s+times?\s+(?:per\s+)?(?:year|quarter|month)\b"
    r"|\b(?:weekly|monthly|quarterly)\s+(?:europe|london|paris)\b",
    re.I,
)
_CARIBBEAN_RE = re.compile(
    r"\b(?:caribbean|bahamas|nassau|st\s+maarten|(?:caribbean|tropical)\s+islands?)\b",
    re.I,
)
_SOUTH_AMERICA_RE = re.compile(r"\b(?:south\s+america|brazil|são\s+paulo|sao\s+paulo)\b", re.I)
_RUNWAY_OVER_LUXURY_RE = re.compile(
    r"\b(?:runway|field)\s+(?:flex|flexibility|access|performance)\b.*\b(?:>|over|vs\.?|than)\b.*\b(?:luxury|cabin)\b"
    r"|\b(?:luxury|cabin)\b.*\b(?:less|lower|secondary|overrated)\b",
    re.I,
)
_MULTI_PACIFIC_RE = re.compile(
    r"\b(?:tokyo|japan|hong\s+kong|seoul|pacific|asia)\b", re.I
)
_TRANSATLANTIC_RE = re.compile(
    r"\b(?:london|paris|geneva|europe|transatlantic|uk|berlin|moscow|frankfurt|munich|zurich)\b",
    re.I,
)
_EUROPE_CITY_RE = re.compile(
    r"\b(?:london|paris|geneva|berlin|moscow|moscaw|frankfurt|munich|zurich|dublin|rome|milan)\b",
    re.I,
)
_FREQ_LOW_RE = re.compile(
    r"\b(?:twice\s+(?:a\s+)?month|bi[- ]?monthly|2x\s+month|few\s+times\s+(?:a\s+)?month|"
    r"monthly|quarterly|occasional(?:ly)?)\b",
    re.I,
)
_MIDDLE_EAST_ULR_RE = re.compile(
    r"\b(?:dubai|riyadh|doha|abu\s+dhabi|middle\s+east|jeddah)\b",
    re.I,
)
_CORROSION_ENV_RE = re.compile(r"\b(?:caribbean|salt|corrosion|humid|tropical)\b", re.I)
_ISLAND_OPS_RE = re.compile(
    r"\b(?:island|short\s+runway|runway\s+flex|unpaved|gravel|st\s+thomas|nassau)\b",
    re.I,
)

_PAX_NORM_RE = re.compile(
    r"\b(\d{1,3})\s*(?:executives?|people|pax|passengers)\b",
    re.I,
)
_MULTI_ROLE_MOUNTAIN_RE = re.compile(
    r"\b(?:aspen|jackson\s+hole|mountain|short[- ]field|short[- ]strip|"
    r"telluride|sun\s+valley)\b",
    re.I,
)
_INDUSTRIAL_FIELD_RE = re.compile(
    r"\b(?:industrial\s+airports?|smaller\s+industrial|factory\s+site|plant\s+site|"
    r"oil\s+sites?|remote\s+(?:oil|field))\b",
    re.I,
)
_ONE_AIRCRAFT_RE = re.compile(
    r"\bone aircraft\b|\bone large jet\b|\bone jet\b|"
    r"\bconsolidat(?:e|ing)\s+(?:two|2)\s+aircraft\b|\bsingle platform\b",
    re.I,
)
_OWNERSHIP_DEBATE_RE = re.compile(
    r"\b(?:charter|fractional|ownership|own(?:ing)?|netjets|"
    r"debating ownership|justify ownership)\b",
    re.I,
)
_CHARTER_HOURS_RE = re.compile(
    r"(?:around\s+)?(\d{2,3})\s+hours?\s+(?:annually|a year|per year|/year)",
    re.I,
)
_FUEL_STOP_AVERSE_RE = re.compile(
    r"\b(?:hate|no|avoid|zero|without)\s+(?:fuel\s+)?stops?\b"
    r"|\bfuel\s+stop(?:s)?\s+(?:are|is)\s+(?:unacceptable|not)\b",
    re.I,
)
_BAND_ULR_RE = re.compile(r"ultra-long|transatlantic|multi-leg ultra", re.I)
_BAND_SHORT_FIELD_RE = re.compile(
    r"mountain|short-runway|short-strip|field-access|domestic field", re.I
)

_RECOMMEND_AIRCRAFT_MIN_CONF = 0.62


@dataclass
class MissionUnderstandingPacket:
    explicit_constraints: Dict[str, Any] = field(default_factory=dict)
    inferred_constraints: Dict[str, Any] = field(default_factory=dict)
    operational_environment: List[str] = field(default_factory=list)
    ownership_profile: str = "unknown"
    travel_pattern: str = "unknown"
    corridor_type: str = "unknown"
    runway_complexity: str = "standard"
    dispatch_priority: str = "standard"
    comfort_priority: str = "standard"
    operating_cost_priority: str = "standard"
    nonstop_priority: str = "standard"
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    overall_confidence: float = 0.0
    fallback_operational_band: List[str] = field(default_factory=list)
    operational_synthesis: str = ""
    recommend_aircraft: bool = False
    utilization_style: str = UtilizationStyle.UNKNOWN
    understanding_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "explicit_constraints": dict(self.explicit_constraints),
            "inferred_constraints": dict(self.inferred_constraints),
            "operational_environment": list(self.operational_environment),
            "ownership_profile": self.ownership_profile,
            "travel_pattern": self.travel_pattern,
            "corridor_type": self.corridor_type,
            "runway_complexity": self.runway_complexity,
            "dispatch_priority": self.dispatch_priority,
            "comfort_priority": self.comfort_priority,
            "operating_cost_priority": self.operating_cost_priority,
            "nonstop_priority": self.nonstop_priority,
            "confidence_scores": dict(self.confidence_scores),
            "overall_confidence": round(self.overall_confidence, 3),
            "fallback_operational_band": list(self.fallback_operational_band),
            "operational_synthesis": self.operational_synthesis,
            "recommend_aircraft": self.recommend_aircraft,
            "utilization_style": self.utilization_style,
            "understanding_notes": list(self.understanding_notes),
        }


def _history_user_blob(history: Optional[Sequence[Dict[str, str]]], limit: int = 8) -> str:
    if not history:
        return ""
    parts: List[str] = []
    for turn in list(history)[-limit:]:
        if not isinstance(turn, dict):
            continue
        if str(turn.get("role") or "").lower() != "user":
            continue
        c = (turn.get("content") or "").strip()
        if c:
            parts.append(c)
    return "\n".join(parts)


def _peak_stage_nm(profile: MissionProfile) -> float:
    best = 0.0
    for label in profile.route_labels():
        try:
            from services.consultant.route_feasibility import estimate_route_distance_nm

            best = max(best, float(estimate_route_distance_nm(label) or 0))
        except Exception:
            pass
    return best


def _infer_corridor_and_band(
    profile: MissionProfile,
    mission: MissionState,
    *,
    peak_nm: float,
    route_labels_override: Optional[Sequence[str]] = None,
) -> tuple[str, List[str]]:
    blob = " ".join(route_labels_override or profile.route_labels()).lower()
    category = classify_mission_category(mission)
    pax = int(profile.passengers or mission.passenger_count or 0)

    if peak_nm >= 4500 or (_MULTI_PACIFIC_RE.search(blob) and _TRANSATLANTIC_RE.search(blob)):
        return (
            "multi_leg_ultra_long",
            ["Multi-leg ultra-long-range executive band"],
        )

    transatlantic_signal = (
        peak_nm >= 2800
        or _TRANSATLANTIC_RE.search(blob)
        or getattr(category, "value", str(category)) == "transatlantic_executive"
        or _EUROPE_CITY_RE.search(blob)
    )
    if transatlantic_signal:
        ulr_required = (
            peak_nm >= 5000
            or _MIDDLE_EAST_ULR_RE.search(blob)
            or (profile.nonstop_required and peak_nm >= 4200)
        )
        if ulr_required and pax >= 8:
            return (
                "transatlantic_ulr",
                ["Transatlantic ultra-long-range executive band"],
            )
        if pax <= 4:
            return (
                "transatlantic_super_mid",
                [
                    "Transatlantic super-mid / heavy-cabin band — modest passenger load "
                    "rarely justifies ULR-class capital; supplemental charter on peak legs "
                    "is often more rational than G650-class ownership."
                ],
            )
        if pax <= 6 and not profile.nonstop_required:
            return (
                "transatlantic_super_mid",
                [
                    "Transatlantic super-mid / heavy-cabin band — frequency and cabin "
                    "utilization do not require ULR-class unless nonstop on longest stage is mandatory."
                ],
            )
        if pax <= 8 and peak_nm < 4500:
            return (
                "transatlantic_heavy",
                ["Transatlantic heavy-cabin / super-mid executive band"],
            )
        return (
            "transatlantic_ulr",
            ["Transatlantic ultra-long-range executive band"],
        )
    if _CARIBBEAN_RE.search(blob) or _SOUTH_AMERICA_RE.search(blob):
        return (
            "caribbean_regional",
            ["Caribbean island-ops short-runway band"],
        )
    if profile.mountain_airports or profile.mountain_airport_priority:
        return (
            "mountain_field",
            ["Mountain field-flexible short-strip band"],
        )
    if peak_nm >= 1700:
        return (
            "continental_super_mid",
            ["Long-leg super-mid executive band"],
        )
    if peak_nm > 0:
        return (
            "regional_midsize",
            ["Regional super-mid executive band"],
        )
    return ("unspecified_corridor", ["Executive super-mid planning band"])


def _collect_enterprise_signals(
    text: str,
    packet: MissionUnderstandingPacket,
) -> None:
    m_emp = _EMPLOYEES_RE.search(text)
    m_rev = _REVENUE_RE.search(text)
    m_eu = _EUROPE_FREQ_RE.search(text)

    if m_emp:
        n = int(m_emp.group(1))
        packet.inferred_constraints["enterprise_employees"] = n
        packet.confidence_scores["enterprise"] = min(1.0, 0.35 + (n / 5000.0) * 0.4)
        packet.understanding_notes.append(f"Enterprise scale ~{n} employees inferred.")
        if n >= 500:
            packet.utilization_style = UtilizationStyle.BOARD_TRANSPORT
            packet.ownership_profile = "corporate_shuttle_candidate"
            packet.travel_pattern = "executive_shuttle"
            packet.dispatch_priority = "high"
            packet.nonstop_priority = "high"
            packet.inferred_constraints["ownership_viability_threshold"] = "fractional_or_full"

    if m_rev:
        packet.inferred_constraints["enterprise_revenue_signal"] = True
        packet.confidence_scores["enterprise"] = max(
            packet.confidence_scores.get("enterprise", 0.0), 0.55
        )
        packet.ownership_profile = "corporate_shuttle_candidate"
        packet.inferred_constraints["capital_efficiency_relevant"] = True

    if m_eu:
        trips = int(m_eu.group(1)) if m_eu.group(1) else 12
        packet.inferred_constraints["europe_trip_frequency"] = trips
        packet.travel_pattern = "transatlantic_executive"
        packet.corridor_type = "transatlantic_ulr"
        packet.nonstop_priority = "high"
        packet.dispatch_priority = "high"
        packet.inferred_constraints["crew_and_ownership_relevant"] = True
        packet.confidence_scores["transatlantic_utilization"] = min(1.0, 0.4 + trips / 80.0)


def _collect_regional_environment(
    text: str,
    profile: MissionProfile,
    packet: MissionUnderstandingPacket,
) -> None:
    try:
        from services.mission.mission_place_index import places_captured_from_mission

        captured = places_captured_from_mission(profile, text)
        packet.explicit_constraints["places_captured"] = captured
        multi_continent = len(
            {p for p in captured if p not in ("Caribbean", "West Africa", "Europe", "Transatlantic")}
        ) >= 3 and (
            _EUROPE_CITY_RE.search(text)
            or any(c in captured for c in ("Frankfurt", "London", "Zurich", "Paris"))
        )
    except Exception:
        multi_continent = False

    caribbean_scope = bool(_CARIBBEAN_RE.search(text)) or (
        _SOUTH_AMERICA_RE.search(text)
        and not multi_continent
        and not _EUROPE_CITY_RE.search(text)
        and "caribbean" in (text or "").lower()
    )
    if caribbean_scope:
        packet.corridor_type = "caribbean_regional"
        packet.fallback_operational_band = list(
            dict.fromkeys(
                packet.fallback_operational_band
                + ["Caribbean executive regional jet band"]
            )
        )
        packet.operational_environment.extend(
            [
                "Island and tropical operating environment — heat, humidity, and corrosion exposure.",
                "Runway access matters, but executive dispatch and pressurized jet credibility remain the floor.",
            ]
        )
        packet.runway_complexity = "regional_access"
        packet.inferred_constraints["corrosion_exposure"] = True
        packet.inferred_constraints["island_ops"] = True
        packet.confidence_scores["regional_environment"] = 0.72
    elif _SOUTH_AMERICA_RE.search(text) and multi_continent:
        packet.fallback_operational_band = list(
            dict.fromkeys(
                packet.fallback_operational_band
                + ["Latin America / transatlantic executive band"]
            )
        )

    if _CORROSION_ENV_RE.search(text):
        packet.inferred_constraints["corrosion_exposure"] = True
    if _ISLAND_OPS_RE.search(text):
        if packet.runway_complexity != "mountain":
            packet.runway_complexity = "regional_access"
        packet.inferred_constraints["short_runway_likely"] = True

    if _RUNWAY_OVER_LUXURY_RE.search(text) or (
        profile.runway_priority in (PriorityLevel.HIGH, PriorityLevel.MEDIUM)
        and profile.cabin_priority == PriorityLevel.NONE
    ):
        packet.comfort_priority = "secondary"
        if not packet.inferred_constraints.get("mountain_ops"):
            packet.runway_complexity = "regional_access"
        packet.inferred_constraints["runway_over_cabin"] = True
        if _EUROPE_CITY_RE.search(text) or profile.international_ops:
            packet.inferred_constraints["international_jet_floor"] = True
            packet.inferred_constraints["field_capable_super_mid_floor"] = True
        packet.confidence_scores["priority_tradeoff"] = 0.8


def _collect_passenger_utilization_realism(
    text: str,
    profile: MissionProfile,
    packet: MissionUnderstandingPacket,
) -> None:
    """Moderate cabin-class escalation from passenger count, frequency, and economics."""
    pax = profile.passengers
    if pax is None:
        m = _PAX_NORM_RE.search(text)
        if m:
            try:
                pax = int(m.group(1))
            except (TypeError, ValueError):
                pax = None
    if not isinstance(pax, int) or pax <= 0:
        return

    low_frequency = bool(_FREQ_LOW_RE.search(text))
    corridor = packet.corridor_type or ""

    if pax <= 4 and corridor.startswith("transatlantic"):
        packet.inferred_constraints["cabin_utilization_modest"] = True
        packet.inferred_constraints["planning_band_ceiling"] = "super_midsize"
        packet.inferred_constraints["supplemental_charter_viable"] = True
        packet.ownership_profile = packet.ownership_profile or "fractional_or_charter_supplement"
        packet.understanding_notes.append(
            f"{pax} passengers on transatlantic corridor — avoid automatic ULR escalation; "
            "super-mid + occasional charter supplement is the rational planning frame."
        )
        if packet.corridor_type == "transatlantic_ulr":
            packet.corridor_type = "transatlantic_super_mid"
        packet.fallback_operational_band = [
            b.replace("ultra-long-range", "super-mid / heavy-cabin")
            if "ultra-long" in b.lower()
            else b
            for b in packet.fallback_operational_band
        ]

    if pax <= 6 and low_frequency and "transatlantic" in corridor:
        packet.inferred_constraints["supplemental_charter_viable"] = True
        packet.inferred_constraints.setdefault("planning_band_ceiling", "super_midsize")
        packet.understanding_notes.append(
            "Low-frequency transatlantic utilization — structure and supplemental lift "
            "precede ULR-class capital commitment."
        )

    if pax >= 10 and corridor.startswith("transatlantic"):
        packet.inferred_constraints.pop("planning_band_ceiling", None)


def _collect_international_operational_posture(
    text: str,
    profile: MissionProfile,
    packet: MissionUnderstandingPacket,
    *,
    peak_nm: float,
) -> None:
    """Europe / intercontinental missions — jet floor, winter margin, dispatch practicality."""
    blob = " ".join(profile.route_labels()).lower() + " " + text.lower()
    if not (_EUROPE_CITY_RE.search(blob) or profile.international_ops):
        return

    profile.international_ops = True
    packet.inferred_constraints["international_jet_floor"] = True
    packet.inferred_constraints.setdefault("planning_band_ceiling", "super_midsize")

    if peak_nm >= 3000 or _EUROPE_CITY_RE.search(blob):
        packet.inferred_constraints["transatlantic_super_mid_floor"] = True
        packet.operational_environment.append(
            "Europe / intercontinental stage — light-jet and entry-level midsize bands lack "
            "consistent winter margin, pressurization comfort, and dispatch practicality at full passenger load."
        )

    if re.search(r"\bwinter\b|\bheadwind\b|\bdecember\b|\bjanuary\b", text, re.I):
        packet.inferred_constraints["westbound_winter_pressure"] = True
        packet.operational_environment.append(
            "Winter transatlantic planning — reserve fuel and alternate policy dominate; "
            "cost-optimized light jets are operationally thin."
        )

    if isinstance(profile.passengers, int) and profile.passengers >= 6:
        packet.inferred_constraints["minimum_jet_cabin_floor"] = True


def _collect_priority_balance(
    text: str,
    profile: MissionProfile,
    packet: MissionUnderstandingPacket,
) -> None:
    """Cost + runway flexibility must not collapse to utility/light jets on international legs."""
    cost_high = (
        packet.operating_cost_priority == "high"
        or profile.operating_cost_priority in (PriorityLevel.HIGH, PriorityLevel.MEDIUM)
    )
    runway_flex = (
        packet.runway_complexity in ("regional_access", "high", "mountain")
        or profile.runway_priority in (PriorityLevel.HIGH, PriorityLevel.MEDIUM)
        or packet.inferred_constraints.get("runway_over_cabin")
    )
    if not (cost_high and runway_flex):
        return

    packet.inferred_constraints["balanced_cost_dispatch"] = True
    if packet.inferred_constraints.get("international_jet_floor") or profile.international_ops:
        packet.inferred_constraints["field_capable_super_mid_floor"] = True
        packet.inferred_constraints.setdefault("planning_band_ceiling", "super_midsize")
        packet.understanding_notes.append(
            "Cost-sensitive but runway-flexible international mission — planning floor is "
            "field-capable super-mid (pressurized jet dispatch), not light-jet economics."
        )
    if packet.comfort_priority == "secondary":
        packet.inferred_constraints["dispatch_over_pure_cost"] = True


def _collect_route_posture(
    profile: MissionProfile,
    mission: MissionState,
    packet: MissionUnderstandingPacket,
    *,
    peak_nm_override: Optional[float] = None,
    route_labels_override: Optional[Sequence[str]] = None,
) -> None:
    peak = peak_nm_override if peak_nm_override is not None else _peak_stage_nm(profile)
    corridor, band = _infer_corridor_and_band(
        profile,
        mission,
        peak_nm=peak,
        route_labels_override=route_labels_override,
    )
    if packet.corridor_type == "unknown":
        packet.corridor_type = corridor
    packet.fallback_operational_band = list(
        dict.fromkeys(packet.fallback_operational_band + band)
    )

    if profile.nonstop_required or mission.nonstop_requirement:
        packet.nonstop_priority = "high"
        packet.explicit_constraints["nonstop"] = True
    if mission.westbound or profile.westbound_sensitive:
        packet.inferred_constraints["westbound_winter_pressure"] = bool(
            (mission.seasonal_constraints or "").lower().find("winter") >= 0
            or (profile.seasonal_note or "").find("winter") >= 0
        )
        packet.operational_environment.append(
            "Westbound long-stage planning — headwind and alternate fuel drive dispatch reliability."
        )
    if len(profile.routes) >= 2 or len(mission.routes or []) >= 2:
        packet.travel_pattern = "multi_leg"
        packet.inferred_constraints["dual_use_or_multi_leg"] = True
        packet.understanding_notes.append("Multiple city-pairs — treat as portfolio mission, not single leg.")


def _collect_traveler_posture(
    text: str,
    profile: MissionProfile,
    packet: MissionUnderstandingPacket,
) -> None:
    """Infer executive vs utility posture from passenger load and language."""
    pax = profile.passengers
    if pax is None:
        m = _PAX_NORM_RE.search(text)
        if m:
            try:
                pax = int(m.group(1))
            except (TypeError, ValueError):
                pax = None

    if isinstance(pax, int) and 6 <= pax <= 14:
        if packet.utilization_style == UtilizationStyle.UNKNOWN:
            packet.utilization_style = UtilizationStyle.EXECUTIVE_SHUTTLE
        packet.inferred_constraints["executive_travel_profile"] = True
        packet.dispatch_priority = "high"
        if pax >= 8:
            packet.inferred_constraints["minimum_jet_cabin_floor"] = True
        packet.confidence_scores["traveler_posture"] = 0.65

    if _EXECUTIVE_RE.search(text):
        packet.utilization_style = UtilizationStyle.EXECUTIVE_SHUTTLE
        packet.inferred_constraints["executive_travel_profile"] = True
        packet.dispatch_priority = "high"


def _collect_regional_hub_corridor(
    text: str,
    profile: MissionProfile,
    packet: MissionUnderstandingPacket,
) -> None:
    """When regions are named without city pairs, infer hub-spoke corridor posture."""
    try:
        from services.mission.mission_corridor_routes import (
            count_resolved_city_hubs_in_text,
            enrich_profile_routes_from_corridor,
        )

        if count_resolved_city_hubs_in_text(text) >= 2:
            return
        if enrich_profile_routes_from_corridor(text, profile):
            return
    except Exception:
        pass

    if profile.routes:
        return
    ql = text.lower()
    hub: Optional[str] = None
    if re.search(r"\bmiami\b", ql):
        hub = "Miami"
    elif re.search(r"\bnew york\b|\bnyc\b", ql):
        hub = "New York"
    elif re.search(r"\bdallas\b", ql):
        hub = "Dallas"

    if not hub:
        return
    if not (_CARIBBEAN_RE.search(text) or _SOUTH_AMERICA_RE.search(text)):
        return

    # Anchor on catalog-verified Caribbean leg; South America is regional scope, not an unresolved long stage.
    routes = [f"{hub} -> Caribbean"]
    packet.explicit_constraints["routes"] = routes
    packet.explicit_constraints["regions_served"] = ["Caribbean", "South America"]
    from services.mission.models import Route

    for lbl in routes:
        r = Route.from_label(lbl)
        if r:
            profile.routes.append(r)
    profile.international_ops = True
    for region in ("Caribbean", "South America"):
        if region not in profile.regions:
            profile.regions.append(region)
    packet.travel_pattern = "regional_executive_international"
    packet.corridor_type = "caribbean_regional"
    packet.confidence_scores["regional_hub_inference"] = 0.52
    packet.understanding_notes.append(
        f"Hub-spoke regional corridor inferred from {hub} + Caribbean/South America references."
    )


_EXECUTIVE_RE = re.compile(
    r"\b(?:executives?|leadership|board|ceo|cfo|partners?|private equity)\b",
    re.I,
)


def _collect_multi_role_signals(
    text: str,
    packet: MissionUnderstandingPacket,
    profile: Optional[MissionProfile] = None,
) -> None:
    """Detect incompatible mission bands from destination mix in the query."""
    has_transatlantic = bool(
        _TRANSATLANTIC_RE.search(text) or re.search(r"\bdubai\b", text, re.I)
    )
    has_middle_east = bool(_MIDDLE_EAST_ULR_RE.search(text))
    route_blob = " ".join(profile.route_labels() if profile else [])
    has_mountain = bool(
        _MULTI_ROLE_MOUNTAIN_RE.search(f"{text} {route_blob}")
        and re.search(
            r"\b(?:aspen|jackson\s+hole|telluride|sun\s+valley|ktex|kase|kege)\b",
            f"{text} {route_blob}",
            re.I,
        )
    )
    has_industrial = bool(
        _INDUSTRIAL_FIELD_RE.search(text)
        or re.search(
            r"\b(?:remote\s+drilling|arctic\s+oil|west\s+africa|oil\s+platform)\b",
            route_blob,
            re.I,
        )
    )
    has_caribbean = bool(_CARIBBEAN_RE.search(text))
    has_pacific = bool(_MULTI_PACIFIC_RE.search(text))

    bands_to_add: List[str] = []
    if has_transatlantic:
        bands_to_add.append("Transatlantic super-mid / heavy-cabin executive band")
    if has_pacific:
        bands_to_add.append("Multi-leg ultra-long-range executive band")
    if has_middle_east:
        bands_to_add.append("Middle East ULR continuation band")
    if has_industrial:
        bands_to_add.append("Domestic field-access executive band")
        packet.runway_complexity = "regional_access"
        packet.inferred_constraints["industrial_airport_access"] = True
        if re.search(
            r"\b(?:matters?\s+more\s+than\s+cabin|runway|unpaved|reliability\s+into)\b",
            text,
            re.I,
        ):
            packet.inferred_constraints["runway_over_cabin"] = True
    if has_mountain:
        bands_to_add.append("Mountain field-flexible short-strip band")
        packet.runway_complexity = "mountain"
        packet.inferred_constraints["mountain_ops"] = True
    if has_caribbean:
        bands_to_add.append("Caribbean executive regional jet band")

    packet.fallback_operational_band = list(
        dict.fromkeys(packet.fallback_operational_band + bands_to_add)
    )

    if bands_are_incompatible(packet.fallback_operational_band):
        packet.inferred_constraints["incompatible_mission_bands"] = True
        packet.inferred_constraints["dual_use_or_multi_leg"] = True
        packet.travel_pattern = "multi_leg"
        packet.understanding_notes.append(
            "Incompatible operational bands detected — portfolio mission."
        )
    elif len(bands_to_add) >= 2 or (has_pacific and has_transatlantic):
        packet.inferred_constraints["dual_use_or_multi_leg"] = True
        packet.travel_pattern = "multi_leg"
        packet.understanding_notes.append(
            "Multi-corridor mission — single ULR platform may cover all stated long-range legs."
        )
        if has_middle_east and has_transatlantic:
            packet.understanding_notes.append(
                "Continuation mission detected — Middle East nonstop legs change the planning band for that subset."
            )

    if _ONE_AIRCRAFT_RE.search(text):
        packet.inferred_constraints["single_aircraft_request"] = True


def _collect_ownership_signals(text: str, packet: MissionUnderstandingPacket) -> None:
    if _OWNERSHIP_DEBATE_RE.search(text):
        packet.inferred_constraints["ownership_economics_relevant"] = True
        if packet.ownership_profile in ("unknown", ""):
            packet.ownership_profile = "charter_transition"

    if re.search(
        r"\boutgrown\b.*\b(?:current|setup|aircraft|jet|program)\b"
        r"|\b(?:current|existing)\s+(?:setup|aircraft|jet|program)\b",
        text,
        re.I,
    ):
        packet.inferred_constraints["ownership_economics_relevant"] = True
        packet.inferred_constraints["upgrade_from_current_setup"] = True
        if packet.ownership_profile in ("unknown", ""):
            packet.ownership_profile = "charter_transition"

    m = _CHARTER_HOURS_RE.search(text)
    if m:
        try:
            hrs = int(m.group(1))
            packet.inferred_constraints["annual_charter_hours"] = hrs
            packet.inferred_constraints["ownership_economics_relevant"] = True
            packet.confidence_scores["ownership_economics"] = 0.7
        except (TypeError, ValueError):
            pass

    if _FUEL_STOP_AVERSE_RE.search(text):
        packet.nonstop_priority = "high"
        packet.inferred_constraints["fuel_stop_averse"] = True
        packet.dispatch_priority = "high"


def bands_are_incompatible(bands: Sequence[str]) -> bool:
    if len(bands) < 2:
        return False
    has_ulr = any(_BAND_ULR_RE.search(b) for b in bands)
    has_short = any(_BAND_SHORT_FIELD_RE.search(b) for b in bands)
    return has_ulr and has_short


def needs_portfolio_synthesis(
    query: str,
    packet: MissionUnderstandingPacket,
) -> bool:
    """Alias — fleet doctrine only when structural decomposition is proven."""
    from services.mission.structural_decomposition import needs_structural_decomposition

    return needs_structural_decomposition(packet, query=query).required


def needs_structural_decomposition_flag(packet: MissionUnderstandingPacket) -> bool:
    from services.mission.structural_decomposition import needs_structural_decomposition

    return needs_structural_decomposition(packet).required


def needs_ownership_overlay(query: str, packet: MissionUnderstandingPacket) -> bool:
    if packet.inferred_constraints.get("ownership_economics_relevant"):
        return True
    ql = (query or "").lower()
    if _OWNERSHIP_DEBATE_RE.search(ql) and (
        _CHARTER_HOURS_RE.search(ql) or "debating ownership" in ql
    ):
        return True
    return False


def format_portfolio_synthesis(
    query: str,
    packet: MissionUnderstandingPacket,
) -> str:
    bands = packet.fallback_operational_band[:4]
    lines = [
        "Fleet Structure:",
        "",
        "* This mission spans incompatible operational bands — one aircraft is unlikely to cover both credibly.",
    ]
    if packet.inferred_constraints.get("single_aircraft_request"):
        lines.append(
            "* I would not sell this as a single platform — the bands below trade runway access against oceanic nonstop range."
        )
    if any(_BAND_ULR_RE.search(b) for b in bands) and any(
        _BAND_SHORT_FIELD_RE.search(b) for b in bands
    ):
        lines.append(
            "* Typical multi-aircraft portfolio: an ultra-long-range platform for transatlantic/oceanic legs, "
            "plus a field-performance type for mountain, island, or industrial airport access."
        )
        lines.append(
            "* Consolidating into one jet forces a compromise — either short-field capability or nonstop range "
            "will suffer on the missions you care about most."
        )
    else:
        lines.append(
            "* Consider splitting by mission band — separate platforms often beat one compromised airframe."
        )
    return "\n".join(lines)


def format_ownership_economics_overlay(
    query: str,
    mission: MissionState,
) -> str:
    from services.orchestration.ownership_simulator import simulate_ownership_economics

    sim = simulate_ownership_economics(query, mission=mission)
    hours = sim.annual_hours or 250
    lines = [
        "Ownership Economics:",
        "",
        f"* Utilization: {sim.utilization_band}",
        f"* Structure: {sim.structure_recommendation.replace('_', ' ')}",
    ]
    if hours:
        lines.append(
            f"* At ~{hours} hr/year, compare burdened ownership DOC (~${int(sim.all_in_hour_usd):,}/hr all-in, directional) "
            f"to fractional or charter before anchoring on airframe."
        )
    if hours and hours < 200:
        lines.append(
            "* Below ~200 hr/year, full ownership rarely beats fractional unless dispatch control is non-negotiable."
        )
    elif hours and hours >= 250:
        lines.append(
            "* In the 250–400 hr band, ownership becomes rational if charter friction and schedule control "
            "outweigh capital efficiency — verify downtime and crew assumptions."
        )
    else:
        lines.append(
            "* Size utilization and structure first; aircraft model follows once the ownership band is credible."
        )
    return "\n".join(lines)


def _synthesize_operational_prose(packet: MissionUnderstandingPacket) -> str:
    lines: List[str] = []
    if packet.operational_environment:
        lines.append(" ".join(packet.operational_environment[:2]))
    if packet.ownership_profile not in ("unknown", ""):
        lines.append(
            f"Ownership posture reads as {packet.ownership_profile.replace('_', ' ')} "
            f"with {packet.travel_pattern.replace('_', ' ')} utilization."
        )
    if packet.runway_complexity == "high":
        lines.append(
            "Field-performance and runway access outweigh brochure cabin metrics on this profile."
        )
    if packet.nonstop_priority == "high":
        lines.append(
            "Nonstop and dispatch reliability are latent priorities even when not repeated in this turn."
        )
    if packet.fallback_operational_band:
        band = ", ".join(packet.fallback_operational_band[:4])
        lines.append(f"Credible aircraft class conversation starts in: {band}.")
    if not lines:
        lines.append(
            "Operational picture is still forming — corridor and priorities need one more anchor "
            "(city pair or frequency) before a tight shortlist."
        )
    return " ".join(lines)


def build_mission_understanding(
    query: str,
    profile: MissionProfile,
    mission: MissionState,
    *,
    broker_memory: Optional[Dict[str, Any]] = None,
    history: Optional[Sequence[Dict[str, str]]] = None,
    inferred: Optional[InferredMissionProfile] = None,
    use_llm: Optional[bool] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> MissionUnderstandingPacket:
    """
    Full mission understanding pass — runs after field extraction, before ranking.
    """
    ql = (query or "").strip()
    hist_blob = _history_user_blob(history)
    combined = f"{hist_blob}\n{ql}".strip()

    packet = MissionUnderstandingPacket()

    from services.mission.mission_context_reconciliation import assess_mission_continuity

    continuity = assess_mission_continuity(
        ql,
        profile,
        broker_memory=broker_memory,
        prior_graph=None,
    )
    context_text = combined if continuity.apply_structural_memory else ql
    if continuity.mission_pivot:
        packet.understanding_notes.append(
            f"Mission pivot — current turn isolated from prior session ({continuity.reason})."
        )
    packet.confidence_scores["continuity_confidence"] = continuity.continuity_confidence

    # Phase 2 semantic stabilization — intent only, no routes or aircraft
    from services.mission.mission_semantic_model import stabilize_mission_semantics

    stabilize_mission_semantics(
        context_text,
        profile,
        mission,
        packet,
        data_used=data_used if isinstance(data_used, dict) else None,
    )

    # Latent passenger norm inference:
    # on follow-ups, the operational posture still depends on the passenger load.
    pax_norm: Optional[int] = profile.passengers
    if pax_norm is None:
        m_pax = _PAX_NORM_RE.search(combined)
        if m_pax:
            try:
                pax_norm = int(m_pax.group(1))
            except (TypeError, ValueError):
                pax_norm = None

    packet.explicit_constraints = {
        "passengers": pax_norm,
        "routes": profile.route_labels(),
        "nonstop_required": profile.nonstop_required,
        "regions": list(profile.regions),
    }

    inf_mem = broker_memory if continuity.apply_posture_memory else None
    if continuity.mission_pivot and isinstance(broker_memory, dict):
        inf_mem = {
            k: broker_memory[k]
            for k in ("nonstop_preference",)
            if broker_memory.get(k)
        } or None

    inf = inferred or infer_mission_profile(ql, profile, broker_memory=inf_mem)
    packet.utilization_style = inf.utilization_style
    packet.confidence_scores["profile_inference"] = inf.confidence

    if inf.dispatch_priority:
        packet.dispatch_priority = "high"
    if inf.cost_sensitive:
        packet.operating_cost_priority = "high"
    if inf.nonstop_preference:
        packet.nonstop_priority = "high"
    if inf.airport_access_priority and packet.runway_complexity not in ("mountain",):
        packet.runway_complexity = "regional_access"
    if inf.cabin_priority_inferred:
        packet.comfort_priority = "high"

    # When current-turn extraction is generic, recover route evidence from the
    # prior chat history (for corridor posture + confidence gating only).
    route_labels_for_inference: Optional[List[str]] = None
    peak_nm_for_inference: Optional[float] = None
    if not profile.routes and hist_blob.strip() and continuity.apply_structural_memory:
        try:
            from services.mission.route_extractor import extract_routes

            inferred_routes = list(extract_routes(hist_blob))
            route_labels = [r.route.label() for r in inferred_routes if r.confidence >= 0.75]
            if route_labels:
                route_labels_for_inference = route_labels
                packet.explicit_constraints["routes"] = route_labels
                packet.confidence_scores["history_route_inference"] = 0.35
                try:
                    from services.consultant.route_feasibility import (
                        estimate_route_distance_nm,
                    )

                    peak_nm_for_inference = max(
                        float(estimate_route_distance_nm(lbl) or 0) for lbl in route_labels
                    )
                except Exception:
                    peak_nm_for_inference = None
        except Exception:
            route_labels_for_inference = None

    try:
        from services.mission.mission_corridor_routes import (
            detect_field_access_posture,
            enrich_profile_routes_from_corridor,
        )

        enrich_profile_routes_from_corridor(context_text, profile)
        if detect_field_access_posture(context_text):
            packet.inferred_constraints["industrial_airport_access"] = True
            packet.inferred_constraints["runway_over_cabin"] = True
            packet.runway_complexity = "regional_access"
            if "Domestic field-access executive band" not in (
                packet.fallback_operational_band or []
            ):
                packet.fallback_operational_band.append(
                    "Domestic field-access executive band"
                )
    except Exception:
        pass

    _collect_enterprise_signals(context_text, packet)
    _collect_traveler_posture(context_text, profile, packet)
    _collect_regional_hub_corridor(ql if profile.routes else context_text, profile, packet)
    _collect_regional_environment(ql, profile, packet)
    _collect_multi_role_signals(context_text, packet, profile)
    _collect_ownership_signals(context_text, packet)
    _collect_route_posture(
        profile,
        mission,
        packet,
        peak_nm_override=peak_nm_for_inference,
        route_labels_override=route_labels_for_inference,
    )
    peak_final = peak_nm_for_inference if peak_nm_for_inference is not None else _peak_stage_nm(profile)
    _collect_passenger_utilization_realism(context_text, profile, packet)
    _collect_international_operational_posture(ql, profile, packet, peak_nm=peak_final)
    _collect_priority_balance(ql, profile, packet)

    if isinstance(broker_memory, dict) and continuity.apply_posture_memory:
        if broker_memory.get("operational_philosophy"):
            packet.inferred_constraints["session_philosophy"] = broker_memory["operational_philosophy"]
            packet.confidence_scores["session_continuity"] = 0.25
        if broker_memory.get("nonstop_preference"):
            packet.nonstop_priority = "high"
        if continuity.apply_structural_memory and broker_memory.get("enterprise_scale"):
            packet.inferred_constraints["enterprise_scale"] = broker_memory["enterprise_scale"]
            packet.ownership_profile = str(
                broker_memory.get("ownership_profile") or packet.ownership_profile
            )

    # Confidence model: route evidence and explicit mission intent should dominate.
    # Avoid over-tight gating that would suppress normal, fully-specified missions.
    inf_conf = float(inferred.confidence) if inferred is not None else float(inf.confidence)
    pax_evidence = 1.0 if isinstance(packet.explicit_constraints.get("passengers"), int) else 0.0
    routes_evidence = 1.0 if bool(packet.explicit_constraints.get("routes")) else 0.0
    nonstop_evidence = 1.0 if packet.nonstop_priority == "high" else 0.0

    base = 0.18
    base += 0.42 * routes_evidence
    base += 0.12 * nonstop_evidence
    base += 0.10 * pax_evidence
    base += 0.10 * (1.0 if packet.dispatch_priority == "high" else 0.0)
    base += 0.08 * (1.0 if packet.runway_complexity == "high" else 0.0)
    base += 0.07 * (1.0 if packet.operational_environment else 0.0)
    base += 0.11 * max(0.0, min(1.0, inf_conf))
    packet.overall_confidence = min(1.0, base)

    packet.operational_synthesis = _synthesize_operational_prose(packet)
    rule_confidence = float(packet.overall_confidence)

    # Hybrid layer: LLM enriches latent inference; rules keep explicit facts + gating.
    llm_enabled = use_llm
    if llm_enabled is None:
        try:
            from services.mission.llm_mission_understanding import mission_understanding_llm_enabled

            llm_enabled = mission_understanding_llm_enabled()
        except Exception:
            llm_enabled = False

    if llm_enabled:
        try:
            from services.mission.llm_mission_understanding import infer_mission_understanding_llm
            from services.mission.mission_understanding_merge import merge_llm_understanding_into_packet

            llm_result = infer_mission_understanding_llm(
                query,
                profile,
                mission,
                history=history,
                rule_snapshot={
                    "corridor_type": packet.corridor_type,
                    "travel_pattern": packet.travel_pattern,
                    "ownership_profile": packet.ownership_profile,
                    "operational_synthesis": packet.operational_synthesis,
                    "inferred_constraints": dict(packet.inferred_constraints),
                },
            )
            packet = merge_llm_understanding_into_packet(
                packet,
                llm_result,
                rule_confidence=rule_confidence,
            )
            packet.confidence_scores["understanding_source"] = (
                1.0 if llm_result.ok else 0.0
            )
        except Exception:
            packet.understanding_notes.append("LLM understanding merge failed; rules-only packet retained.")

    # Recommendation gate: require strong enough mission understanding signal.
    # Do not require an extracted city-pair route; persistent context can imply
    # corridor posture even when a single-turn extract is incomplete.
    packet.recommend_aircraft = packet.overall_confidence >= _RECOMMEND_AIRCRAFT_MIN_CONF

    try:
        from services.mission.operational_synthesis import enrich_operational_synthesis

        mission_for_synth = mission
        packet.operational_synthesis = enrich_operational_synthesis(
            packet,
            mission_for_synth,
            profile,
            query=ql,
        )
    except Exception:
        pass

    return packet


def refresh_mission_understanding_gate(
    packet: MissionUnderstandingPacket,
    profile: MissionProfile,
    mission: MissionState,
    *,
    inferred_confidence: Optional[float] = None,
) -> MissionUnderstandingPacket:
    """
    Recompute the recommendation gate using *actual* merged mission evidence.

    This is needed because the understanding packet can be built on a generic follow-up
    query (no city pair in text). The orchestration layer may later merge persistent
    mission routes; we must refresh ``overall_confidence`` and ``recommend_aircraft``
    accordingly without losing conversation-aware posture signals.
    """
    inf_conf = (
        float(inferred_confidence)
        if inferred_confidence is not None
        else float(packet.confidence_scores.get("profile_inference") or 0.0)
    )

    pax_evidence = isinstance(packet.explicit_constraints.get("passengers"), int) and (
        packet.explicit_constraints.get("passengers") or 0
    ) > 0
    if not pax_evidence and isinstance(profile.passengers, int):
        pax_evidence = profile.passengers > 0

    routes_evidence = bool(profile.routes) or bool(getattr(mission, "routes", None))
    nonstop_evidence = (
        packet.nonstop_priority == "high"
        or bool(getattr(profile, "nonstop_required", False))
        or bool(getattr(mission, "nonstop_requirement", False))
    )

    base = 0.18
    base += 0.42 * (1.0 if routes_evidence else 0.0)
    base += 0.12 * (1.0 if nonstop_evidence else 0.0)
    base += 0.10 * (1.0 if pax_evidence else 0.0)
    base += 0.10 * (1.0 if packet.dispatch_priority == "high" else 0.0)
    base += 0.08 * (1.0 if packet.runway_complexity == "high" else 0.0)
    base += 0.07 * (1.0 if packet.operational_environment else 0.0)
    base += 0.11 * max(0.0, min(1.0, inf_conf))
    packet.overall_confidence = min(1.0, base)

    # Keep explicit constraints aligned to the merged mission snapshot.
    packet.explicit_constraints["routes"] = profile.route_labels()
    dist = profile.passenger_distribution
    planning = (
        dist.planning_load
        if dist is not None and dist.planning_load is not None
        else profile.passengers
    )
    packet.explicit_constraints["passengers"] = (
        int(planning) if isinstance(planning, int) else packet.explicit_constraints.get("passengers")
    )
    if dist is not None:
        packet.explicit_constraints["passenger_distribution"] = dist.to_dict()
        if dist.is_variable:
            packet.inferred_constraints["passenger_load_variable"] = True
            packet.inferred_constraints["planning_passenger_load"] = dist.planning_load

    packet.recommend_aircraft = packet.overall_confidence >= _RECOMMEND_AIRCRAFT_MIN_CONF
    return packet


def apply_understanding_to_profile(
    profile: MissionProfile,
    packet: MissionUnderstandingPacket,
    *,
    inferred: Optional[InferredMissionProfile] = None,
) -> MissionProfile:
    """Merge understanding into mission profile for feasibility and ranking."""
    if inferred:
        profile = apply_inference_to_profile(profile, inferred)

    pax = packet.explicit_constraints.get("passengers")
    if profile.passengers is None and isinstance(pax, int) and pax > 0:
        profile.passengers = pax

    if packet.runway_complexity == "high" or packet.runway_complexity == "mountain":
        profile.runway_priority = PriorityLevel.HIGH
        profile.short_field_priority = PriorityLevel.HIGH
        profile.mountain_airport_priority = profile.mountain_airports or bool(
            packet.inferred_constraints.get("island_ops")
        )
    elif packet.runway_complexity == "regional_access":
        profile.runway_priority = PriorityLevel.HIGH
        profile.short_field_priority = PriorityLevel.MEDIUM

    if packet.operating_cost_priority == "high" and profile.operating_cost_priority == PriorityLevel.NONE:
        profile.operating_cost_priority = PriorityLevel.HIGH

    if packet.comfort_priority == "high" and profile.cabin_priority == PriorityLevel.NONE:
        profile.cabin_priority = PriorityLevel.MEDIUM

    if packet.nonstop_priority == "high" or packet.inferred_constraints.get("fuel_stop_averse"):
        profile.nonstop_required = True

    routes_hint = packet.explicit_constraints.get("routes")
    if isinstance(routes_hint, list) and routes_hint and not profile.routes:
        from services.mission.models import Route

        for lbl in routes_hint:
            if not isinstance(lbl, str):
                continue
            r = Route.from_label(lbl)
            if r:
                profile.routes.append(r)

    if packet.inferred_constraints.get("corrosion_exposure") or packet.corridor_type == "caribbean_regional":
        profile.international_ops = True
        if "caribbean" not in [r.lower() for r in profile.regions]:
            profile.regions.append("Caribbean")

    if packet.travel_pattern in ("transatlantic_executive", "executive_shuttle"):
        profile.international_ops = True

    ceiling = packet.inferred_constraints.get("planning_band_ceiling")
    if isinstance(ceiling, str) and ceiling:
        profile.planning_band_ceiling = ceiling
    if packet.inferred_constraints.get("international_jet_floor"):
        profile.international_jet_floor = True
    if packet.inferred_constraints.get("balanced_cost_dispatch"):
        profile.balanced_cost_dispatch = True

    if (
        packet.utilization_style == UtilizationStyle.BOARD_TRANSPORT
        and not (profile.passenger_distribution and profile.passenger_distribution.is_variable)
        and (profile.passengers or 0) < 10
    ):
        profile.passengers = max(profile.passengers or 0, 12)
        if profile.passenger_distribution:
            profile.passenger_distribution.planning_load = profile.passengers

    return profile


def apply_understanding_to_mission_state(
    mission: MissionState,
    packet: MissionUnderstandingPacket,
) -> MissionState:
    routes_hint = packet.explicit_constraints.get("routes")
    if (not mission.routes) and isinstance(routes_hint, list) and routes_hint:
        # When mission routes are missing from turn-isolated extraction, we may
        # still have strong corridor posture evidence from prior chat turns.
        mission.routes = routes_hint

    if packet.nonstop_priority == "high":
        mission.nonstop_requirement = True
    # Multi-band missions keep mountain/industrial on the packet — not as a global rank category.
    from services.mission.mission_ranking_projection import is_segmented_mission

    segmented = is_segmented_mission(packet)
    if not segmented and (
        packet.runway_complexity in ("high", "mountain")
        or packet.inferred_constraints.get("mountain_ops")
    ):
        mission.mountain_airport_requirement = True
    elif not segmented and packet.runway_complexity == "regional_access":
        mission.runway_constraints = "short_field_preferred"
    if packet.operating_cost_priority == "high":
        mission.operating_cost_priority = "high"
    if packet.comfort_priority == "high":
        mission.cabin_priority = "high"
    if packet.inferred_constraints.get("westbound_winter_pressure"):
        mission.westbound = True
        if not mission.seasonal_constraints:
            mission.seasonal_constraints = "winter"
    if packet.inferred_constraints.get("international_jet_floor"):
        if (mission.cabin_priority or "").lower() not in ("high",):
            mission.cabin_priority = "medium"
    return mission


def format_understanding_first_advisory(
    mission: MissionState,
    packet: MissionUnderstandingPacket,
    *,
    recommendations: Optional[Sequence[Any]] = None,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
    route_certainty_degraded: bool = False,
) -> str:
    """
    Broker response when ranking is thin — single narrative authority, then options.
    """
    from services.mission.narrative_authority import (
        build_narrative_authority_payload,
        dedupe_advisory_body,
        dedupe_recommendation_models,
        render_narrative_authority,
    )
    from services.broker.broker_language import sanitize_broker_language

    payload = build_narrative_authority_payload(
        mission,
        packet,
        query=query,
        data_used=data_used,
        route_certainty_degraded=route_certainty_degraded,
    )
    lines = [render_narrative_authority(payload)]

    recs = dedupe_recommendation_models(
        [r for r in (recommendations or []) if not getattr(r, "avoid", False)]
    )

    if packet.recommend_aircraft and recs:
        lines.extend(["", "Aircraft Options:", ""])
        for r in recs[:3]:
            model = getattr(r, "model", str(r))
            fit = getattr(r, "fit", "") or "directional fit"
            lines.append(f"* {model} — {fit} (verify payload and season operationally).")
    elif payload.segments:
        lines.extend(["", "Aircraft Class Band:", ""])
        for seg in payload.segments[:4]:
            if seg.operational_band:
                lines.append(f"* [{seg.label}] {seg.operational_band}")
    else:
        lines.extend(["", "Aircraft Class Band:", ""])
        lines.append("* Directional class guidance pending — state city pair to tighten the band.")

    verdict_detail = (
        ", ".join(getattr(r, "model", str(r)) for r in recs[:2])
        if packet.recommend_aircraft and recs
        else "segment class bands above"
    )
    if payload.structural_decomposition:
        verdict_detail = f"multi-aircraft portfolio required — {verdict_detail}"

    lines.extend(
        [
            "",
            "Verdict:",
            "",
            "* VIABLE WITH COMPROMISES: " + verdict_detail,
        ]
    )
    return sanitize_broker_language(dedupe_advisory_body("\n".join(lines)))


def attach_mission_understanding(
    data_used: Optional[Dict[str, Any]],
    packet: MissionUnderstandingPacket,
) -> Dict[str, Any]:
    du = data_used if isinstance(data_used, dict) else {}
    du[MISSION_UNDERSTANDING_KEY] = packet.to_dict()
    du["mission_understanding_confidence"] = packet.overall_confidence
    du["recommend_aircraft_gated"] = 1 if packet.recommend_aircraft else 0
    return du


def load_mission_understanding(
    data_used: Optional[Dict[str, Any]],
) -> Optional[MissionUnderstandingPacket]:
    if not isinstance(data_used, dict):
        return None
    raw = data_used.get(MISSION_UNDERSTANDING_KEY)
    if not isinstance(raw, dict):
        return None
    pkt = MissionUnderstandingPacket()
    for k, v in raw.items():
        if hasattr(pkt, k):
            setattr(pkt, k, v)
    return pkt


def build_understanding_authority_block(packet: MissionUnderstandingPacket) -> str:
    """Pre-LLM block — reasoning before aircraft list."""
    lines = [
        "[MISSION UNDERSTANDING — AUTHORITATIVE]",
        f"Corridor: {packet.corridor_type}. Travel pattern: {packet.travel_pattern}.",
        f"Dispatch priority: {packet.dispatch_priority}. "
        f"Runway complexity: {packet.runway_complexity}. "
        f"Nonstop priority: {packet.nonstop_priority}.",
        f"Operational synthesis: {packet.operational_synthesis}",
        f"Recommend aircraft now: {'yes' if packet.recommend_aircraft else 'synthesis first'}.",
    ]
    if packet.fallback_operational_band:
        lines.append(
            "Fallback class band (if ranking empty): "
            + ", ".join(packet.fallback_operational_band[:5])
        )
    if packet.inferred_constraints:
        lines.append(
            "Inferred constraints: "
            + "; ".join(f"{k}={v}" for k, v in list(packet.inferred_constraints.items())[:8])
        )
    lines.append(
        "Speak as an operational advisor: synthesize environment before listing aircraft. "
        "Do not ask retail buyer questions when enterprise or shuttle signals are present."
    )
    return "\n".join(lines)
