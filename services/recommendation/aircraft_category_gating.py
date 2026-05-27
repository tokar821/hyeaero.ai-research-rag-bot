"""
Aircraft category gating — determine mission class, restrict candidates, score in-band only.

Mission bands (operational, not marketing brochure classes):

  LIGHT JET        — short domestic, ~4–7 pax, < 2,000 nm practical stage
  SUPER MIDSIZE    — coast-to-coast / transcon, Europe-edge, ~6–9 pax
  LARGE CABIN      — consistent transatlantic, ~10–14 pax
  ULTRA LONG RANGE — westbound Pacific, Dubai, Tokyo nonstop; winter reserve critical

Turboprops never enter ultra-long-range (or large-cabin / transatlantic) pools.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set

from services.consultant.mission_state import MissionState
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES
from services.mission.models import MissionProfile
from services.recommendation.mission_ranker import MissionCategory, mission_max_leg_nm

logger = logging.getLogger(__name__)

# Practical stage thresholds (nm) — aligned with operational planning, not brochure range
LIGHT_JET_MAX_LEG_NM = 2000.0
SUPER_MID_MIN_LEG_NM = 2000.0
SUPER_MID_MAX_LEG_NM = 4799.0
LARGE_CABIN_MIN_LEG_NM = 2800.0
ULR_MIN_LEG_NM = 4800.0
TRANSPACIFIC_ULR_MIN_LEG_NM = 4200.0

_CATALOG_LIGHT = frozenset({"light"})
_CATALOG_SUPER_MID = frozenset({"super-midsize"})
_CATALOG_LARGE = frozenset({"large"})
_CATALOG_ULR = frozenset({"ultra-long"})
_CATALOG_MOUNTAIN_FIELD = frozenset({"super-midsize", "light", "turboprop"})


class GatedMissionCategory(str, Enum):
    LIGHT_JET = "light_jet"
    SUPER_MIDSIZE = "super_midsize"
    LARGE_CABIN = "large_cabin"
    ULTRA_LONG_RANGE = "ultra_long_range"


_ALLOWED_CATALOG_BY_GATE: Dict[GatedMissionCategory, frozenset[str]] = {
    GatedMissionCategory.LIGHT_JET: _CATALOG_LIGHT,
    GatedMissionCategory.SUPER_MIDSIZE: _CATALOG_SUPER_MID,
    GatedMissionCategory.LARGE_CABIN: _CATALOG_LARGE,
    GatedMissionCategory.ULTRA_LONG_RANGE: _CATALOG_ULR,
}

_LEGACY_MAP: Dict[GatedMissionCategory, MissionCategory] = {
    GatedMissionCategory.LIGHT_JET: MissionCategory.REGIONAL_UTILITY,
    GatedMissionCategory.SUPER_MIDSIZE: MissionCategory.COAST_TO_COAST,
    GatedMissionCategory.LARGE_CABIN: MissionCategory.TRANSATLANTIC_EXECUTIVE,
    GatedMissionCategory.ULTRA_LONG_RANGE: MissionCategory.ULTRA_LONG_RANGE,
}


def _route_blob(mission: MissionState) -> str:
    return " ".join((mission.routes or [])).lower()


def _is_transpacific_route(blob: str, max_leg: float) -> bool:
    if re.search(r"\btranspacific\b", blob):
        return True
    if max_leg < 3500:
        return False
    return bool(
        max_leg >= TRANSPACIFIC_ULR_MIN_LEG_NM
        and re.search(r"\b(?:tokyo|seoul|beijing|hong\s+kong|singapore|sydney|dubai)\b", blob)
        and re.search(
            r"\b(?:san\s+francisco|los\s+angeles|seattle|new\s+york|nyc|honolulu|west\s+coast)\b",
            blob,
        )
    )


def _is_ultra_long_corridor(blob: str) -> bool:
    return bool(
        re.search(r"\b(?:new\s+york|nyc|jfk)\b", blob) and re.search(r"\bdubai\b", blob)
        or re.search(r"\b(?:los\s+angeles|la\b|lax)\b", blob) and re.search(r"\blondon\b", blob)
        or re.search(r"\b(?:san\s+francisco|sfo)\b", blob) and re.search(r"\btokyo\b", blob)
    )


_EAST_TRANSCON_RE = re.compile(
    r"\b(?:new\s+york|nyc|boston|washington|philadelphia|chicago|teterboro|jfk|teb)\b",
    re.I,
)
_WEST_TRANSCON_RE = re.compile(
    r"\b(?:los\s+angeles|san\s+francisco|seattle|lax|sfo|oak|san\s+diego)\b",
    re.I,
)


def _is_short_regional_leg(blob: str, max_leg: float, *, passengers: int = 6) -> bool:
    """Caribbean / island short legs — light-jet pool only for small executive loads."""
    if passengers >= 8:
        return False
    if max_leg <= 0 or max_leg >= LIGHT_JET_MAX_LEG_NM:
        return False
    if max_leg < 1200:
        return True
    return bool(
        re.search(
            r"\b(?:caribbean|nassau|bahamas|turks|caicos|short\s+runway|regional)\b",
            blob,
            re.I,
        )
    )


def _is_us_transcon(blob: str, max_leg: float) -> bool:
    if re.search(r"\b(?:coast|transcon|transcontinental)\b", blob, re.I):
        return True
    if max_leg < SUPER_MID_MIN_LEG_NM:
        return False
    return bool(_EAST_TRANSCON_RE.search(blob) and _WEST_TRANSCON_RE.search(blob))


def _is_transatlantic_route(blob: str, max_leg: float) -> bool:
    if max_leg >= LARGE_CABIN_MIN_LEG_NM and re.search(
        r"\b(?:london|paris|geneva|europe|dublin|zurich|berlin|moscow|frankfurt|munich)\b",
        blob,
    ):
        return True
    return bool(re.search(r"\btransatlantic\b", blob) and max_leg >= 2400)


def _europe_cities_in_blob(blob: str) -> bool:
    return bool(
        re.search(
            r"\b(?:london|paris|geneva|berlin|moscow|frankfurt|munich|zurich|europe)\b",
            blob,
            re.I,
        )
    )


def _cap_gate_from_profile(
    gate: "MissionCategoryGateResult",
    mission_profile: Optional[MissionProfile],
) -> "MissionCategoryGateResult":
    """Apply mission-understanding planning band ceiling."""
    if mission_profile is None:
        return gate
    ceiling = (mission_profile.planning_band_ceiling or "").lower().replace("-", "_")
    if not ceiling:
        return gate
    order = {
        GatedMissionCategory.LIGHT_JET: 0,
        GatedMissionCategory.SUPER_MIDSIZE: 1,
        GatedMissionCategory.LARGE_CABIN: 2,
        GatedMissionCategory.ULTRA_LONG_RANGE: 3,
    }
    cap_map = {
        "super_midsize": GatedMissionCategory.SUPER_MIDSIZE,
        "super_midsize_band": GatedMissionCategory.SUPER_MIDSIZE,
        "large_cabin": GatedMissionCategory.LARGE_CABIN,
        "ultra_long_range": GatedMissionCategory.ULTRA_LONG_RANGE,
    }
    cap = cap_map.get(ceiling)
    if cap is None:
        return gate
    if order.get(gate.category, 0) > order.get(cap, 99):
        gate.category = cap
        gate.reasons.append(
            f"Mission understanding ceiling — capped at {cap.value.replace('_', ' ')} "
            f"from passenger utilization / frequency realism."
        )
    return gate


def determine_gated_mission_category(
    mission: MissionState,
    *,
    mission_profile: Optional[MissionProfile] = None,
) -> "MissionCategoryGateResult":
    """
    Classify mission into exactly one gated band before recommendations.

    Uses stage length, passengers, and route semantics (Pacific, Dubai, Europe).
    """
    max_leg = mission_max_leg_nm(mission)
    pax = int(mission.passenger_count or 6)
    blob = _route_blob(mission)
    reasons: List[str] = []

    profile = mission_profile
    intl_floor = bool(
        profile and (profile.international_jet_floor or profile.balanced_cost_dispatch)
    )

    # Mountain / hot-high — super-mid field-performance band (not light-only pool)
    if mission.mountain_airport_requirement or re.search(
        r"\b(?:aspen|telluride|jackson|sun\s+valley|mountain\s+airport|hot/high)\b",
        blob,
        re.I,
    ):
        reasons.append(
            "Mountain or hot/high airport — field-performance band (STOL / light / super-mid)."
        )
        return MissionCategoryGateResult(
            category=GatedMissionCategory.SUPER_MIDSIZE,
            max_leg_nm=max_leg,
            passengers=pax,
            reasons=reasons,
            allowed_catalog_override=_CATALOG_MOUNTAIN_FIELD,
        )

    transpacific = _is_transpacific_route(blob, max_leg)
    ulr_corridor = _is_ultra_long_corridor(blob)
    transatlantic = _is_transatlantic_route(blob, max_leg)
    westbound_pacific = bool(
        mission.westbound
        or re.search(r"\bwestbound\b", blob)
        or (transpacific and re.search(r"\b(?:francisco|angeles|seattle)\b", blob))
    )

    # --- ULTRA LONG RANGE ---
    if (
        max_leg >= ULR_MIN_LEG_NM
        or (transpacific and max_leg >= TRANSPACIFIC_ULR_MIN_LEG_NM)
        or (ulr_corridor and mission.nonstop_requirement)
        or (westbound_pacific and transpacific and max_leg >= 4000)
    ):
        reasons.append(
            "Ultra-long-range band: westbound Pacific, Dubai, Tokyo-class stage, or "
            f"≥ {int(ULR_MIN_LEG_NM)} nm with winter reserve critical."
        )
        return MissionCategoryGateResult(
            category=GatedMissionCategory.ULTRA_LONG_RANGE,
            max_leg_nm=max_leg,
            passengers=pax,
            reasons=reasons,
        )

    # --- LARGE CABIN ---
    if transatlantic or (max_leg >= LARGE_CABIN_MIN_LEG_NM and max_leg < ULR_MIN_LEG_NM):
        if intl_floor and pax <= 8:
            reasons.append(
                "International jet floor with modest passenger load — super-midsize planning band "
                "(not light jet) despite cost sensitivity."
            )
            result = MissionCategoryGateResult(
                category=GatedMissionCategory.SUPER_MIDSIZE,
                max_leg_nm=max_leg,
                passengers=pax,
                reasons=reasons,
            )
            return _cap_gate_from_profile(result, profile)
        if pax >= 10 or max_leg >= 3000:
            reasons.append(
                "Large-cabin band: consistent transatlantic or "
                f"≥ {int(LARGE_CABIN_MIN_LEG_NM)} nm with executive load."
            )
            return MissionCategoryGateResult(
                category=GatedMissionCategory.LARGE_CABIN,
                max_leg_nm=max_leg,
                passengers=pax,
                reasons=reasons,
            )

    # Passenger-driven bump to large cabin on long legs
    if pax >= 12 and max_leg >= 2400:
        reasons.append("Passenger load ≥ 12 — large-cabin band minimum.")
        return MissionCategoryGateResult(
            category=GatedMissionCategory.LARGE_CABIN,
            max_leg_nm=max_leg,
            passengers=pax,
            reasons=reasons,
        )

    if _is_us_transcon(blob, max_leg):
        reasons.append("Super-midsize band: U.S. coast-to-coast / transcontinental city pair.")
        return MissionCategoryGateResult(
            category=GatedMissionCategory.SUPER_MIDSIZE,
            max_leg_nm=max_leg,
            passengers=pax,
            reasons=reasons,
        )

    # Passenger band — light jets are typical 4–7 pax; 8+ needs super-midsize or larger
    if (
        pax > 7
        and max_leg < ULR_MIN_LEG_NM
        and not transatlantic
        and not _is_short_regional_leg(blob, max_leg, passengers=pax)
    ):
        if pax >= 10 and max_leg >= LARGE_CABIN_MIN_LEG_NM:
            reasons.append(f"Passenger load {pax} — large-cabin band minimum.")
            return MissionCategoryGateResult(
                category=GatedMissionCategory.LARGE_CABIN,
                max_leg_nm=max_leg,
                passengers=pax,
                reasons=reasons,
            )
        reasons.append(
            f"Passenger load {pax} exceeds light-jet band (typical 4–7 pax) — super-midsize minimum."
        )
        return MissionCategoryGateResult(
            category=GatedMissionCategory.SUPER_MIDSIZE,
            max_leg_nm=max_leg,
            passengers=pax,
            reasons=reasons,
        )

    # --- LIGHT JET (before super-mid so sub-2,000 nm domestic stays in light pool) ---
    if max_leg > 0 and max_leg < LIGHT_JET_MAX_LEG_NM:
        reasons.append(
            f"Light-jet band: short domestic stage < {int(LIGHT_JET_MAX_LEG_NM)} nm, "
            "typical 4–7 pax missions."
        )
        return MissionCategoryGateResult(
            category=GatedMissionCategory.LIGHT_JET,
            max_leg_nm=max_leg,
            passengers=pax,
            reasons=reasons,
        )

    # --- SUPER MIDSIZE ---
    if max_leg >= SUPER_MID_MIN_LEG_NM and max_leg <= SUPER_MID_MAX_LEG_NM:
        if pax <= 9 or max_leg < LARGE_CABIN_MIN_LEG_NM:
            reasons.append(
                "Super-midsize band: coast-to-coast / transcon or Europe-edge "
                f"({int(SUPER_MID_MIN_LEG_NM)}–{int(SUPER_MID_MAX_LEG_NM)} nm)."
            )
            return MissionCategoryGateResult(
                category=GatedMissionCategory.SUPER_MIDSIZE,
                max_leg_nm=max_leg,
                passengers=pax,
                reasons=reasons,
            )
    if max_leg >= 1500 and re.search(r"\b(?:coast|transcon|transcontinental)\b", blob):
        reasons.append("Super-midsize band: explicit coast-to-coast / transcon mission.")
        return MissionCategoryGateResult(
            category=GatedMissionCategory.SUPER_MIDSIZE,
            max_leg_nm=max_leg,
            passengers=pax,
            reasons=reasons,
        )

    if max_leg == 0:
        if intl_floor or _europe_cities_in_blob(blob):
            reasons.append(
                "Europe / international references without resolved stage length — "
                "super-midsize minimum (light jet not credible)."
            )
            result = MissionCategoryGateResult(
                category=GatedMissionCategory.SUPER_MIDSIZE,
                max_leg_nm=max_leg,
                passengers=pax,
                reasons=reasons,
            )
            return _cap_gate_from_profile(result, profile)
        if pax <= 7:
            reasons.append("No route on file — default light-jet band from passenger count.")
            return MissionCategoryGateResult(
                category=GatedMissionCategory.LIGHT_JET,
                max_leg_nm=max_leg,
                passengers=pax,
                reasons=reasons,
            )
        reasons.append("No route on file — default super-midsize advisory band.")
        return MissionCategoryGateResult(
            category=GatedMissionCategory.SUPER_MIDSIZE,
            max_leg_nm=max_leg,
            passengers=pax,
            reasons=reasons,
        )

    # Fallback by length
    if max_leg >= LARGE_CABIN_MIN_LEG_NM:
        reasons.append("Stage length fallback — large-cabin band.")
        return MissionCategoryGateResult(
            category=GatedMissionCategory.LARGE_CABIN,
            max_leg_nm=max_leg,
            passengers=pax,
            reasons=reasons,
        )
    if max_leg >= SUPER_MID_MIN_LEG_NM:
        reasons.append("Stage length fallback — super-midsize band.")
        return MissionCategoryGateResult(
            category=GatedMissionCategory.SUPER_MIDSIZE,
            max_leg_nm=max_leg,
            passengers=pax,
            reasons=reasons,
        )

    reasons.append("Stage length fallback — light-jet band.")
    return MissionCategoryGateResult(
        category=GatedMissionCategory.LIGHT_JET,
        max_leg_nm=max_leg,
        passengers=pax,
        reasons=reasons,
    )


@dataclass
class MissionCategoryGateResult:
    """Mission band + restricted catalog categories for scoring."""

    category: GatedMissionCategory
    max_leg_nm: float = 0.0
    passengers: int = 6
    reasons: List[str] = field(default_factory=list)
    candidate_models: List[str] = field(default_factory=list)
    excluded_models: List[str] = field(default_factory=list)
    exclusion_log: List[Dict[str, str]] = field(default_factory=list)
    allowed_catalog_override: Optional[frozenset[str]] = None

    @property
    def legacy_category(self) -> MissionCategory:
        return _LEGACY_MAP[self.category]

    @property
    def allowed_catalog_categories(self) -> frozenset[str]:
        if self.allowed_catalog_override is not None:
            return self.allowed_catalog_override
        return _ALLOWED_CATALOG_BY_GATE[self.category]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gated_category": self.category.value,
            "legacy_category": self.legacy_category.value,
            "max_leg_nm": round(self.max_leg_nm, 1),
            "passengers": self.passengers,
            "allowed_catalog_categories": sorted(self.allowed_catalog_categories),
            "candidate_count": len(self.candidate_models),
            "excluded_count": len(self.excluded_models),
            "reasons": list(self.reasons),
            "exclusion_log": list(self.exclusion_log),
        }


def normalize_catalog_category(raw: str) -> str:
    return (raw or "").strip().lower().replace("_", "-")


def aircraft_catalog_category(model: str) -> str:
    return normalize_catalog_category((AIRCRAFT_PROFILES.get(model) or {}).get("category", ""))


def model_allowed_in_gate(model: str, gate: MissionCategoryGateResult) -> Optional[str]:
    """
    Return None if model may be scored; else exclusion reason.

    Turboprops are never mixed into ULR, large-cabin, or transatlantic bands.
    """
    spec = AIRCRAFT_PROFILES.get(model)
    if not spec:
        return f"Unknown aircraft: {model}"

    cat = normalize_catalog_category(str(spec.get("category") or ""))
    practical = float(spec.get("practical_nm") or 0)

    if cat == "turboprop":
        if gate.category == GatedMissionCategory.ULTRA_LONG_RANGE:
            return "Turboprops are never scored on ultra-long-range missions."
        if gate.category == GatedMissionCategory.LARGE_CABIN:
            return "Turboprops are never scored on transatlantic / large-cabin missions."
        if gate.category == GatedMissionCategory.SUPER_MIDSIZE and gate.max_leg_nm >= 2200:
            return "Turboprops excluded from long super-midsize / transcon missions."
        if gate.max_leg_nm >= LIGHT_JET_MAX_LEG_NM:
            return "Turboprop excluded — stage length exceeds light-jet band."

    allowed = gate.allowed_catalog_categories
    if cat not in allowed:
        return (
            f"{model} ({cat or 'unknown'}) outside {gate.category.value} pool "
            f"(allowed: {', '.join(sorted(allowed))})."
        )

    if gate.category == GatedMissionCategory.LIGHT_JET and practical >= LIGHT_JET_MAX_LEG_NM:
        return (
            f"{model} practical range ~{int(practical)} nm exceeds light-jet "
            f"< {int(LIGHT_JET_MAX_LEG_NM)} nm ceiling."
        )

    if (
        gate.category == GatedMissionCategory.SUPER_MIDSIZE
        and practical < 1700
        and gate.allowed_catalog_override != _CATALOG_MOUNTAIN_FIELD
    ):
        return f"{model} practical range too short for super-midsize mission band."

    if gate.category == GatedMissionCategory.LARGE_CABIN and practical < 3600:
        return f"{model} practical range too short for consistent transatlantic large-cabin band."

    if gate.category == GatedMissionCategory.ULTRA_LONG_RANGE and practical < 5200:
        return f"{model} practical range below ultra-long-range floor (~5200 nm)."

    return None


def filter_candidates_by_category_gate(
    candidates: Sequence[str],
    gate: MissionCategoryGateResult,
) -> MissionCategoryGateResult:
    """Restrict pool — only aircraft within the gated mission band proceed to scoring."""
    survivors: List[str] = []
    excluded: List[str] = []
    log: List[Dict[str, str]] = []

    for model in candidates:
        reason = model_allowed_in_gate(model, gate)
        if reason:
            excluded.append(model)
            log.append(
                {
                    "aircraft_name": model,
                    "reason": reason,
                    "mission_constraint_failed": "category_gate",
                    "gated_category": gate.category.value,
                }
            )
            logger.info("CATEGORY_GATE_EXCLUDE: model=%s reason=%s", model, reason)
        else:
            survivors.append(model)

    gate.candidate_models = survivors
    gate.excluded_models = excluded
    gate.exclusion_log = log
    return gate


def apply_mission_category_gating(
    mission: MissionState,
    candidates: Sequence[str],
    *,
    mission_profile: Optional[MissionProfile] = None,
) -> MissionCategoryGateResult:
    """
    Full gating pass: classify mission → restrict candidates → return audit result.

    Call before feasibility scoring / ranking so only in-band aircraft are scored.
    """
    gate = determine_gated_mission_category(mission, mission_profile=mission_profile)
    gate = _cap_gate_from_profile(gate, mission_profile)
    return filter_candidates_by_category_gate(list(candidates), gate)


def gated_to_legacy_category(gated: GatedMissionCategory) -> MissionCategory:
    return _LEGACY_MAP[gated]


def models_in_catalog_band(gated: GatedMissionCategory) -> List[str]:
    """All catalog models whose category matches the gated band (pre-feasibility pool)."""
    allowed = _ALLOWED_CATALOG_BY_GATE[gated]
    return [
        model
        for model in AIRCRAFT_PROFILES
        if aircraft_catalog_category(model) in allowed
        and aircraft_catalog_category(model) != "turboprop"
    ]


def category_exclusion_feasibility_results(
    gate: MissionCategoryGateResult,
) -> Dict[str, Any]:
    """
    FeasibilityResult entries for models removed by category gating (audit / eliminated_models).

    Out-of-band aircraft are marked infeasible before range scoring so callers retain a full map.
    """
    from services.mission.feasibility_engine import FeasibilityResult

    out: Dict[str, FeasibilityResult] = {}
    for entry in gate.exclusion_log:
        model = entry.get("aircraft_name") or ""
        if not model:
            continue
        reason = entry.get("reason") or "Outside mission category pool."
        out[model] = FeasibilityResult(
            feasible=False,
            elimination_reasons=[reason],
            operational_risk_level="eliminated",
            notes=[reason],
        )
    return out
