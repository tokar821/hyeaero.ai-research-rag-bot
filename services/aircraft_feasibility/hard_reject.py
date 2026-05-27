"""Hard rejection rules — no soft penalties for impossible aircraft."""

from __future__ import annotations

from typing import Any, List, Mapping

from services.aircraft_feasibility.constants import (
    MIN_DISPATCH_MARGIN_NM,
    MIN_DISPATCH_MARGIN_ULR_NM,
    RUNWAY_LIMIT_DEFAULT_FT,
    RUNWAY_LIMIT_INTERNATIONAL_FT,
    RUNWAY_LIMIT_MOUNTAIN_FT,
    RUNWAY_LIMIT_SHORT_FIELD_FT,
    TRANSATLANTIC_NONSTOP_MIN_PRACTICAL_NM,
    TRANSATLANTIC_STAGE_NM,
    TRANSPACIFIC_NONSTOP_MIN_PRACTICAL_NM,
    TRANSPACIFIC_STAGE_NM,
    TRANSPACIFIC_WINTER_WESTBOUND_MIN_PRACTICAL_NM,
)
from services.aircraft_feasibility.mission_context import FeasibilityMissionContext
from services.aircraft_feasibility.payload_range import PayloadAdjustedRange
from services.aircraft_feasibility.range_margin import MissionRangeRequirement
from services.aircraft_feasibility.route_realism_validator import is_ultra_long_corridor_mission

_LIGHT_JET_CATEGORIES = frozenset({"light", "turboprop"})


def _runway_limit_ft(mission: FeasibilityMissionContext) -> float:
    if mission.mountain_airports or mission.hot_high_ops:
        return RUNWAY_LIMIT_MOUNTAIN_FT
    if mission.short_runway_ops or mission.runway_priority_high:
        return RUNWAY_LIMIT_SHORT_FIELD_FT
    if mission.transatlantic or mission.transpacific or mission.international_ops:
        return RUNWAY_LIMIT_INTERNATIONAL_FT
    return RUNWAY_LIMIT_DEFAULT_FT


def apply_hard_reject_rules(
    *,
    model: str,
    aircraft: Mapping[str, Any],
    mission: FeasibilityMissionContext,
    requirement: MissionRangeRequirement,
    adjusted: PayloadAdjustedRange,
    runway_penalty_nm: float,
) -> List[str]:
    """
    Return rejection reasons. Non-empty list means aircraft is eliminated — not ranked.
    """
    reasons: List[str] = []
    cat = str(aircraft.get("category") or "")
    practical = float(aircraft.get("practical_nm") or 0.0)
    stage = mission.stage_distance_nm
    available = adjusted.available_nm
    required = requirement.total_required_nm
    margin = available - required if required > 0 else available

    # --- Unknown / zero performance ---
    if practical <= 0:
        reasons.append(f"Unknown or missing performance data for {model}.")
        return reasons

    if required > 0 and margin < 0 and not reasons:
        reasons.append(
            f"Practical available ~{int(available)} nm < mission required ~{int(required)} nm "
            f"(NBAA IFR reserves, winter westbound, passenger payload, baggage, hot/high, "
            f"and runway penalties applied; brochure range not used)."
        )

    # --- RULE: ultra-long corridors — no light jets on nonstop NYC–Dubai, LA–London, SFO–Tokyo ---
    corridor, corridor_id = is_ultra_long_corridor_mission(mission)
    if (
        corridor
        and mission.nonstop_required
        and not mission.stop_required
        and cat in _LIGHT_JET_CATEGORIES
    ):
        reasons.append(
            f"Light-jet platform ({model}) hard-rejected for nonstop {corridor_id or 'ultra-long'} "
            f"corridor (~{int(stage)} nm) — tech stop required or move up to super-mid/ULR."
        )

    min_margin = MIN_DISPATCH_MARGIN_ULR_NM if cat == "ultra-long" else MIN_DISPATCH_MARGIN_NM
    if required > 0 and 0 <= margin < min_margin and not reasons:
        reasons.append(
            f"Dispatch margin ~{int(margin)} nm below minimum {int(min_margin)} nm "
            f"for reliable nonstop operations."
        )

    # --- RULE: passenger envelope ---
    pax_max = int(aircraft.get("pax_max_long_range") or aircraft.get("pax_typical") or 8)
    if mission.passengers > pax_max:
        reasons.append(
            f"Passenger count {mission.passengers} exceeds long-range envelope ({pax_max}) for {model}."
        )

    # --- RULE: runway length (hard) ---
    runway_ft = float(aircraft.get("runway_ft") or 9999)
    limit_ft = _runway_limit_ft(mission)
    if runway_ft > limit_ft:
        reasons.append(
            f"Runway requirement ~{int(runway_ft)} ft exceeds mission airport limit ~{int(limit_ft)} ft."
        )

    # --- RULE: short-field / runway priority eliminates heavy platforms ---
    if mission.short_runway_ops or mission.runway_priority_high:
        short_score = float(aircraft.get("short_field_score") or 0.5)
        if cat in ("ultra-long", "large") or short_score < 0.55:
            reasons.append(
                f"Short-field / runway-flex mission incompatible with {cat} platform ({model})."
            )

    # --- RULE: mountain / hot-high field performance ---
    if mission.mountain_airports or mission.hot_high_ops:
        hot_high = float(aircraft.get("hot_high_score") or 0.5)
        if hot_high < 0.60:
            reasons.append(
                f"Mountain / hot-and-high performance inadequate for {model} "
                f"(hot_high_score {hot_high:.2f} < 0.60)."
            )

    # --- RULE: transatlantic nonstop category gate ---
    if mission.transatlantic and mission.nonstop_required and stage >= TRANSATLANTIC_STAGE_NM:
        if cat == "super-midsize" or practical < TRANSATLANTIC_NONSTOP_MIN_PRACTICAL_NM:
            reasons.append(
                f"Transatlantic nonstop (~{int(stage)} nm) requires large-cabin / ULR practical range "
                f"(>= {int(TRANSATLANTIC_NONSTOP_MIN_PRACTICAL_NM)} nm); {model} is {cat} "
                f"(~{int(practical)} nm practical)."
            )

    # --- RULE: transpacific nonstop category gate ---
    if mission.transpacific and mission.nonstop_required and stage >= TRANSPACIFIC_STAGE_NM:
        min_practical = TRANSPACIFIC_NONSTOP_MIN_PRACTICAL_NM
        if mission.winter_westbound_transpacific:
            min_practical = TRANSPACIFIC_WINTER_WESTBOUND_MIN_PRACTICAL_NM
        if cat != "ultra-long" or practical < min_practical:
            reasons.append(
                f"Transpacific nonstop (~{int(stage)} nm) requires ultra-long-range practical capability "
                f"(>= {int(min_practical)} nm); {model} is {cat} (~{int(practical)} nm practical)."
            )

    # --- RULE: super-midsize hard block on transpacific (even without explicit nonstop) ---
    if (
        mission.transpacific
        and stage >= TRANSPACIFIC_STAGE_NM
        and cat == "super-midsize"
    ):
        reasons.append(
            f"Super-midsize ({model}) hard-rejected for transpacific stage (~{int(stage)} nm)."
        )

    # --- RULE: regional overbuy (wrong tool for stage length) ---
    if stage > 0 and stage < 1500 and cat == "large":
        reasons.append(
            f"Large-cabin platform ({model}) not operationally justified for ~{int(stage)} nm stage."
        )
    if stage > 0 and stage < 2000 and cat == "ultra-long":
        reasons.append(
            f"Ultra-long-range platform ({model}) not operationally justified for ~{int(stage)} nm stage."
        )

    # --- RULE: winter westbound transpacific explicit fail when margin negative ---
    if mission.winter_westbound_transpacific and required > 0 and margin < 0:
        if not any("Transpacific" in r or "Practical available" in r for r in reasons):
            reasons.append(
                "Winter westbound transpacific: insufficient range margin on conservative assumptions."
            )

    # Runway penalty as extra nm burden — if large, reinforce rejection
    if runway_penalty_nm >= 400 and not any("Runway" in r for r in reasons):
        reasons.append(
            f"Hot/high runway penalty (~{int(runway_penalty_nm)} nm equivalent) exceeds {model} margin."
        )

    return reasons
