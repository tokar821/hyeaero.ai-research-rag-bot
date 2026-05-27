"""Payload-adjusted practical range — available nm after conservative deductions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from services.aircraft_feasibility.constants import (
    BAGGAGE_NM_PENALTY,
    HOT_HIGH_AVAILABLE_PENALTY_NM,
    MOUNTAIN_AVAILABLE_PENALTY_NM,
    PAX_NM_PENALTY_CAP,
    PAX_NM_PENALTY_PER_SEAT,
    WESTBOUND_REQUIRED_FACTOR,
    WINTER_AVAILABLE_DEDUCTION_FACTOR,
)
from services.aircraft_feasibility.mission_context import FeasibilityMissionContext


@dataclass(frozen=True)
class PayloadAdjustedRange:
    """Operational range available after payload / environment deductions."""

    practical_baseline_nm: float
    payload_penalty_nm: float
    winter_penalty_nm: float
    mountain_penalty_nm: float
    hot_high_penalty_nm: float
    available_nm: float


def compute_payload_adjusted_range(
    aircraft: Mapping[str, Any],
    mission: FeasibilityMissionContext,
) -> PayloadAdjustedRange:
    """
    Available range (nm) from ``practical_nm`` baseline — never brochure.

    Deductions are conservative and applied before margin comparison.
    """
    baseline = float(aircraft.get("practical_nm") or 0.0)
    if baseline <= 0:
        return PayloadAdjustedRange(
            practical_baseline_nm=0.0,
            payload_penalty_nm=0.0,
            winter_penalty_nm=0.0,
            mountain_penalty_nm=0.0,
            hot_high_penalty_nm=0.0,
            available_nm=0.0,
        )

    typical = int(aircraft.get("pax_typical") or 6)
    pax = mission.passengers

    payload_pen = 0.0
    if pax > typical:
        payload_pen = min(PAX_NM_PENALTY_CAP, (pax - typical) * PAX_NM_PENALTY_PER_SEAT)
    if mission.baggage_high:
        payload_pen += BAGGAGE_NM_PENALTY

    winter_pen = 0.0
    cat = str(aircraft.get("category") or "")
    if mission.westbound_sensitive and mission.winter_ops:
        # ULR platforms carry winter westbound margin in operational planning — lighter deduction
        winter_pen = baseline * (0.04 if cat == "ultra-long" else WINTER_AVAILABLE_DEDUCTION_FACTOR)
    elif mission.westbound_sensitive:
        winter_pen = baseline * (0.03 if cat == "ultra-long" else WESTBOUND_REQUIRED_FACTOR * 0.5)

    mountain_pen = MOUNTAIN_AVAILABLE_PENALTY_NM if mission.mountain_airports else 0.0
    hot_high_pen = HOT_HIGH_AVAILABLE_PENALTY_NM if mission.hot_high_ops else 0.0

    if cat in ("ultra-long", "large") and mission.mountain_airports:
        mountain_pen += 150.0

    available = baseline - payload_pen - winter_pen - mountain_pen - hot_high_pen
    available = max(available, 0.0)

    return PayloadAdjustedRange(
        practical_baseline_nm=baseline,
        payload_penalty_nm=payload_pen,
        winter_penalty_nm=winter_pen,
        mountain_penalty_nm=mountain_pen,
        hot_high_penalty_nm=hot_high_pen,
        available_nm=available,
    )
