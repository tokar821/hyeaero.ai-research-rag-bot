"""
Airport operational constraints — elevation, runway, hot/high, climb gradient, category.

Uses :mod:`services.airport.airport_database` for expanded ICAO/place resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from services.airport.airport_database import (
    AirportOperationalProfile,
    mission_airport_profiles,
    profile_from_icao,
    resolve_airports_in_text,
)

# Re-export for callers
__all__ = [
    "AirportOperationalProfile",
    "AirportConstraintEliminationResult",
    "apply_airport_constraint_elimination",
    "mission_airport_constraints",
    "resolve_airports_for_route",
]

_HEAVY_CATEGORIES = frozenset({"ultra-long", "large"})


@dataclass
class AirportConstraintEliminationResult:
    survivors: List[str] = field(default_factory=list)
    eliminated: List[str] = field(default_factory=list)
    reasons: Dict[str, str] = field(default_factory=dict)
    airports: List[AirportOperationalProfile] = field(default_factory=list)

    def _eliminate(self, model: str, reason: str) -> None:
        if model in self.survivors:
            self.survivors.remove(model)
        if model not in self.eliminated:
            self.eliminated.append(model)
        self.reasons[model] = reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "survivors": list(self.survivors),
            "eliminated": list(self.eliminated),
            "reasons": dict(self.reasons),
            "airports": [a.to_dict() for a in self.airports],
        }


def resolve_airports_for_route(route_label: str) -> List[AirportOperationalProfile]:
    return resolve_airports_in_text(route_label)


def mission_airport_constraints(route_labels: List[str]) -> List[AirportOperationalProfile]:
    return mission_airport_profiles(route_labels)


def _aircraft_meets_airport(
    model: str,
    spec: Dict[str, Any],
    airport: AirportOperationalProfile,
) -> Tuple[bool, str]:
    cat = str(spec.get("category") or "").lower()
    runway_ft = float(spec.get("runway_ft") or 9999)
    short_score = float(spec.get("short_field_score") or 0.5)
    hot_high = float(spec.get("hot_high_score") or 0.5)
    climb_capable = float(spec.get("climb_gradient_capable_pct") or 0)
    if climb_capable <= 0:
        climb_capable = max(short_score * 7.5, hot_high * 6.5)

    if airport.operational_category in ("mountain_high",):
        if cat in _HEAVY_CATEGORIES:
            return False, (
                f"{model}: {cat} platform not field-credible at {airport.name} "
                f"({airport.elevation_ft} ft elevation, {airport.hot_high_category} hot/high)."
            )
        if runway_ft > airport.max_recommended_runway_ft:
            return False, (
                f"{model}: runway need ~{int(runway_ft)} ft exceeds "
                f"{airport.name} operational limit ~{airport.max_recommended_runway_ft} ft."
            )
        min_hot = 0.65
        if cat in ("super-midsize", "super midsize", "midsize"):
            min_hot = 0.58
        if hot_high < min_hot:
            return False, (
                f"{model}: hot/high score {hot_high:.2f} below mountain strip minimum "
                f"for {airport.name} (DA ~{airport.density_altitude_summer_ft} ft summer)."
            )
        if climb_capable < airport.climb_gradient_pct * 0.85:
            return False, (
                f"{model}: climb capability insufficient for {airport.name} "
                f"(requires ~{airport.climb_gradient_pct}% gradient)."
            )
        return True, ""

    if airport.operational_category == "hot_high_hub":
        if hot_high < 0.55 and cat in _HEAVY_CATEGORIES:
            return False, (
                f"{model}: hot/high performance marginal for {airport.name} "
                f"summer operations (DA ~{airport.density_altitude_summer_ft} ft)."
            )

    if airport.operational_category in ("caribbean_short", "caribbean"):
        if cat in ("ultra-long", "large") and runway_ft > 6000:
            return False, (
                f"{model}: runway footprint too large for {airport.name} "
                f"({airport.operational_category} ops)."
            )
        if short_score < 0.55 and cat not in ("light", "turboprop", "midsize"):
            return False, (
                f"{model}: short-field performance inadequate for "
                f"{airport.operational_category} airport access."
            )

    return True, ""


def apply_airport_constraint_elimination(
    models: List[str],
    *,
    route_labels: List[str],
    model_specs: Dict[str, Dict[str, Any]],
) -> AirportConstraintEliminationResult:
    """Eliminate aircraft that fail airport operational constraints on mission endpoints."""
    airports = mission_airport_constraints(route_labels)
    result = AirportConstraintEliminationResult(survivors=list(models), airports=airports)
    if not airports:
        return result

    for model in list(result.survivors):
        spec = model_specs.get(model) or model_specs.get(model.lower()) or {}
        for airport in airports:
            ok, reason = _aircraft_meets_airport(model, spec, airport)
            if not ok:
                result._eliminate(model, reason)
                break

    return result
