"""
Operational-band elimination — rank only inside surviving band.

Brokers eliminate impossible categories first; G280 vs G650 on TEB–LON winter
is not a valid comparison — only aircraft in the same operational band compete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from services.mission.models import MissionProfile, PriorityLevel


class OperationalBand(str, Enum):
    LIGHT_JET = "light_jet"
    MIDSIZE = "midsize"
    SUPER_MID = "super_mid"
    LARGE_CABIN = "large_cabin"
    ULTRA_LONG_RANGE = "ultra_long_range"
    TURBOPROP = "turboprop"
    SHORT_FIELD = "short_field"


# Model → band (subset; unknown models resolved via category catalog)
_MODEL_BAND: Dict[str, OperationalBand] = {
    "citation cj3+": OperationalBand.LIGHT_JET,
    "citation cj4": OperationalBand.LIGHT_JET,
    "phenom 300e": OperationalBand.LIGHT_JET,
    "pc-12 ngx": OperationalBand.TURBOPROP,
    "king air 350": OperationalBand.TURBOPROP,
    "citation latitude": OperationalBand.MIDSIZE,
    "citation longitude": OperationalBand.SUPER_MID,
    "challenger 350": OperationalBand.SUPER_MID,
    "challenger 3500": OperationalBand.SUPER_MID,
    "praetor 600": OperationalBand.SUPER_MID,
    "g280": OperationalBand.SUPER_MID,
    "falcon 2000lxs": OperationalBand.LARGE_CABIN,
    "g450": OperationalBand.LARGE_CABIN,
    "g500": OperationalBand.LARGE_CABIN,
    "g600": OperationalBand.LARGE_CABIN,
    "g650": OperationalBand.ULTRA_LONG_RANGE,
    "g650er": OperationalBand.ULTRA_LONG_RANGE,
    "g700": OperationalBand.ULTRA_LONG_RANGE,
    "global 7500": OperationalBand.ULTRA_LONG_RANGE,
    "global 8000": OperationalBand.ULTRA_LONG_RANGE,
    "falcon 8x": OperationalBand.ULTRA_LONG_RANGE,
    "falcon 10x": OperationalBand.ULTRA_LONG_RANGE,
}

_CATEGORY_TO_BAND = {
    # Accept both verbose and short profile category labels.
    "light": OperationalBand.LIGHT_JET,
    "light jet": OperationalBand.LIGHT_JET,
    "midsize": OperationalBand.MIDSIZE,
    "super-midsize": OperationalBand.SUPER_MID,
    "super midsize": OperationalBand.SUPER_MID,
    "large": OperationalBand.LARGE_CABIN,
    "large cabin": OperationalBand.LARGE_CABIN,
    "ultra-long-range": OperationalBand.ULTRA_LONG_RANGE,
    "ultra-long": OperationalBand.ULTRA_LONG_RANGE,
    "ultra long range": OperationalBand.ULTRA_LONG_RANGE,
    "turboprop": OperationalBand.TURBOPROP,
}


@dataclass
class BandEliminationResult:
    target_band: Optional[OperationalBand]
    survivors: List[str] = field(default_factory=list)
    eliminated: List[str] = field(default_factory=list)
    downgraded: List[str] = field(default_factory=list)
    compromise_labels: Dict[str, str] = field(default_factory=dict)
    elimination_reasons: Dict[str, str] = field(default_factory=dict)
    corridor_nm: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_band": self.target_band.value if self.target_band else None,
            "survivors": list(self.survivors),
            "eliminated": list(self.eliminated),
            "downgraded": list(self.downgraded),
            "compromise_labels": dict(self.compromise_labels),
            "elimination_reasons": dict(self.elimination_reasons),
            "corridor_nm": self.corridor_nm,
        }


def _normalize_model(name: str) -> str:
    return (name or "").strip().lower()


def model_operational_band(model: str, category: Optional[str] = None) -> Optional[OperationalBand]:
    key = _normalize_model(model)
    if key in _MODEL_BAND:
        return _MODEL_BAND[key]
    if category:
        cat = category.strip().lower()
        return _CATEGORY_TO_BAND.get(cat)
    return None


def _required_bands_for_corridor(
    distance_nm: float,
    *,
    nonstop: bool,
    mountain: bool,
) -> Set[OperationalBand]:
    """Minimum operational bands credible for corridor — eliminates lower bands."""
    bands: Set[OperationalBand] = set()
    if mountain:
        bands.add(OperationalBand.TURBOPROP)
        bands.add(OperationalBand.SHORT_FIELD)
        bands.add(OperationalBand.LIGHT_JET)
        bands.add(OperationalBand.MIDSIZE)
        bands.add(OperationalBand.SUPER_MID)
        if distance_nm >= 1200:
            bands.add(OperationalBand.LARGE_CABIN)
        return bands
    if distance_nm >= 3200:
        bands.add(OperationalBand.ULTRA_LONG_RANGE)
        return bands
    if distance_nm >= 2400:
        bands.add(OperationalBand.ULTRA_LONG_RANGE)
        bands.add(OperationalBand.LARGE_CABIN)
        return bands
    if distance_nm >= 1800 and nonstop:
        bands.add(OperationalBand.ULTRA_LONG_RANGE)
        bands.add(OperationalBand.LARGE_CABIN)
        bands.add(OperationalBand.SUPER_MID)
        return bands
    if distance_nm >= 1400:
        bands.add(OperationalBand.SUPER_MID)
        bands.add(OperationalBand.LARGE_CABIN)
        bands.add(OperationalBand.ULTRA_LONG_RANGE)
        return bands
    if distance_nm >= 800:
        bands.add(OperationalBand.LIGHT_JET)
        bands.add(OperationalBand.MIDSIZE)
        bands.add(OperationalBand.SUPER_MID)
        bands.add(OperationalBand.LARGE_CABIN)
        return bands
    bands.add(OperationalBand.LIGHT_JET)
    bands.add(OperationalBand.MIDSIZE)
    bands.add(OperationalBand.TURBOPROP)
    return bands


def determine_operational_band(
    profile: MissionProfile,
    feasible_models: List[str],
    *,
    distance_nm: Optional[float] = None,
    model_categories: Optional[Dict[str, str]] = None,
) -> BandEliminationResult:
    """
    Eliminate models outside corridor-credible operational band(s).
    Returns survivors for in-band ranking only.
    """
    result = BandEliminationResult(target_band=None, corridor_nm=distance_nm)
    if not feasible_models:
        return result

    dist = float(distance_nm or 0)
    nonstop = bool(profile.nonstop_required)
    mountain = bool(
        profile.mountain_airport_priority
        or profile.mountain_airports
        or profile.short_field_priority != PriorityLevel.NONE
    )

    allowed = _required_bands_for_corridor(dist, nonstop=nonstop, mountain=mountain)
    if not allowed:
        result.survivors = list(feasible_models)
        return result

    cats = model_categories or {}
    by_band: Dict[OperationalBand, List[str]] = {}
    for model in feasible_models:
        band = model_operational_band(model, cats.get(_normalize_model(model)))
        if band is None:
            result.survivors.append(model)
            continue
        if band in allowed:
            by_band.setdefault(band, []).append(model)
        else:
            reason = (
                f"Outside operational band for {int(dist)} nm corridor "
                f"({band.value} not credible vs {', '.join(b.value for b in allowed)})"
            )
            from services.elimination.conditional_downgrade import (
                compromise_label_for_reason,
                elimination_severity,
            )

            if elimination_severity(reason, distance_nm=dist, elimination_kind="band") == "hard":
                result.eliminated.append(model)
            else:
                result.downgraded.append(model)
                result.compromise_labels[model] = compromise_label_for_reason(
                    reason, distance_nm=dist
                ).value
            result.elimination_reasons[model] = reason

    if not by_band:
        # All catalog models fell outside corridor-credible bands — do not restore eliminated aircraft.
        result.survivors = [] if result.eliminated else list(feasible_models)
        return result

    if mountain:
        # Field-performance missions: retain every model in an allowed band for ranking
        # (PC-24 light + super-mid field types are not collapsed to a single band).
        for band, models in by_band.items():
            if band in allowed:
                result.survivors.extend(models)
        if result.survivors:
            result.target_band = OperationalBand.SHORT_FIELD
        return result

    # Target band = highest credible band with survivors (ULR > large > super-mid ...)
    order = [
        OperationalBand.ULTRA_LONG_RANGE,
        OperationalBand.LARGE_CABIN,
        OperationalBand.SUPER_MID,
        OperationalBand.MIDSIZE,
        OperationalBand.LIGHT_JET,
        OperationalBand.TURBOPROP,
        OperationalBand.SHORT_FIELD,
    ]
    for band in order:
        if band in by_band and band in allowed:
            result.target_band = band
            result.survivors = by_band[band]
            for other_band, models in by_band.items():
                if other_band != band:
                    for m in models:
                        reason = (
                            f"Not comparable in-band; mission band is {band.value}"
                        )
                        result.downgraded.append(m)
                        result.compromise_labels[m] = "VIABLE WITH COMPROMISES"
                        result.elimination_reasons[m] = reason
            break

    return result


def filter_models_to_operational_band(
    models: List[str],
    band_result: BandEliminationResult,
    *,
    include_downgraded: bool = True,
) -> List[str]:
    if not band_result.survivors and not (include_downgraded and band_result.downgraded):
        return []
    survivor_set = {_normalize_model(m) for m in band_result.survivors}
    out = [m for m in models if _normalize_model(m) in survivor_set]
    if include_downgraded:
        seen = {_normalize_model(m) for m in out}
        for m in band_result.downgraded:
            key = _normalize_model(m)
            if key not in seen:
                out.append(m)
                seen.add(key)
    return out


def models_comparable_in_band(
    model_a: str,
    model_b: str,
    *,
    model_categories: Optional[Dict[str, str]] = None,
) -> Tuple[bool, str]:
    """True only when both models share operational band (valid broker comparison)."""
    cats = model_categories or {}
    ba = model_operational_band(model_a, cats.get(_normalize_model(model_a)))
    bb = model_operational_band(model_b, cats.get(_normalize_model(model_b)))
    if ba is None or bb is None:
        return True, ""
    if ba == bb:
        return True, ""
    return False, (
        f"Invalid cross-band comparison ({ba.value} vs {bb.value}) — "
        "compare only within the same operational band."
    )
