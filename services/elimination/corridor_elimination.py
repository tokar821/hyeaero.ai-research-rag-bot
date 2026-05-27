"""
Corridor hard elimination — verified route authority only.

Eliminates aircraft before ranking; eliminated models must not reach LLM context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.mission.models import MissionProfile
from services.mission.route_distance_authority import (
    RouteDistanceResolution,
    peak_catalog_stage_nm,
    peak_verified_stage_nm,
    resolve_mission_route_authority,
)

_TRANSATLANTIC_NONSTOP_MIN_NM = 2600.0
_SUPER_MID_CATEGORIES = frozenset({"super-midsize", "super midsize", "midsize"})
_LIGHT_CATEGORIES = frozenset({"light", "turboprop"})
_ULR_ONLY_MIN_NM = 4500.0


@dataclass
class CorridorEliminationResult:
    survivors: List[str] = field(default_factory=list)
    eliminated: List[str] = field(default_factory=list)
    reasons: Dict[str, str] = field(default_factory=dict)
    corridor_id: str = ""
    verified_stage_nm: float = 0.0
    route_confidence_min: float = 1.0

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
            "corridor_id": self.corridor_id,
            "verified_stage_nm": round(self.verified_stage_nm, 1),
            "route_confidence_min": round(self.route_confidence_min, 3),
        }


def _is_winter(profile: MissionProfile) -> bool:
    note = (profile.seasonal_note or "").lower()
    return "winter" in note or "january" in note or "february" in note


def _is_transatlantic_route(resolutions: List[RouteDistanceResolution]) -> bool:
    for r in resolutions:
        blob = (r.route_label or "").lower()
        if re.search(r"\b(?:london|paris|geneva|europe|dublin|frankfurt)\b", blob):
            if re.search(r"\b(?:new\s+york|nyc|teterboro|teb|boston)\b", blob):
                return True
        if r.distance_nm >= _TRANSATLANTIC_NONSTOP_MIN_NM and r.is_verified:
            if re.search(r"london|paris|europe", blob):
                return True
    return False


def _model_category(model: str, categories: Dict[str, str]) -> str:
    return (categories.get(model.lower()) or "").strip().lower()


def apply_corridor_hard_elimination(
    models: List[str],
    profile: MissionProfile,
    *,
    model_categories: Optional[Dict[str, str]] = None,
    route_resolutions: Optional[List[RouteDistanceResolution]] = None,
) -> CorridorEliminationResult:
    """Hard corridor gate using verified distances only."""
    cats = model_categories or {}
    resolutions = route_resolutions or resolve_mission_route_authority(
        [r.label() for r in (profile.routes or [])]
    )
    peak_nm = peak_verified_stage_nm(resolutions)
    catalog_peak_nm = peak_catalog_stage_nm(resolutions)
    conf_min = min((r.confidence for r in resolutions), default=0.0) if resolutions else 0.0

    result = CorridorEliminationResult(
        survivors=list(models),
        verified_stage_nm=peak_nm,
        route_confidence_min=conf_min,
    )

    if not models or peak_nm <= 0:
        return result

    nonstop = bool(profile.nonstop_required)
    winter = _is_winter(profile)
    transatlantic = _is_transatlantic_route(resolutions)
    hard_peak = catalog_peak_nm if catalog_peak_nm > 0 else 0.0
    elim_peak = hard_peak if hard_peak > 0 else peak_nm

    # Hard nonstop corridor elimination requires catalog-verified stage length.
    if nonstop and hard_peak <= 0:
        return result

    if transatlantic and nonstop and elim_peak >= _TRANSATLANTIC_NONSTOP_MIN_NM and hard_peak > 0:
        result.corridor_id = "transatlantic_nonstop"
        for model in list(result.survivors):
            cat = _model_category(model, cats)
            if cat in _LIGHT_CATEGORIES:
                result._eliminate(
                    model,
                    "Light/turboprop not credible for verified transatlantic nonstop corridor.",
                )
            elif cat in _SUPER_MID_CATEGORIES or (
                winter and cat in ("midsize", "super-midsize", "super midsize")
            ):
                result._eliminate(
                    model,
                    "Super-mid not credible for verified transatlantic nonstop"
                    + (" winter mission." if winter else " mission."),
                )

    if elim_peak >= _ULR_ONLY_MIN_NM and nonstop and hard_peak > 0:
        result.corridor_id = result.corridor_id or "ultra_long_nonstop"
        for model in list(result.survivors):
            cat = _model_category(model, cats)
            if cat not in ("ultra-long", "ultra long range") and cat in (
                _LIGHT_CATEGORIES
                | _SUPER_MID_CATEGORIES
                | frozenset({"midsize", "large", "large cabin"})
            ):
                result._eliminate(
                    model,
                    f"Corridor ~{int(elim_peak)} nm catalog-verified nonstop requires ULR band.",
                )

    return result
