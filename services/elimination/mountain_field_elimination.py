"""
Mountain / hot-high field-performance elimination.

Authoritative for short mountain legs (e.g. Dallas–Aspen, Aspen–Telluride).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.mission.models import MissionProfile, PriorityLevel
from services.mission.route_distance_authority import (
    peak_verified_stage_nm,
    resolve_mission_route_authority,
)

_MOUNTAIN_AIRPORT_RE = re.compile(
    r"\b(?:aspen|telluride|sun\s+valley|jackson\s+hole|hailey|eagle|"
    r"montrose|truckee|lake\s+tahoe|aspen|ase|tex)\b",
    re.I,
)

_HEAVY_CATEGORIES = frozenset({"ultra-long", "large", "super-midsize", "super midsize"})
_PREFERRED_SHORT_FIELD = frozenset({
    "pc-24",
    "pilatus pc-24",
    "pc-12 ngx",
    "pc-12",
    "citation cj4",
    "citation cj3+",
    "phenom 300e",
})


@dataclass
class MountainEliminationResult:
    survivors: List[str] = field(default_factory=list)
    eliminated: List[str] = field(default_factory=list)
    reasons: Dict[str, str] = field(default_factory=dict)
    mountain_mission: bool = False

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
            "mountain_mission": self.mountain_mission,
        }


def detect_mountain_mission(profile: MissionProfile, query: str = "") -> bool:
    if profile.mountain_airports or profile.mountain_airport_priority:
        return True
    if profile.short_field_priority != PriorityLevel.NONE:
        blob = " ".join(r.label() for r in (profile.routes or [])) + " " + (query or "")
        if _MOUNTAIN_AIRPORT_RE.search(blob):
            return True
    blob = " ".join(r.label() for r in (profile.routes or [])).lower()
    return bool(_MOUNTAIN_AIRPORT_RE.search(blob) or _MOUNTAIN_AIRPORT_RE.search(query or ""))


def apply_mountain_field_elimination(
    models: List[str],
    profile: MissionProfile,
    *,
    query: str = "",
    model_specs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> MountainEliminationResult:
    """Eliminate heavy platforms on verified short mountain legs."""
    specs = model_specs or {}
    resolutions = resolve_mission_route_authority([r.label() for r in (profile.routes or [])])
    peak_nm = peak_verified_stage_nm(resolutions)

    result = MountainEliminationResult(survivors=list(models))
    if not detect_mountain_mission(profile, query):
        return result

    result.mountain_mission = True
    if peak_nm > 900:
        return result

    for model in list(result.survivors):
        spec = specs.get(model) or specs.get(model.lower()) or {}
        cat = str(spec.get("category") or "").lower()
        short_score = float(spec.get("short_field_score") or 0.5)
        hot_high = float(spec.get("hot_high_score") or 0.5)
        runway_ft = float(spec.get("runway_ft") or 9999)

        key = model.strip().lower()
        if key in _PREFERRED_SHORT_FIELD:
            continue

        if cat in _HEAVY_CATEGORIES:
            result._eliminate(
                model,
                "Heavy cabin class not field-flexible for verified hot/high mountain leg.",
            )
        elif short_score < 0.62 or hot_high < 0.62:
            result._eliminate(
                model,
                f"Insufficient hot/high field performance (short_field={short_score:.2f}, "
                f"hot_high={hot_high:.2f}) for mountain airport mission.",
            )
        elif runway_ft > 4800 and peak_nm < 500:
            result._eliminate(
                model,
                f"Runway footprint ~{int(runway_ft)} ft not aligned with short mountain leg.",
            )

    return result
