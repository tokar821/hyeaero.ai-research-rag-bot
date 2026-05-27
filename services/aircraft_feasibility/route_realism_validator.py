"""
Route realism validator — conservative stage lengths and ultra-long corridor detection.

Never fabricates city pairs: distances come from the catalog table or validated route
extraction only. Heuristic fallbacks are flagged so downstream logic does not treat them
as firm operational facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from services.aircraft_feasibility.mission_context import FeasibilityMissionContext
from services.mission.route_distance_authority import resolve_route_distance

# Canonical ultra-long corridors — light jets must not be recommended nonstop unless stop_required.
ULTRA_LONG_CORRIDORS: Tuple[Dict[str, Any], ...] = (
    {
        "id": "nyc_dubai",
        "origin_re": r"\b(?:new\s+york|nyc|teterboro|jfk|teb)\b",
        "dest_re": r"\b(?:dubai|dxb)\b",
        "catalog_keys": (
            "new york -> dubai",
            "nyc -> dubai",
            "jfk -> dubai",
            "teterboro -> london",
            "teb -> london",
            "nyc -> london",
        ),
        "min_stage_nm": 5500.0,
    },
    {
        "id": "la_london",
        "origin_re": r"\b(?:los\s+angeles|la\b|lax|van\s+nuys)\b",
        "dest_re": r"\b(?:london|lhr|lgw|stansted)\b",
        "catalog_keys": (
            "los angeles -> london",
            "la -> london",
            "los angeles -> london",
        ),
        "min_stage_nm": 5200.0,
    },
    {
        "id": "sfo_tokyo",
        "origin_re": r"\b(?:san\s+francisco|sfo|oak)\b",
        "dest_re": r"\b(?:tokyo|hnd|nrt)\b",
        "catalog_keys": ("san francisco -> tokyo", "sfo -> tokyo"),
        "min_stage_nm": 5000.0,
    },
)

@dataclass
class RouteRealismResult:
    """Route validation outcome for feasibility gating."""

    realistic: bool = True
    ultra_long_corridor: bool = False
    corridor_id: Optional[str] = None
    stage_distance_nm: float = 0.0
    distance_source: str = "none"  # catalog | route_table | heuristic | none
    route_label: str = ""
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "realistic": self.realistic,
            "ultra_long_corridor": self.ultra_long_corridor,
            "corridor_id": self.corridor_id,
            "stage_distance_nm": round(self.stage_distance_nm, 1),
            "distance_source": self.distance_source,
            "route_label": self.route_label,
            "issues": list(self.issues),
        }


def _normalize_route_key(label: str) -> str:
    s = (label or "").strip().lower()
    s = s.replace("→", "->")
    return re.sub(r"\s+", " ", s)


def resolve_stage_distance_nm(route_label: str) -> Tuple[float, str]:
    """
    Return (distance_nm, source). Does not invent places — only measures known labels.
    """
    if not (route_label or "").strip():
        return 0.0, "none"

    resolution = resolve_route_distance(route_label)
    if resolution.source == "unresolved":
        return 0.0, "unresolved"
    return resolution.distance_nm, resolution.source


def match_ultra_long_corridor(route_label: str, stage_nm: float) -> Optional[str]:
    """Return corridor id when route matches a known ultra-long city pair."""
    blob = _normalize_route_key(route_label)
    if not blob:
        return None

    for corridor in ULTRA_LONG_CORRIDORS:
        for ck in corridor.get("catalog_keys") or ():
            if _normalize_route_key(ck) == blob:
                return str(corridor["id"])

        if re.search(corridor["origin_re"], blob) and re.search(corridor["dest_re"], blob):
            min_nm = float(corridor.get("min_stage_nm") or 4500)
            effective_nm = stage_nm if stage_nm > 0 else min_nm
            if effective_nm >= min_nm * 0.85:
                return str(corridor["id"])
    return None


def is_ultra_long_corridor_mission(mission: FeasibilityMissionContext) -> Tuple[bool, Optional[str]]:
    """Whether mission is on a corridor where light jets are banned for nonstop."""
    cid = match_ultra_long_corridor(mission.route_label, mission.stage_distance_nm)
    return cid is not None, cid


def validate_route_realism(mission: FeasibilityMissionContext) -> RouteRealismResult:
    """
    Validate route labeling and distance realism before aircraft evaluation.
    """
    label = (mission.route_label or "").strip()
    if not label:
        return RouteRealismResult(
            realistic=True,
            distance_source="none",
            issues=[],
        )

    stage_nm, source = resolve_stage_distance_nm(label)
    if stage_nm <= 0:
        return RouteRealismResult(
            realistic=False,
            route_label=label,
            distance_source=source,
            issues=["Route places could not be resolved to a stage length."],
        )

    issues: List[str] = []
    if source == "unresolved":
        issues.append("Route distance unresolved — ranked recommendations blocked.")
        return RouteRealismResult(
            realistic=False,
            route_label=label,
            distance_source=source,
            issues=issues,
        )

    corridor_id = match_ultra_long_corridor(label, stage_nm)
    ultra = corridor_id is not None

    realistic = source in ("catalog", "geodesic")

    return RouteRealismResult(
        realistic=realistic,
        ultra_long_corridor=ultra,
        corridor_id=corridor_id,
        stage_distance_nm=stage_nm,
        distance_source=source,
        route_label=label,
        issues=issues,
    )
