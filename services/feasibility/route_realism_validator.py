"""
Route realism validator — canonical ULR corridors and light-jet prohibitions.

Never allow light jets (or turboprops) on these nonstop-style missions unless the user
accepts fuel stops (``stop_required`` / ``nonstop_required=false``):

  - NYC → Dubai
  - LA → London
  - SFO → Tokyo
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from services.consultant.route_feasibility import estimate_route_distance_nm

# Light / entry jets — never on ULR oceanic corridors without stops
_LIGHT_JET_CATEGORIES = frozenset({"light", "turboprop"})


@dataclass(frozen=True)
class CanonicalCorridor:
    corridor_id: str
    label: str
    origin_patterns: Tuple[str, ...]
    dest_patterns: Tuple[str, ...]
    min_stage_nm: float


CANONICAL_ULR_CORRIDORS: Tuple[CanonicalCorridor, ...] = (
    CanonicalCorridor(
        corridor_id="nyc_dubai",
        label="New York → Dubai",
        origin_patterns=(
            r"\b(?:new\s+york|nyc|teterboro|teb|jfk|ewr|lga)\b",
        ),
        dest_patterns=(
            r"\b(?:dubai|dxb|dwc|abu\s+dhabi)\b",
        ),
        min_stage_nm=5200.0,
    ),
    CanonicalCorridor(
        corridor_id="la_london",
        label="Los Angeles → London",
        origin_patterns=(
            r"\b(?:los\s+angeles|lax|van\s+nuys|santa\s+ana|orange\s+county)\b",
        ),
        dest_patterns=(
            r"\b(?:london|lhr|lgw|stansted|farnborough|paris|geneva)\b",
        ),
        min_stage_nm=4800.0,
    ),
    CanonicalCorridor(
        corridor_id="sfo_tokyo",
        label="San Francisco → Tokyo",
        origin_patterns=(
            r"\b(?:san\s+francisco|sfo|oak|oakland)\b",
        ),
        dest_patterns=(
            r"\b(?:tokyo|hnd|nrt|narita)\b",
        ),
        min_stage_nm=4200.0,
    ),
)


@dataclass(frozen=True)
class RouteRealismResult:
    """Outcome of matching a mission against canonical corridor rules."""

    matched: bool = False
    corridor_id: str = ""
    corridor_label: str = ""
    stage_distance_nm: float = 0.0
    light_jet_prohibited: bool = False
    stop_required_exempts: bool = False
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "matched": self.matched,
            "corridor_id": self.corridor_id,
            "corridor_label": self.corridor_label,
            "stage_distance_nm": round(self.stage_distance_nm, 1),
            "light_jet_prohibited": self.light_jet_prohibited,
            "stop_required_exempts": self.stop_required_exempts,
            "notes": list(self.notes),
        }


def _blob_matches(blob: str, patterns: Tuple[str, ...]) -> bool:
    return any(re.search(p, blob, re.I) for p in patterns)


def match_canonical_corridor(
    route_label: str,
    *,
    stage_distance_nm: float = 0.0,
) -> Optional[CanonicalCorridor]:
    """Return the matched ULR corridor, if any."""
    blob = (route_label or "").strip().lower()
    if not blob:
        return None

  # Require directional cue — avoid matching two cities in unrelated context
    if not re.search(r"\b(?:to|->|→)\b", blob):
        return None

    for corridor in CANONICAL_ULR_CORRIDORS:
        if not _blob_matches(blob, corridor.origin_patterns):
            continue
        if not _blob_matches(blob, corridor.dest_patterns):
            continue
        if stage_distance_nm > 0 and stage_distance_nm < corridor.min_stage_nm * 0.75:
            continue
        return corridor
    return None


def validate_route_realism(
    *,
    route_label: str = "",
    stage_distance_nm: float = 0.0,
    nonstop_required: bool = True,
    stop_required: bool = False,
) -> RouteRealismResult:
    """
    Validate whether the mission hits a canonical ULR city-pair.

    ``stop_required`` (or ``nonstop_required=False``) exempts the light-jet hard ban —
    user accepts fuel stops / multi-leg operations.
    """
    stage = stage_distance_nm
    if stage <= 0 and route_label:
        stage = estimate_route_distance_nm(route_label)

    corridor = match_canonical_corridor(route_label, stage_distance_nm=stage)
    if corridor is None:
        return RouteRealismResult(stage_distance_nm=stage)

    exempts = bool(stop_required) or not nonstop_required
    return RouteRealismResult(
        matched=True,
        corridor_id=corridor.corridor_id,
        corridor_label=corridor.label,
        stage_distance_nm=stage,
        light_jet_prohibited=not exempts,
        stop_required_exempts=exempts,
        notes=(
            f"Canonical ULR corridor: {corridor.label} (~{int(stage)} nm).",
            "Light jets prohibited unless stop_required=true.",
        ),
    )


def light_jet_corridor_rejection_reason(
    *,
    model: str,
    aircraft_category: str,
    realism: RouteRealismResult,
) -> Optional[str]:
    """Hard-reject reason for light jets on canonical corridors; None if allowed."""
    if not realism.matched or not realism.light_jet_prohibited:
        return None
    cat = (aircraft_category or "").strip().lower()
    if cat not in _LIGHT_JET_CATEGORIES:
        return None
    return (
        f"route_realism[{realism.corridor_id}]: {model} ({cat}) cannot realistically operate "
        f"{realism.corridor_label} nonstop — practical range and reserves require ULR or "
        f"midsize+ with fuel stops (set stop_required=true if tech stops are acceptable)."
    )
