"""
Wind and seasonal realism — structured headwind / winter penalties for planning.

Feeds reserves, route feasibility, and dispatch reliability (not brochure range).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.consultant.mission_state import MissionState
from services.mission.models import MissionProfile

_WESTBOUND_ROUTE_RE = re.compile(
    r"\b(?:westbound|west\s*bound|heading\s+west|europe\s*to\s*us|uk\s*to\s*(?:us|new\s+york)|"
    r"london\s*to\s*(?:new\s+york|teb|teterboro|boston))\b",
    re.I,
)


@dataclass
class WindAdjustment:
    stage_distance_nm: float
    westbound_penalty_nm: float
    winter_extra_nm: float
    total_penalty_nm: float
    effective_wind_factor: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_distance_nm": round(self.stage_distance_nm, 1),
            "westbound_penalty_nm": round(self.westbound_penalty_nm, 1),
            "winter_extra_nm": round(self.winter_extra_nm, 1),
            "total_penalty_nm": round(self.total_penalty_nm, 1),
            "effective_wind_factor": round(self.effective_wind_factor, 4),
            "notes": list(self.notes),
        }


def _is_winter(mission: MissionState, profile: Optional[MissionProfile]) -> bool:
    seasonal = (mission.seasonal_constraints or "").lower()
    if "winter" in seasonal or "january" in seasonal or "february" in seasonal:
        return True
    if profile and (profile.seasonal_note or "").lower().find("winter") >= 0:
        return True
    return False


def _is_westbound(
    mission: MissionState,
    profile: Optional[MissionProfile],
    *,
    route_label: str = "",
) -> bool:
    if mission.westbound or (profile and profile.westbound_sensitive):
        return True
    blob = " ".join(mission.routes or []) + " " + (route_label or "")
    return bool(_WESTBOUND_ROUTE_RE.search(blob))


def compute_wind_adjustment(
    mission: MissionState,
    *,
    profile: Optional[MissionProfile] = None,
    stage_distance_nm: float = 0.0,
    route_label: str = "",
) -> WindAdjustment:
    """
    Planning-equivalent NM penalty for dominant wind/season on the stage.

    Uses conservative broker factors (not live METAR).
    """
    stage = max(float(stage_distance_nm or 0), 0.0)
    west = _is_westbound(mission, profile, route_label=route_label)
    winter = _is_winter(mission, profile)

    west_pen = 0.0
    winter_extra = 0.0
    notes: List[str] = []

    if west and stage > 0:
        # ~6–8% stage equivalent on typical westbound executive leg
        west_pen = stage * 0.07
        notes.append(f"Westbound planning margin ~{int(west_pen)} nm ({int(stage)} nm stage).")

    if winter and west and stage > 0:
        winter_extra = stage * 0.04
        notes.append(
            f"Winter westbound headwind reserve ~{int(winter_extra)} nm — "
            "treat brochure range as optimistic."
        )
    elif winter and stage >= 1800:
        winter_extra = stage * 0.02
        notes.append(f"Seasonal winter fuel margin ~{int(winter_extra)} nm on long stage.")

    total = west_pen + winter_extra
    factor = min(0.22, total / max(stage, 400.0)) if stage > 0 else 0.0

    return WindAdjustment(
        stage_distance_nm=stage,
        westbound_penalty_nm=west_pen,
        winter_extra_nm=winter_extra,
        total_penalty_nm=total,
        effective_wind_factor=factor,
        notes=notes,
    )
