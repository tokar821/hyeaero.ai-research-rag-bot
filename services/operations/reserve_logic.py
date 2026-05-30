"""
Reserve logic — NBAA-style reserve margin beyond simple range > route checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from services.consultant.mission_state import MissionState
from services.mission.adapters import mission_state_to_profile
from services.mission.models import MissionProfile


@dataclass
class ReserveMarginAssessment:
    required_nm: float
    reserve_nm: float
    payload_margin_nm: float
    dispatch_margin_nm: float
    planning_mode: str
    broker_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required_nm": round(self.required_nm, 1),
            "reserve_nm": round(self.reserve_nm, 1),
            "payload_margin_nm": round(self.payload_margin_nm, 1),
            "dispatch_margin_nm": round(self.dispatch_margin_nm, 1),
            "planning_mode": self.planning_mode,
            "broker_summary": self.broker_summary,
        }


def assess_reserve_margin(
    mission: Any,
    *,
    stage_nm: float,
    practical_nm: float,
    query: str = "",
    mission_profile: Optional[MissionProfile] = None,
) -> ReserveMarginAssessment:
    """Full reserve stack — not brochure range alone."""
    ms = mission if isinstance(mission, MissionState) else MissionState()
    profile = mission_profile or mission_state_to_profile(ms)

    try:
        from services.operational.mission_operational_assessment import build_mission_operational_context
        from services.operational.payload_realism import build_mission_payload_profile

        ctx = build_mission_operational_context(ms, profile, query=query)
        reserve_total = ctx.reserve.total_reserve_nm
        mode = ctx.planning_mode
    except Exception:
        from services.operational.reserve_profiles import infer_planning_mode

        mode = infer_planning_mode(profile, query=query).value
        reserve_total = 200.0

    dispatch_margin = practical_nm - stage_nm - reserve_total
    payload_margin = max(0.0, dispatch_margin) * 0.35

    if dispatch_margin >= 400:
        summary = "Comfortable NBAA reserve and dispatch margin for stated mission."
    elif dispatch_margin >= 150:
        summary = "Adequate reserve margin — monitor winter westbound and payload restrictions."
    elif dispatch_margin >= 0:
        summary = "Thin dispatch margin — technically possible, not reliably dependable."
    else:
        summary = "Insufficient margin after NBAA reserves — nonstop not realistic."

    return ReserveMarginAssessment(
        required_nm=stage_nm,
        reserve_nm=reserve_total,
        payload_margin_nm=payload_margin,
        dispatch_margin_nm=dispatch_margin,
        planning_mode=str(mode),
        broker_summary=summary,
    )


__all__ = ["ReserveMarginAssessment", "assess_reserve_margin"]
