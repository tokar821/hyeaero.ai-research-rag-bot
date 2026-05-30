"""
Operational realism bridge — unifies seasonal, reserve, and dispatch modules for ranking/capability.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.consultant.mission_state import MissionState
from services.mission.adapters import mission_state_to_profile
from services.mission.models import MissionProfile


def apply_seasonal_penalty_to_context(
    mission: MissionState,
    *,
    query: str = "",
    peak_stage_nm: float = 0.0,
) -> Dict[str, Any]:
    """Return seasonal penalty metadata for operational context builders."""
    from services.operations.seasonal_penalties import infer_seasonal_penalty

    route = (mission.routes or [""])[0] if mission.routes else ""
    penalty = infer_seasonal_penalty(mission, query=query, route_label=route)
    return {
        "extra_nm": penalty.extra_nm,
        "payload_factor": penalty.payload_factor,
        "label": penalty.label,
        "dispatch_note": penalty.dispatch_note,
        "peak_stage_nm": peak_stage_nm,
    }


def assess_mission_operational_realism(
    mission: MissionState,
    model: str,
    profile: Dict[str, Any],
    *,
    query: str = "",
    mission_profile: Optional[MissionProfile] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Full operational realism pass for capability and ranking overlays.
    """
    mp = mission_profile or mission_state_to_profile(mission)
    spec = profile or {}
    practical_nm = float(spec.get("practical_nm") or spec.get("range_nm") or 0)

    stage_nm = 0.0
    if mission.routes:
        try:
            from services.consultant.named_aircraft_capability import _route_distance_nm

            stage_nm = _route_distance_nm(mission)
        except Exception:
            pass
    if stage_nm <= 0 and isinstance(data_used, dict):
        ctx = data_used.get("mission_operational_context") or {}
        if isinstance(ctx, dict):
            stage_nm = float(ctx.get("peak_stage_nm") or 0)

    seasonal = apply_seasonal_penalty_to_context(mission, query=query, peak_stage_nm=stage_nm)

    from services.operations.reserve_logic import assess_reserve_margin
    from services.operations.dispatch_reality import assess_dispatch_reality

    reserve = assess_reserve_margin(
        mission,
        stage_nm=stage_nm or practical_nm * 0.6,
        practical_nm=practical_nm,
        query=query,
        mission_profile=mp,
    )

    operational_context = None
    if isinstance(data_used, dict):
        raw_ctx = data_used.get("mission_operational_context")
        if isinstance(raw_ctx, dict):
            try:
                from services.operational.mission_operational_assessment import (
                    build_mission_operational_context,
                )

                operational_context = build_mission_operational_context(mission, mp, query=query)
            except Exception:
                operational_context = None

    dispatch = assess_dispatch_reality(
        model,
        spec,
        mission,
        query=query,
        operational_context=operational_context,
    )

    return {
        "seasonal": seasonal,
        "reserve": reserve.to_dict(),
        "dispatch": dispatch.to_dict(),
        "broker_summary": reserve.broker_summary,
        "dispatch_label": dispatch.broker_label,
    }


def merge_operational_realism_into_capability(
    evaluation: Dict[str, Any],
    realism: Dict[str, Any],
) -> Dict[str, Any]:
    """Adjust capability verdict using operational realism (not brochure range alone)."""
    reserve = realism.get("reserve") or {}
    dispatch = realism.get("dispatch") or {}
    margin = float(reserve.get("dispatch_margin_nm") or 0)
    reasons = list(evaluation.get("reasons") or [])

    if not dispatch.get("technically_possible"):
        evaluation["verdict"] = "NOT REALISTIC"
        reasons.insert(0, str(dispatch.get("explanation") or "Not dispatch-reliable with NBAA reserves."))
    elif not dispatch.get("operationally_dependable"):
        if evaluation.get("verdict") == "FEASIBLE":
            evaluation["verdict"] = "MARGINAL"
        reasons.append(str(dispatch.get("explanation") or ""))
    elif margin < 150:
        if evaluation.get("verdict") == "FEASIBLE":
            evaluation["verdict"] = "MARGINAL"
        reasons.append(str(reserve.get("broker_summary") or ""))

    seasonal = realism.get("seasonal") or {}
    extra_nm = float(seasonal.get("extra_nm") or 0)
    if extra_nm > 300:
        note = seasonal.get("dispatch_note")
        if note and note not in reasons:
            reasons.append(str(note))
        if evaluation.get("verdict") == "FEASIBLE" and extra_nm >= 400:
            evaluation["verdict"] = "MARGINAL"
            reasons.insert(
                0,
                "Winter westbound Pacific stage with NBAA reserves — brochure range is not dispatch planning range.",
            )
    if seasonal.get("label") == "winter_westbound_pacific":
        cat = str(evaluation.get("category") or "").lower()
        margin = float(reserve.get("dispatch_margin_nm") or 0)
        if cat in ("super-midsize", "midsize", "light"):
            evaluation["verdict"] = "NOT REALISTIC"
            reasons.insert(
                0,
                "Winter westbound Pacific with NBAA reserves — this category is not dispatch-reliable on the stage.",
            )
        elif margin < 200 or evaluation.get("verdict") in ("FEASIBLE", "MARGINAL"):
            evaluation["verdict"] = "NOT REALISTIC"
            reasons.insert(
                0,
                "Winter westbound Pacific with NBAA reserves — stage length and headwind margin exceed dependable ULR dispatch planning.",
            )

    evaluation["reasons"] = [r for r in reasons if r][:6]
    evaluation["operational_realism"] = realism
    return evaluation


__all__ = [
    "apply_seasonal_penalty_to_context",
    "assess_mission_operational_realism",
    "merge_operational_realism_into_capability",
]
