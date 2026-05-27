"""
Pre-recommendation mission validation checks for QA.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from evals.aviation_qa.schemas import RealismExpectations


def validate_mission_before_recommendation(
    *,
    query: str,
    turn_profile: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
    mission_category: Optional[str] = None,
    realism: Optional[RealismExpectations] = None,
) -> Tuple[str, List[str]]:
    """
    Validate pipeline mission handling.

    Returns ``(status, issues)`` where status is PASS | WARN | FAIL.
    """
    issues: List[str] = []
    routes = turn_profile.get("routes") or []
    rec_models = [
        str(r.get("model") or "")
        for r in (recommendations or [])
        if isinstance(r, dict) and r.get("model")
    ]

    # No fallback shortlist when mission is infeasible
    if not rec_models and routes:
        # Expected for impossible missions — good
        pass
    elif not rec_models and not routes:
        issues.append("no_route_extracted_and_no_recommendations")

    # Hard gate: long-haul transpacific / NY-Tokyo — not short hops like Tokyo–Seoul
    blob = (query or "").lower() + " " + str(routes).lower()
    try:
        from services.consultant.route_feasibility import estimate_route_distance_nm
        from services.recommendation.mission_ranker import mission_max_leg_nm
        from services.consultant.mission_state import MissionState

        route_labels = []
        for r in routes:
            if isinstance(r, dict):
                route_labels.append(f"{r.get('origin', '')} -> {r.get('destination', '')}")
            else:
                route_labels.append(str(r))
        ms = MissionState(routes=route_labels)
        max_leg = mission_max_leg_nm(ms)
    except Exception:
        max_leg = 0.0

    long_haul_tokyo = bool(
        re.search(r"new\s+york.*tokyo|nyc.*tokyo|san\s+francisco.*tokyo|sfo.*tokyo", blob)
    )
    ulr_mission = max_leg >= 4800 or long_haul_tokyo
    if max_leg and max_leg < 2000:
        ulr_mission = False
    elif ulr_mission and not any(x in blob for x in ("nonstop", "westbound", "winter", "ulr")):
        ulr_mission = max_leg >= 4800
    if ulr_mission:
        super_mids = {
            "Challenger 350",
            "Praetor 600",
            "Citation Longitude",
            "Challenger Longitude",
            "Citation Latitude",
            "Gulfstream G280",
        }
        for m in rec_models[:5]:
            if m in super_mids:
                issues.append(f"fallback_or_invalid_ulr_pick:{m}")

    try:
        from services.mission.models import MissionProfile
        from services.recommendation.hard_mission_elimination import (
            detect_hard_elimination_context,
            hard_elimination_reason,
        )
        from services.state.mission_state import persistent_to_mission_profile, load_persistent_mission_state

        # Build minimal profile from turn dict
        mp = MissionProfile()
        for r in routes:
            if isinstance(r, dict):
                from services.mission.models import Route

                mp.routes.append(
                    Route(origin=str(r.get("origin") or ""), destination=str(r.get("destination") or ""))
                )
            elif isinstance(r, str) and "->" in r:
                parts = r.split("->", 1)
                from services.mission.models import Route

                mp.routes.append(Route(origin=parts[0].strip(), destination=parts[1].strip()))
        mp.passengers = turn_profile.get("passengers")
        mp.nonstop_required = bool(turn_profile.get("nonstop_required"))
        mp.westbound_sensitive = bool(turn_profile.get("westbound_sensitive"))

        ctx = detect_hard_elimination_context(mp)
        if ctx:
            for m in rec_models:
                reason = hard_elimination_reason(m, ctx)
                if reason:
                    issues.append(f"hard_gate_violation:{m}")
    except Exception:
        pass

    if realism and realism.min_aircraft_class == "ultra-long" and rec_models:
        light_sm = {"Citation CJ2", "Citation CJ4", "Learjet 75", "Citation Latitude"}
        if any(m in light_sm for m in rec_models[:3]):
            issues.append("aircraft_class_below_minimum")

    if issues:
        critical = any(
            x.startswith(("fallback_or_invalid", "hard_gate_violation", "aircraft_class"))
            for x in issues
        )
        return ("FAIL" if critical else "WARN"), issues
    return "PASS", []
