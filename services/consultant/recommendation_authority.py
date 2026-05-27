"""
Enforce pipeline authority on user-facing answers — LLM cannot override mission kernel law.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from services.consultant.comparison_engine import StructuredComparison
from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.consultant.route_feasibility import RouteFeasibilityAssessment


def allowed_recommendation_models(
    recommendations: List[AircraftRecommendation],
    *,
    comparison_models: Optional[List[str]] = None,
    hard_excluded: Optional[Set[str]] = None,
) -> Set[str]:
    """Models the narrator may name as recommendations."""
    allowed = {r.model for r in recommendations if not r.avoid}
    if comparison_models:
        allowed.update(comparison_models)
    if hard_excluded:
        allowed -= set(hard_excluded)
    return allowed


def detect_unauthorized_aircraft(
    text: str,
    allowed: Set[str],
) -> List[str]:
    """Models mentioned in prose but not in the pipeline shortlist."""
    if not allowed or not (text or "").strip():
        return []
    try:
        from services.consultant.recommendation_engine import detect_models_from_text
    except Exception:
        return []

    mentioned = detect_models_from_text(text)
    return [m for m in mentioned if m not in allowed]


def reconcile_answer_with_pipeline(
    answer: str,
    *,
    mission: MissionState,
    recommendations: List[AircraftRecommendation],
    route_assessments: Optional[List[RouteFeasibilityAssessment]] = None,
    comparison: Optional[StructuredComparison] = None,
    query: str = "",
    turn_seed: str = "",
    comparison_models: Optional[List[str]] = None,
    hard_excluded: Optional[Set[str]] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> tuple[str, bool]:
    """
    LLM may format prose only. If draft diverges from mission authority kernel, reject merge.
    """
    viable = [r for r in recommendations if not r.avoid]

    try:
        from services.mission.mission_understanding_engine import load_mission_understanding
        from services.mission.mission_authority_kernel import (
            attach_kernel_enforcement_report,
            build_mission_authority_kernel,
            enforce_kernel_authority,
            filter_recommendations_by_kernel,
            load_mission_authority_kernel,
            project_kernel_advisory,
        )

        pkt = load_mission_understanding(data_used)
        if pkt is not None:
            kernel = load_mission_authority_kernel(data_used) or build_mission_authority_kernel(
                mission,
                pkt,
                recommendations=viable,
                query=query,
                data_used=data_used,
                route_certainty_degraded=bool(
                    isinstance(data_used, dict) and data_used.get("route_blocks_ranking")
                ),
            )
            filtered = filter_recommendations_by_kernel(viable, kernel)
            canonical = project_kernel_advisory(kernel, filtered)

            if (answer or "").strip():
                enforced, report = enforce_kernel_authority(
                    answer,
                    kernel,
                    filtered,
                )
                attach_kernel_enforcement_report(data_used, report)
                if report.reject_merge:
                    return enforced, True

            if not (answer or "").strip():
                return canonical, True
    except Exception:
        pass

    if not viable:
        return answer, False

    allowed = allowed_recommendation_models(
        viable,
        comparison_models=comparison_models,
        hard_excluded=hard_excluded,
    )
    unauthorized = detect_unauthorized_aircraft(answer, allowed)

    from services.consultant.response_format_validation import (
        validateResponseFormatting,
    )

    fmt = validateResponseFormatting(answer, recommendations=viable)
    needs_regen = bool(unauthorized) or not fmt.ok

    try:
        from services.telemetry.reasoning_packet_enforcement import (
            enforce_reasoning_packet_authority,
            extract_reasoning_packet,
        )

        packet = extract_reasoning_packet(data_used)
        if packet and (unauthorized or not fmt.ok):
            enforced, pkt_report = enforce_reasoning_packet_authority(
                answer,
                data_used=data_used,
                recommendations=viable,
                mission=mission,
                route_assessments=route_assessments,
                comparison_models=comparison_models,
                query=query,
                turn_seed=turn_seed,
            )
            if pkt_report.regenerated or not pkt_report.ok:
                if isinstance(data_used, dict):
                    data_used["reasoning_packet_enforcement"] = pkt_report.to_dict()
                answer = enforced
                needs_regen = True
    except Exception:
        pass

    if not needs_regen:
        try:
            from services.mission.mission_understanding_engine import load_mission_understanding
            from services.mission.mission_authority_kernel import (
                build_mission_authority_kernel,
                enforce_kernel_authority,
                filter_recommendations_by_kernel,
                load_mission_authority_kernel,
            )

            pkt = load_mission_understanding(data_used)
            if pkt is not None:
                kernel = load_mission_authority_kernel(data_used) or build_mission_authority_kernel(
                    mission,
                    pkt,
                    recommendations=viable,
                    query=query,
                    data_used=data_used,
                )
                answer, _ = enforce_kernel_authority(
                    answer,
                    kernel,
                    filter_recommendations_by_kernel(viable, kernel),
                )
        except Exception:
            pass
        return answer, False

    from services.consultant.broker_advisory_layer import format_broker_advisory_response

    regenerated = format_broker_advisory_response(
        mission,
        viable,
        route_assessments=route_assessments,
        query=query,
        data_used=data_used,
    )
    try:
        from services.telemetry.reasoning_packet_enforcement import (
            enforce_reasoning_packet_authority,
        )

        regenerated, pkt_report = enforce_reasoning_packet_authority(
            regenerated,
            data_used=data_used,
            recommendations=viable,
            mission=mission,
            route_assessments=route_assessments,
            comparison_models=comparison_models,
            query=query,
            turn_seed=turn_seed,
        )
        if isinstance(data_used, dict):
            data_used["reasoning_packet_enforcement"] = pkt_report.to_dict()
    except Exception:
        pass

    return regenerated, True
