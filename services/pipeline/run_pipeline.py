"""
Advisory pipeline — feasibility filtering MUST run before scoring/ranking.

Flow:
  1. extract_mission_profile
  2. generate_candidate_aircraft_list
  3. evaluate_mission_feasibility (each candidate)
  4. filter feasible == False
  5. score + rank survivors only
  6. explain (no eliminated aircraft in output)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation, detect_models_from_text
from services.state.mission_state import (
    MissionState as PersistentMissionState,
    advance_persistent_mission_state,
    load_persistent_mission_state,
    persist_mission_state_patch,
    persistent_to_mission_profile,
    to_consultant_mission_state,
)
from services.state.mission_validation import validate_mission_state_consistency
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES
from services.graph.aircraft_capability_graph import evaluate_capability_graph
from services.mission.feasibility_engine import FeasibilityResult
from services.mission.memory_bridge import extract_mission_with_memory
from services.mission.models import MissionProfile
from services.recommendation.mission_ranker import MissionCategory, classify_mission_category, rank_missions

logger = logging.getLogger(__name__)


def _mission_constraint_failed(reasons: List[str]) -> str:
    """Map elimination reasons to a primary failed constraint label."""
    blob = " ".join(reasons).lower()
    if "passenger" in blob:
        return "passenger_load"
    if "practical range" in blob or "insufficient for mission" in blob:
        return "route_range"
    if "runway" in blob or "short-field" in blob:
        return "runway_performance"
    if "westbound" in blob or "winter" in blob:
        return "westbound_winter_margin"
    if "mountain" in blob or "hot-high" in blob or "hot/high" in blob:
        return "mountain_airport"
    if "short-field mission" in blob or "overbuy" in blob:
        return "mission_platform_fit"
    return "operational_feasibility"


def _log_feasibility_elimination(
    aircraft_name: str,
    result: FeasibilityResult,
) -> Dict[str, Any]:
    reason = "; ".join(result.elimination_reasons) if result.elimination_reasons else "not_feasible"
    constraint = _mission_constraint_failed(result.elimination_reasons)
    entry = {
        "aircraft_name": aircraft_name,
        "reason": reason,
        "mission_constraint_failed": constraint,
    }
    logger.info(
        "FEASIBILITY_ELIMINATION: aircraft=%s constraint=%s reason=%s",
        aircraft_name,
        constraint,
        reason,
    )
    return entry


@dataclass
class AdvisoryPipelineResult:
    mission_profile: MissionProfile
    mission_state: MissionState
    persistent_mission_state: PersistentMissionState
    mission_category: MissionCategory
    recommendations: List[AircraftRecommendation]
    feasibility_map: Dict[str, FeasibilityResult] = field(default_factory=dict)
    feasible_models: List[str] = field(default_factory=list)
    eliminated_models: List[str] = field(default_factory=list)
    elimination_log: List[Dict[str, Any]] = field(default_factory=dict)
    capability_graph: Dict[str, Any] = field(default_factory=dict)
    mission_validation: Dict[str, Any] = field(default_factory=dict)
    recommendation_audit: Dict[str, Any] = field(default_factory=dict)


def extract_mission_profile(
    query: str,
    *,
    conversation_state: Optional[Dict[str, Any]] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> tuple[MissionProfile, MissionProfile, Any]:
    """Turn-isolated extract with optional memory merge. Returns (turn_only, merged, next_memory)."""
    return extract_mission_with_memory(
        query,
        conversation_state=conversation_state,
        data_used=data_used,
    )


def _resolve_catalog_model(name: str) -> Optional[str]:
    """Map shorthand (G650) to catalog keys (Gulfstream G650)."""
    from services.recommendation.hard_mission_elimination import _MODEL_ALIASES

    key = _MODEL_ALIASES.get(name, name)
    if key in AIRCRAFT_PROFILES:
        return key
    if name in AIRCRAFT_PROFILES:
        return name
    return None


def generate_candidate_aircraft_list(
    query: str,
    *,
    explicit_models: Optional[List[str]] = None,
) -> List[str]:
    """
    Full operational catalog for advisory missions; explicit list only for comparisons.
    """
    if explicit_models:
        resolved = []
        for m in explicit_models:
            key = _resolve_catalog_model(m)
            if key and key not in resolved:
                resolved.append(key)
        return resolved
    mentioned = detect_models_from_text(query or "")
    if mentioned and len(mentioned) >= 2 and any(
        tok in (query or "").lower() for tok in ("compare", " versus ", " vs ", "versus")
    ):
        resolved = []
        for m in mentioned:
            key = _resolve_catalog_model(m)
            if key and key not in resolved:
                resolved.append(key)
        return resolved
    return list(AIRCRAFT_PROFILES.keys())


def filter_candidates_by_feasibility(
    mission_profile: MissionProfile,
    candidates: List[str],
    *,
    override_experimental: bool = False,  # noqa: ARG001 — graph uses hard constraints only
) -> tuple[List[str], Dict[str, FeasibilityResult], List[Dict[str, Any]]]:
    """
    Hard-filter via capability graph — only passing aircraft proceed to scoring.
    """
    graph_result = evaluate_capability_graph(mission_profile, candidates)
    feasible = list(graph_result.feasible_aircraft_list)
    elimination_log: List[Dict[str, Any]] = list(graph_result.filter_log)

    feasibility_map: Dict[str, FeasibilityResult] = {}
    for ex in graph_result.excluded_aircraft_list:
        feasibility_map[ex.model] = FeasibilityResult(
            feasible=False,
            elimination_reasons=[ex.failed_constraint_reason],
            operational_risk_level="eliminated",
        )
    for model in feasible:
        rank = next((r for r in graph_result.ranked_results if r.model == model), None)
        feasibility_map[model] = FeasibilityResult(
            feasible=True,
            practical_range_nm=rank.range_fit * 6000 if rank else 0,
            mission_margin_nm=0,
            operational_risk_level="low",
        )

    from services.recommendation.hard_mission_elimination import apply_hard_mission_elimination

    feasible, hard_eliminated, hard_log, hard_ctx = apply_hard_mission_elimination(
        mission_profile,
        feasible,
        feasibility_map,
    )
    elimination_log.extend(hard_log)
    if hard_ctx is not None:
        elimination_log.append(
            {
                "hard_elimination_rule": hard_ctx.rule_id,
                "summary": hard_ctx.summary,
                "required_route_nm": hard_ctx.required_route_nm,
            }
        )

    return feasible, feasibility_map, elimination_log


def run_advisory_pipeline(
    query: str,
    *,
    conversation_state: Optional[Dict[str, Any]] = None,
    data_used: Optional[Dict[str, Any]] = None,
    mission_profile: Optional[MissionProfile] = None,
    explicit_candidates: Optional[List[str]] = None,
    max_results: int = 3,
    override_experimental: bool = False,
) -> AdvisoryPipelineResult:
    """
    End-to-end advisory path: extract → feasibility filter → rank feasible only.

    Delegates to :func:`services.orchestration.pipeline_orchestrator.run_deterministic_stages`.
    """
    from services.orchestration.modes import orchestration_enabled
    from services.orchestration.pipeline_orchestrator import run_deterministic_stages

    if orchestration_enabled():
        result, _trace = run_deterministic_stages(
            query,
            conversation_state=conversation_state,
            data_used=data_used,
            mission_profile=mission_profile,
            explicit_candidates=explicit_candidates,
            max_results=max_results,
            override_experimental=override_experimental,
        )
        if isinstance(data_used, dict):
            data_used["orchestration"] = _trace.to_dict()
        return result

    # Legacy inline path (orchestration disabled)
    from services.recommendation.fit_policy import recommendation_limit_from_query

    result_limit = recommendation_limit_from_query(query)
    if max_results > result_limit:
        max_results = result_limit

    prior_persistent = load_persistent_mission_state(conversation_state, data_used)
    injected_profile = mission_profile

    turn_profile, merged_mem, _mem = extract_mission_profile(
        query,
        conversation_state=conversation_state,
        data_used=data_used,
    )
    if injected_profile is not None:
        merged_mem = injected_profile

    persistent = advance_persistent_mission_state(prior_persistent, turn_profile, query)
    validation = validate_mission_state_consistency(
        prior_persistent, persistent, turn_profile, query
    )
    mission_profile = persistent_to_mission_profile(persistent)
    if injected_profile is not None:
        mission_profile = injected_profile
    else:
        if merged_mem.home_base:
            mission_profile.home_base = merged_mem.home_base
        if merged_mem.fleet_preferences:
            mission_profile.fleet_preferences = list(merged_mem.fleet_preferences)
    if isinstance(data_used, dict):
        data_used.update(persist_mission_state_patch(persistent))
        data_used["mission_state_validation"] = validation.to_dict()

    if injected_profile is not None:
        from services.mission.adapters import mission_profile_to_state

        mission_state = mission_profile_to_state(mission_profile)
    else:
        mission_state = to_consultant_mission_state(persistent)

    if validation.needs_route_clarification:
        return AdvisoryPipelineResult(
            mission_profile=mission_profile,
            mission_state=mission_state,
            persistent_mission_state=persistent,
            mission_category=classify_mission_category(mission_state),
            recommendations=[],
            mission_validation=validation.to_dict(),
        )
    candidates = generate_candidate_aircraft_list(query, explicit_models=explicit_candidates)

    _cat_gate = None
    try:
        from services.recommendation.aircraft_category_gating import (
            apply_mission_category_gating,
        )

        _cat_gate = apply_mission_category_gating(
            mission_state,
            candidates,
            mission_profile=mission_profile,
        )
        candidates = _cat_gate.candidate_models
        if isinstance(data_used, dict):
            data_used["mission_category_gate"] = _cat_gate.to_dict()
    except Exception as exc:
        logger.warning("mission category gating failed (non-fatal): %s", exc)
        _cat_gate = None

    from services.recommendation.hard_mission_elimination import (
        detect_hard_elimination_context,
        hard_elimination_reason,
        hard_gate_allowlist,
    )

    hard_ctx = detect_hard_elimination_context(mission_profile)
    if hard_ctx is not None:
        allowlist = hard_gate_allowlist(mission_profile) or []
        pre_graph_eliminated = [m for m in candidates if m not in allowlist]
        candidates = allowlist
        for model in pre_graph_eliminated:
            reason = hard_elimination_reason(model, hard_ctx)
            if reason:
                logger.info(
                    "HARD_MISSION_ELIMINATION (pre-graph): aircraft=%s reason=%s",
                    model,
                    reason,
                )

    feasible_models, feasibility_map, elimination_log = filter_candidates_by_feasibility(
        mission_profile,
        candidates,
        override_experimental=override_experimental,
    )
    if _cat_gate is not None:
        from services.recommendation.aircraft_category_gating import (
            category_exclusion_feasibility_results,
        )

        for model, fr in category_exclusion_feasibility_results(_cat_gate).items():
            feasibility_map.setdefault(model, fr)

    if not feasible_models and hard_ctx is None:
        graph_snapshot = evaluate_capability_graph(mission_profile, candidates)
        mv = dict(validation.to_dict() or {})
        mv["no_feasible_aircraft"] = True
        mv["realism_block"] = (
            "No aircraft in the operational catalog can realistically satisfy this mission as stated "
            "(NBAA IFR reserves, realistic payload/baggage, and seasonal/westbound margins)."
        )
        return AdvisoryPipelineResult(
            mission_profile=mission_profile,
            mission_state=mission_state,
            persistent_mission_state=persistent,
            mission_category=classify_mission_category(mission_state),
            recommendations=[],
            feasibility_map=feasibility_map,
            feasible_models=[],
            eliminated_models=list(AIRCRAFT_PROFILES.keys()),
            elimination_log=elimination_log,
            capability_graph=graph_snapshot.to_dict(),
            mission_validation=mv,
        )

    if hard_ctx is not None:
        allowlist = hard_gate_allowlist(mission_profile) or []
        feasible_models = list(allowlist)
        for model in allowlist:
            existing = feasibility_map.get(model)
            if existing and existing.feasible:
                continue
            feasibility_map[model] = FeasibilityResult(
                feasible=True,
                practical_range_nm=float(
                    (AIRCRAFT_PROFILES.get(model) or {}).get("practical_nm") or 0
                ),
                mission_margin_nm=0.0,
                operational_risk_level="high",
                notes=[
                    "Hard ULR mission gate — included for ranking; verify payload and winter westbound margins operationally."
                ],
                required_route_nm=hard_ctx.required_route_nm,
            )

    eliminated_models = [
        m
        for m in generate_candidate_aircraft_list(query, explicit_models=explicit_candidates)
        if m not in feasible_models
    ]

    category, recommendations, rank_feas, selection_audit = rank_missions(
        mission_state,
        candidate_models=feasible_models if feasible_models else None,
        max_results=max_results,
        mission_profile=mission_profile,
        override_experimental=override_experimental,
        conversation_state=conversation_state,
        data_used=data_used,
    )
    if rank_feas:
        for model, fr in rank_feas.items():
            feasibility_map.setdefault(model, fr)

    recommendations = [r for r in recommendations if not r.avoid]

    graph_snapshot = evaluate_capability_graph(mission_profile, candidates)

    from services.recommendation.diversity_controls import merge_elimination_log_with_audit

    elimination_log = merge_elimination_log_with_audit(elimination_log, selection_audit)

    return AdvisoryPipelineResult(
        mission_profile=mission_profile,
        mission_state=mission_state,
        persistent_mission_state=persistent,
        mission_category=category,
        recommendations=recommendations[:max_results],
        feasibility_map=feasibility_map,
        feasible_models=feasible_models,
        eliminated_models=eliminated_models,
        elimination_log=elimination_log,
        capability_graph=graph_snapshot.to_dict(),
        mission_validation=validation.to_dict(),
        recommendation_audit=selection_audit.to_dict() if selection_audit else {},
    )
