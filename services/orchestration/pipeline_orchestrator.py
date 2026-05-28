"""
Final consultant orchestration — deterministic decision path with traced stages.

Pipeline order:
  1. Mission Extraction
  2. Feasibility Engine
  3. Aircraft Filtering
  4. Recommendation Ranking
  5. Broker Narrative Generation
  6. Image Verification
  7. Final Response Formatting

The LLM never determines raw feasibility, overrides hard rejects, or hallucinates
operational capability — it only explains, compares, and advises (see ``constants``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from services.consultant.comparison_engine import StructuredComparison, build_structured_comparison
from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation, detect_models_from_text
from services.consultant.recommendation_engine import _AIRCRAFT_PROFILES
from services.consultant.recommendation_authority import reconcile_answer_with_pipeline
from services.consultant.response_format_validation import ensure_validated_consultant_response
from services.consultant.response_formatter import (
    format_consultant_response,
    format_route_clarification_response,
    sanitize_advisor_output,
    should_use_structured_formatter,
)
from services.consultant.route_feasibility import RouteFeasibilityAssessment, assess_mission_routes
from services.consultant.template_suppression import suppress_templates
from services.consultant.visual_models import build_visual_intelligence_bundle
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES
from services.mission.feasibility_engine import FeasibilityResult
from services.mission.models import MissionProfile
from services.orchestration.constants import DECISION_SOURCE, ORCHESTRATION_STAGES
from services.broker.graceful_degradation import (
    broker_degraded_message,
    degraded_empty_shortlist_guidance,
    ensure_non_empty_answer,
)
from services.orchestration.fail_safe import (
    apply_low_confidence_guidance,
    finalize_trace_confidence,
    safe_stage_fallback,
)
from services.orchestration.image_session import advisory_image_context_patch
from services.orchestration.modes import OrchestrationMode, orchestration_mode
from services.orchestration.tracing import OrchestrationTrace, StageRunner
from services.pipeline.run_pipeline import (
    AdvisoryPipelineResult,
    extract_mission_profile,
    filter_candidates_by_feasibility,
    generate_candidate_aircraft_list,
)
from services.recommendation.mission_ranker import classify_mission_category, rank_missions
from services.state.mission_state import (
    MissionState as PersistentMissionState,
    advance_persistent_mission_state,
    load_persistent_mission_state,
    persist_mission_state_patch,
    persistent_to_mission_profile,
    to_consultant_mission_state,
)
from services.state.mission_validation import validate_mission_state_consistency

logger = logging.getLogger(__name__)


def _finalize_reasoning_packet(
    *,
    data_used: Optional[Dict[str, Any]],
    recommendations: List[AircraftRecommendation],
    elimination_log: List[Dict[str, Any]],
) -> None:
    """Attach immutable ``hye_reasoning_packet`` with fleet audit (P3)."""
    if not isinstance(data_used, dict):
        return
    try:
        from services.telemetry.reasoning_packet import (
            attach_reasoning_packet,
            build_reasoning_packet_from_pipeline,
        )
        from services.telemetry.reasoning_packet_enforcement import validate_packet_fleet_audit

        packet = build_reasoning_packet_from_pipeline(
            data_used=data_used,
            recommendations=recommendations,
            operational_context=data_used.get("mission_operational_context"),
            aircraft_operational=data_used.get("aircraft_operational_assessments"),
            elimination_log=elimination_log,
        )
        attach_reasoning_packet(data_used, packet)
        audit_issues = validate_packet_fleet_audit(packet.to_dict())
        if audit_issues:
            logger.error(
                "FLEET_PACKET_AUDIT inconsistent with pipeline: %s",
                audit_issues[:6],
            )
            data_used["fleet_packet_audit_issues"] = audit_issues
        data_used["hye_reasoning_packet_summary"] = {
            "schema_version": packet.schema_version,
            "presented_models": list(packet.presented_models),
            "fleet_audit_segments": len((packet.fleet_audit or {}).get("segments") or []),
            "elimination_count": len(packet.eliminations),
        }
    except Exception as exc:
        logger.warning("reasoning_packet build failed (non-fatal): %s", exc)


def _sync_recommendations_with_fleet_plan(
    recommendations: List[AircraftRecommendation],
    data_used: Optional[Dict[str, Any]],
) -> List[AircraftRecommendation]:
    """When multi-domain decomposition applies, present fleet primaries only — not stale rank survivors."""
    if not isinstance(data_used, dict):
        return recommendations
    fp = data_used.get("fleet_composition_plan")
    if not isinstance(fp, dict) or not fp.get("multi_aircraft_required"):
        return recommendations
    presented = [m for m in (fp.get("presented_models") or []) if m]
    if not presented:
        if fp.get("single_aircraft_structurally_invalid"):
            return []
        return recommendations
    presented_set = {m.lower() for m in presented}
    return [r for r in recommendations if (r.model or "").lower() in presented_set]


def _attach_fleet_composition_plan(
    *,
    mission_profile: MissionProfile,
    mission_state: MissionState,
    recommendations: List[AircraftRecommendation],
    query: str,
    feasible_models: List[str],
    data_used: Optional[Dict[str, Any]],
    elimination_log: List[Dict[str, Any]],
    feasibility_map: Dict[str, Any],
    eliminated_models: List[str],
) -> List[AircraftRecommendation]:
    """Multi-domain decomposition — runs even when ranking returns empty."""
    try:
        from services.fleet.fleet_composition import (
            build_fleet_composition_plan,
            merge_fleet_into_recommendations,
        )

        existing = None
        if isinstance(data_used, dict):
            raw = data_used.get("fleet_composition_plan")
            if isinstance(raw, dict):
                existing = raw

        pkt_incompatible = False
        if isinstance(data_used, dict):
            pkt = data_used.get("mission_understanding_packet") or {}
            if isinstance(pkt, dict):
                inf = pkt.get("inferred_constraints") or {}
                pkt_incompatible = bool(inf.get("incompatible_mission_bands"))

        if existing and existing.get("multi_aircraft_required") and not pkt_incompatible:
            return recommendations

        fleet_plan = build_fleet_composition_plan(
            mission_profile,
            mission_state,
            recommendations,
            query=query,
            feasible_models=feasible_models,
            data_used=data_used,
            elimination_log=elimination_log,
            feasibility_map=feasibility_map,
            explicit_eliminated=eliminated_models,
        )
        if isinstance(data_used, dict):
            data_used["fleet_composition_plan"] = fleet_plan.to_dict()
        if fleet_plan.multi_aircraft_required:
            recommendations = merge_fleet_into_recommendations(
                list(recommendations), fleet_plan
            )
            if isinstance(data_used, dict):
                data_used["fleet_multi_aircraft"] = True
    except Exception as exc:
        logger.warning("fleet_composition failed (non-fatal): %s", exc)
    return recommendations


@dataclass
class ConsultantOrchestrationResult:
    answer: str
    mission_state: MissionState
    recommendations: List[AircraftRecommendation] = field(default_factory=list)
    aircraft_images: List[Dict[str, Any]] = field(default_factory=list)
    pipeline_result: Optional[AdvisoryPipelineResult] = None
    trace: OrchestrationTrace = field(default_factory=OrchestrationTrace)
    data_used_patch: Dict[str, Any] = field(default_factory=dict)
    route_assessments: List[RouteFeasibilityAssessment] = field(default_factory=list)
    comparison: Optional[StructuredComparison] = None
    low_confidence: bool = False
    authority_block: str = ""


def run_deterministic_stages(
    query: str,
    *,
    conversation_state: Optional[Dict[str, Any]] = None,
    data_used: Optional[Dict[str, Any]] = None,
    mission_profile: Optional[MissionProfile] = None,
    explicit_candidates: Optional[List[str]] = None,
    max_results: int = 3,
    override_experimental: bool = False,
    trace: Optional[OrchestrationTrace] = None,
) -> Tuple[AdvisoryPipelineResult, OrchestrationTrace]:
    """
    Stages 1–4: mission extraction → feasibility → filtering → ranking.

    This is the only code path that may decide which aircraft are feasible and ranked.
    """
    from services.recommendation.fit_policy import recommendation_limit_from_query
    from services.graph.aircraft_capability_graph import evaluate_capability_graph
    from services.recommendation.diversity_controls import merge_elimination_log_with_audit
    from services.recommendation.hard_mission_elimination import (
        detect_hard_elimination_context,
        hard_elimination_reason,
        hard_gate_allowlist,
    )

    tr = trace or OrchestrationTrace(mode=orchestration_mode(), decision_source=DECISION_SOURCE)
    tr.decision_source = DECISION_SOURCE

    from services.preprocessing import attach_mission_preprocessing

    pre = attach_mission_preprocessing(data_used, query)
    tr.record(
        "mission_preprocessing",
        "ok",
        details={
            "origin": pre.origin if pre.origin != "UNKNOWN" else None,
            "destination": pre.destination if pre.destination != "UNKNOWN" else None,
            "route_evidence": pre.route_evidence,
        },
    )

    result_limit = recommendation_limit_from_query(query)
    if max_results > result_limit:
        max_results = result_limit

    prior_persistent: PersistentMissionState
    mission_profile_out: MissionProfile
    persistent: PersistentMissionState
    validation: Any
    mission_state: MissionState

    # --- Stage 1: Mission Extraction ---
    with StageRunner(tr, ORCHESTRATION_STAGES[0]) as s1:
        prior_persistent = load_persistent_mission_state(conversation_state, data_used)
        turn_profile, merged_mem, _mem = extract_mission_profile(
            query,
            conversation_state=conversation_state,
            data_used=data_used,
        )
        if mission_profile is not None:
            merged_mem = mission_profile

        persistent = advance_persistent_mission_state(prior_persistent, turn_profile, query)
        validation = validate_mission_state_consistency(
            prior_persistent, persistent, turn_profile, query
        )
        from services.mission.adapters import mission_profile_to_state
        from services.state.session_mission_memory import merge_turn_with_session

        session_merge = merge_turn_with_session(turn_profile, persistent, query)
        if mission_profile is not None:
            mission_profile_out = mission_profile
        else:
            mission_profile_out = session_merge.profile

        if merged_mem.fleet_preferences:
            mission_profile_out.fleet_preferences = list(
                dict.fromkeys(
                    list(mission_profile_out.fleet_preferences)
                    + list(merged_mem.fleet_preferences)
                )
            )

        mission_state = mission_profile_to_state(mission_profile_out)

        try:
            from services.mission.mission_profile_inference import infer_mission_profile
            from services.mission.mission_understanding_engine import (
                apply_understanding_to_mission_state,
                apply_understanding_to_profile,
                attach_mission_understanding,
                build_mission_understanding,
                build_understanding_authority_block,
            )
            from services.session.broker_memory import (
                load_broker_memory,
                save_broker_memory,
                update_broker_memory_from_turn,
                update_broker_memory_from_understanding,
            )

            broker_mem = load_broker_memory(data_used if isinstance(data_used, dict) else None)

            from services.mission.mission_context_reconciliation import (
                assess_mission_continuity,
                reconcile_broker_memory_for_turn,
            )
            from services.mission.mission_operational_graph import load_operational_graph

            prior_graph = load_operational_graph(
                data_used if isinstance(data_used, dict) else None
            )
            continuity = assess_mission_continuity(
                query,
                mission_profile_out,
                broker_memory=broker_mem.to_dict(),
                prior_graph=prior_graph,
            )
            if isinstance(data_used, dict):
                data_used["mission_continuity_assessment"] = continuity.to_dict()

            inf_mem = broker_mem.to_dict()
            if continuity.mission_pivot:
                inf_mem = {
                    k: inf_mem[k]
                    for k in ("nonstop_preference",)
                    if inf_mem.get(k)
                }

            inferred = infer_mission_profile(
                query,
                mission_profile_out,
                broker_memory=inf_mem or None,
            )

            existing_understanding = None
            if isinstance(data_used, dict) and not continuity.mission_pivot:
                try:
                    from services.mission.mission_understanding_engine import (
                        load_mission_understanding,
                    )

                    existing_understanding = load_mission_understanding(data_used)
                except Exception:
                    existing_understanding = None

            understanding_packet = existing_understanding
            built_in_this_stage = understanding_packet is None
            if built_in_this_stage or continuity.mission_pivot:
                _hist = None
                if isinstance(conversation_state, dict):
                    _hist = conversation_state.get("history")

                understanding_packet = build_mission_understanding(
                    query,
                    mission_profile_out,
                    mission_state,
                    broker_memory=broker_mem.to_dict(),
                    history=_hist if isinstance(_hist, list) else None,
                    inferred=inferred,
                    data_used=data_used if isinstance(data_used, dict) else None,
                )
            mission_profile_out = apply_understanding_to_profile(
                mission_profile_out,
                understanding_packet,
                inferred=inferred,
            )
            mission_state = apply_understanding_to_mission_state(
                mission_state, understanding_packet
            )

            # If we preserved an earlier packet (likely built from full history),
            # refresh the recommendation gate using the merged mission snapshot
            # computed for this orchestration turn.
            try:
                from services.mission.mission_understanding_engine import (
                    refresh_mission_understanding_gate,
                )

                understanding_packet = refresh_mission_understanding_gate(
                    understanding_packet,
                    mission_profile_out,
                    mission_state,
                    inferred_confidence=understanding_packet.confidence_scores.get(
                        "profile_inference"
                    )
                    if hasattr(understanding_packet, "confidence_scores")
                    else None,
                )
            except Exception:
                pass

            try:
                from services.mission.mission_operational_graph import (
                    load_operational_graph,
                    save_operational_graph,
                    stabilize_mission_understanding,
                )

                understanding_packet, merged_graph = stabilize_mission_understanding(
                    understanding_packet,
                    query=query,
                    profile=mission_profile_out,
                    mission=mission_state,
                    broker_memory=broker_mem.to_dict(),
                    prior_graph=prior_graph if continuity.apply_structural_memory else None,
                    continuity=continuity,
                )
                mission_profile_out = apply_understanding_to_profile(
                    mission_profile_out,
                    understanding_packet,
                    inferred=inferred,
                )
                mission_state = apply_understanding_to_mission_state(
                    mission_state, understanding_packet
                )
                try:
                    from services.mission.pre_ranking_representation import (
                        apply_pre_ranking_representation,
                    )

                    mission_profile_out, mission_state, understanding_packet = (
                        apply_pre_ranking_representation(
                            query,
                            mission_profile_out,
                            mission_state,
                            understanding_packet,
                            data_used if isinstance(data_used, dict) else None,
                        )
                    )
                    mission_state = apply_understanding_to_mission_state(
                        mission_state, understanding_packet
                    )
                except Exception:
                    pass
                if isinstance(data_used, dict):
                    save_operational_graph(data_used, merged_graph)
                    attach_mission_understanding(data_used, understanding_packet)
                    try:
                        from services.mission.structural_decomposition import (
                            needs_structural_decomposition,
                        )
                        from services.mission.mission_graph import (
                            build_mission_graph,
                            save_mission_graph,
                        )

                        _proof = needs_structural_decomposition(
                            understanding_packet,
                            profile=mission_profile_out,
                            mission=mission_state,
                            query=query,
                            data_used=data_used,
                        )
                        _graph = build_mission_graph(
                            understanding_packet,
                            mission_profile_out,
                            mission_state,
                            structural_incompatibility=_proof.required,
                            query=query,
                        )
                        from services.mission.phase2_structural_synthesis import (
                            apply_phase2_structural_synthesis,
                        )

                        _graph, _, _, _ = apply_phase2_structural_synthesis(
                            _graph,
                            understanding_packet,
                            mission_profile_out,
                            mission_state,
                            query=query,
                            data_used=data_used,
                        )
                        save_mission_graph(data_used, _graph)
                        data_used["fleet_strategy_required"] = bool(_proof.required)
                    except Exception:
                        pass
                    data_used["mission_understanding_authority"] = (
                        build_understanding_authority_block(understanding_packet)
                    )
                    data_used["pre_llm_mission_understanding"] = 1
            except Exception:
                pass

            route_label = (
                mission_profile_out.route_labels()[0]
                if mission_profile_out.routes
                else None
            )
            broker_mem = update_broker_memory_from_turn(
                broker_mem,
                route=route_label,
                inferred_profile=inferred.to_dict(),
                mission_style=inferred.utilization_style,
            )
            broker_mem = reconcile_broker_memory_for_turn(
                broker_mem, understanding_packet, continuity
            )
            if isinstance(data_used, dict):
                save_broker_memory(data_used, broker_mem)
                data_used["inferred_mission_profile"] = inferred.to_dict()
            s1.details["mission_understanding_confidence"] = round(
                understanding_packet.overall_confidence, 3
            )
            s1.details["corridor_type"] = understanding_packet.corridor_type
        except Exception:
            pass

        if isinstance(data_used, dict):
            data_used.update(persist_mission_state_patch(persistent))
            data_used["mission_state_validation"] = validation.to_dict()
            data_used["session_mission_memory"] = {
                **session_merge.to_dict(),
                "inherited_fields": list(
                    dict.fromkeys(
                        list(session_merge.inherited_fields)
                        + list(validation.inherited_fields)
                    )
                ),
            }
        s1.details = {
            "routes": len(mission_state.routes or []),
            "passengers": mission_state.passenger_count,
            "needs_route_clarification": bool(validation.needs_route_clarification),
        }
        if validation.needs_route_clarification:
            s1.confidence = 0.72

    if validation.needs_route_clarification:
        from services.state.mission_state import record_clarification_question_asked

        persistent = record_clarification_question_asked(persistent)
        if isinstance(data_used, dict):
            data_used.update(persist_mission_state_patch(persistent))
        with StageRunner(tr, ORCHESTRATION_STAGES[1]) as s2:
            s2.skip("route_clarification_required")
        with StageRunner(tr, ORCHESTRATION_STAGES[2]) as s3:
            s3.skip("route_clarification_required")
        with StageRunner(tr, ORCHESTRATION_STAGES[3]) as s4:
            s4.skip("route_clarification_required")
        result = AdvisoryPipelineResult(
            mission_profile=mission_profile_out,
            mission_state=mission_state,
            persistent_mission_state=persistent,
            mission_category=classify_mission_category(mission_state),
            recommendations=[],
            mission_validation=validation.to_dict(),
        )
        finalize_trace_confidence(tr, result)
        return result, tr

    candidates: List[str] = []
    feasible_models: List[str] = []
    feasibility_map: Dict[str, FeasibilityResult] = {}
    elimination_log: List[Dict[str, Any]] = []
    hard_ctx = None
    route_blocks_ranking = False

    # --- Stage 2: Feasibility Engine ---
    with StageRunner(tr, ORCHESTRATION_STAGES[1]) as s2:
        try:
            from services.mission.route_distance_authority import mission_route_blocks_ranking

            route_blocks_ranking, route_auth = mission_route_blocks_ranking(
                mission_profile_out.route_labels()
            )
            if isinstance(data_used, dict):
                data_used["route_distance_authority"] = [r.to_dict() for r in route_auth]
                data_used["route_blocks_ranking"] = route_blocks_ranking
            s2.details["route_blocks_ranking"] = route_blocks_ranking
            if route_blocks_ranking:
                s2.confidence = 0.38
        except Exception as exc:
            logger.warning("route distance authority failed (non-fatal): %s", exc)

        try:
            from services.aircraft_feasibility import validate_mission_route_realism

            route_realism = validate_mission_route_realism(mission_profile_out)
            if isinstance(data_used, dict):
                data_used["route_realism"] = route_realism.to_dict()
            s2.details["route_realism"] = route_realism.to_dict()
        except Exception as exc:
            logger.warning("route realism validation failed (non-fatal): %s", exc)

        candidates = generate_candidate_aircraft_list(
            query, explicit_models=explicit_candidates
        )
        cat_gate = None
        try:
            from services.recommendation.aircraft_category_gating import (
                apply_mission_category_gating,
                category_exclusion_feasibility_results,
            )

            cat_gate = apply_mission_category_gating(
                mission_state,
                candidates,
                mission_profile=mission_profile_out,
            )
            feasibility_map.update(category_exclusion_feasibility_results(cat_gate))
            elimination_log.extend(cat_gate.exclusion_log)
            candidates = cat_gate.candidate_models
            if isinstance(data_used, dict):
                data_used["mission_category_gate"] = cat_gate.to_dict()
            s2.details["mission_category_gate"] = cat_gate.category.value
        except Exception as exc:
            logger.warning("mission category gating failed (non-fatal): %s", exc)
        hard_ctx = detect_hard_elimination_context(mission_profile_out)
        if hard_ctx is not None:
            allowlist = hard_gate_allowlist(mission_profile_out) or []
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

        graph_result = evaluate_capability_graph(mission_profile_out, candidates)
        feasible_models = list(graph_result.feasible_aircraft_list)
        elimination_log.extend(graph_result.filter_log)
        for ex in graph_result.excluded_aircraft_list:
            feasibility_map[ex.model] = FeasibilityResult(
                feasible=False,
                elimination_reasons=[ex.failed_constraint_reason],
                operational_risk_level="eliminated",
            )
        for model in feasible_models:
            rank = next((r for r in graph_result.ranked_results if r.model == model), None)
            feasibility_map[model] = FeasibilityResult(
                feasible=True,
                practical_range_nm=rank.range_fit * 6000 if rank else 0,
                mission_margin_nm=0,
                operational_risk_level="low",
            )

        # HACK v1 — hard aviation constraint kernel (authoritative; no downstream override)
        try:
            from services.recommendation.hack_v1_constraint_kernel import (
                HACK_V1_EMPTY_MESSAGE,
                apply_hack_v1_gate,
            )

            pre_hack = list(feasible_models)
            feasible_models, hack_v1 = apply_hack_v1_gate(
                mission_profile_out,
                feasible_models,
                all_candidates=candidates,
                query=query,
                mission_state=mission_state,
                data_used=data_used,
            )
            for rej in hack_v1.rejection_log:
                feasibility_map[rej.model] = FeasibilityResult(
                    feasible=False,
                    elimination_reasons=[rej.reason],
                    operational_risk_level="eliminated",
                    notes=[f"HACK v1 [{rej.rule_id}]"],
                )
                if rej.model in pre_hack:
                    elimination_log.append(
                        {
                            "model": rej.model,
                            "rule": rej.rule_id,
                            "source": "hack_v1",
                            "reason": rej.reason,
                        }
                    )
            s2.details["hack_v1_feasible"] = len(feasible_models)
            s2.details["hack_v1_rejected"] = len(hack_v1.rejection_log)
            if hack_v1.constraint_empty:
                s2.details["hack_v1_constraint_empty"] = True
                if isinstance(data_used, dict):
                    data_used["hack_v1_realism_block"] = HACK_V1_EMPTY_MESSAGE
        except Exception as exc:
            logger.warning("HACK v1 constraint kernel failed (non-fatal): %s", exc)

        s2.details = {
            "candidate_count": len(candidates),
            "feasible_after_graph": len(feasible_models),
            "eliminated_count": len(graph_result.excluded_aircraft_list),
        }
        if not feasible_models and hard_ctx is None:
            s2.confidence = 0.45

    # --- Stage 3: Aircraft Filtering ---
    with StageRunner(tr, ORCHESTRATION_STAGES[2]) as s3:
        from services.recommendation.hard_mission_elimination import apply_hard_mission_elimination

        feasible_models, _hard_eliminated, hard_log, hard_ctx_filter = apply_hard_mission_elimination(
            mission_profile_out,
            feasible_models,
            feasibility_map,
        )
        elimination_log.extend(hard_log)
        if hard_ctx_filter is not None:
            elimination_log.append(
                {
                    "hard_elimination_rule": hard_ctx_filter.rule_id,
                    "summary": hard_ctx_filter.summary,
                    "required_route_nm": hard_ctx_filter.required_route_nm,
                }
            )

        if not feasible_models and hard_ctx is None:
            hack_empty = bool(
                isinstance(data_used, dict) and data_used.get("hack_v1_constraint_empty")
            )
            realism = (
                str(data_used.get("hack_v1_realism_block") or "")
                if isinstance(data_used, dict)
                else ""
            )
            graph_snapshot = evaluate_capability_graph(mission_profile_out, candidates)
            early_eliminated = [
                m
                for m in generate_candidate_aircraft_list(
                    query, explicit_models=explicit_candidates
                )
                if m not in feasible_models
            ]
            mv = dict(validation.to_dict() or {})
            mv["no_feasible_aircraft"] = True
            if hack_empty and realism:
                mv["realism_block"] = realism
                mv["hack_v1_constraint_empty"] = True
            else:
                mv["realism_block"] = (
                    "No aircraft in the operational catalog can realistically satisfy this mission as stated "
                    "(NBAA IFR reserves, realistic payload/baggage, and seasonal/westbound margins)."
                )
            recommendations: List[AircraftRecommendation] = []
            recommendations = _attach_fleet_composition_plan(
                mission_profile=mission_profile_out,
                mission_state=mission_state,
                recommendations=[],
                query=query,
                feasible_models=feasible_models,
                data_used=data_used,
                elimination_log=elimination_log,
                feasibility_map=feasibility_map,
                eliminated_models=early_eliminated,
            )
            recommendations = _sync_recommendations_with_fleet_plan(recommendations, data_used)
            if isinstance(data_used, dict) and data_used.get("fleet_multi_aircraft"):
                mv["multi_domain_operational_decomposition"] = True
                mv["no_feasible_aircraft"] = False
                mv.pop("realism_block", None)
            s3.degraded("no_feasible_aircraft", confidence=0.42)
            with StageRunner(tr, ORCHESTRATION_STAGES[3]) as s4:
                s4.skip("no_global_feasible_survivor")
                if data_used.get("fleet_composition_plan") if isinstance(data_used, dict) else None:
                    s4.details = {"fleet_decomposition": True}
            _finalize_reasoning_packet(
                data_used=data_used,
                recommendations=recommendations,
                elimination_log=elimination_log,
            )
            result = AdvisoryPipelineResult(
                mission_profile=mission_profile_out,
                mission_state=mission_state,
                persistent_mission_state=persistent,
                mission_category=classify_mission_category(mission_state),
                recommendations=recommendations,
                feasibility_map=feasibility_map,
                feasible_models=[],
                eliminated_models=early_eliminated,
                elimination_log=elimination_log,
                capability_graph=graph_snapshot.to_dict(),
                mission_validation=mv,
            )
            finalize_trace_confidence(tr, result)
            return result, tr

        if hard_ctx is not None:
            allowlist = hard_gate_allowlist(mission_profile_out) or []
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

        s3.details = {
            "feasible_count": len(feasible_models),
            "hard_gate": hard_ctx.rule_id if hard_ctx else None,
        }

    eliminated_models = [
        m
        for m in generate_candidate_aircraft_list(query, explicit_models=explicit_candidates)
        if m not in feasible_models
    ]

    fleet_doctrine_defer = False
    ranking_mission = mission_state
    ranking_profile = mission_profile_out
    if isinstance(data_used, dict):
        try:
            from services.mission.mission_understanding_engine import load_mission_understanding
            from services.mission.mission_operational_graph import (
                requires_fleet_decomposition_before_ranking,
                should_defer_ranking_to_fleet,
            )
            from services.mission.mission_ranking_projection import build_ranking_mission_snapshot

            _under_pkt = load_mission_understanding(data_used)
            if _under_pkt is not None:
                ranking_mission, ranking_profile, proj_trace = build_ranking_mission_snapshot(
                    mission_state,
                    _under_pkt,
                    mission_profile_out,
                )
                data_used["ranking_projection_trace"] = proj_trace.to_dict()
                if route_blocks_ranking:
                    data_used["route_blocks_ranking"] = True

            if requires_fleet_decomposition_before_ranking(
                _under_pkt,
                data_used,
                profile=mission_profile_out,
                mission=mission_state,
                query=query,
                feasible_models=feasible_models,
            ) and not data_used.get("fleet_composition_plan"):
                _attach_fleet_composition_plan(
                    mission_profile=mission_profile_out,
                    mission_state=mission_state,
                    recommendations=[],
                    query=query,
                    feasible_models=feasible_models,
                    data_used=data_used,
                    elimination_log=elimination_log,
                    feasibility_map=feasibility_map,
                    eliminated_models=eliminated_models,
                )

            fleet_doctrine_defer = should_defer_ranking_to_fleet(
                _under_pkt,
                data_used,
                profile=mission_profile_out,
                mission=mission_state,
                query=query,
                feasible_models=feasible_models,
            )
            if fleet_doctrine_defer:
                if not data_used.get("fleet_composition_plan"):
                    _attach_fleet_composition_plan(
                        mission_profile=mission_profile_out,
                        mission_state=mission_state,
                        recommendations=[],
                        query=query,
                        feasible_models=feasible_models,
                        data_used=data_used,
                        elimination_log=elimination_log,
                        feasibility_map=feasibility_map,
                        eliminated_models=eliminated_models,
                    )
                data_used["fleet_doctrine_lock"] = True
                data_used["ranking_defer_to_fleet"] = True
        except Exception:
            fleet_doctrine_defer = False

    # --- Stage 4: Recommendation Ranking ---
    recommendations: List[AircraftRecommendation] = []
    category = classify_mission_category(ranking_mission)
    selection_audit = None

    with StageRunner(tr, ORCHESTRATION_STAGES[3]) as s4:
        if route_blocks_ranking and mission_profile_out.routes:
            s4.skip("route_distance_unresolved")
            mv = dict(validation.to_dict() or {})
            mv["route_blocks_ranking"] = True
            mv["realism_block"] = (
                "Route stage length is not verified — ranked aircraft recommendations "
                "are blocked until origin and destination resolve to catalog or geodesic distance."
            )
            recommendations = _attach_fleet_composition_plan(
                mission_profile=mission_profile_out,
                mission_state=mission_state,
                recommendations=[],
                query=query,
                feasible_models=feasible_models,
                data_used=data_used,
                elimination_log=elimination_log,
                feasibility_map=feasibility_map,
                eliminated_models=eliminated_models,
            )
            recommendations = _sync_recommendations_with_fleet_plan(recommendations, data_used)
            mv["synthesis_first_required"] = True
            mv["route_certainty_degraded"] = True
            result = AdvisoryPipelineResult(
                mission_profile=mission_profile_out,
                mission_state=mission_state,
                persistent_mission_state=persistent,
                mission_category=category,
                recommendations=recommendations,
                feasibility_map=feasibility_map,
                feasible_models=feasible_models,
                eliminated_models=eliminated_models,
                elimination_log=elimination_log,
                mission_validation=mv,
            )
            finalize_trace_confidence(tr, result)
            return result, tr

        defer_ranking = fleet_doctrine_defer or bool(
            isinstance(data_used, dict) and data_used.get("ranking_defer_to_fleet")
        )
        if defer_ranking:
            from services.fleet.fleet_composition import recommendations_from_fleet_plan

            fleet_recs = recommendations_from_fleet_plan(
                data_used, mission=mission_state
            )
            if fleet_recs:
                recommendations = fleet_recs
                rank_feas = {}
                selection_audit = None
                s4.details["ranking_deferred"] = "fleet_doctrine"
            else:
                category, recommendations, rank_feas, selection_audit = rank_missions(
                    ranking_mission,
                    candidate_models=feasible_models if feasible_models else None,
                    max_results=max_results,
                    mission_profile=ranking_profile or mission_profile_out,
                    override_experimental=override_experimental,
                    conversation_state=conversation_state,
                    data_used=data_used,
                    query=query,
                )
        else:
            category, recommendations, rank_feas, selection_audit = rank_missions(
                ranking_mission,
                candidate_models=feasible_models if feasible_models else None,
                max_results=max_results,
                mission_profile=ranking_profile or mission_profile_out,
                override_experimental=override_experimental,
                conversation_state=conversation_state,
                data_used=data_used,
                query=query,
            )
        if rank_feas:
            from services.elimination.elimination_invariant import merge_feasibility_maps

            feasibility_map = merge_feasibility_maps(feasibility_map, rank_feas)
        from services.elimination.elimination_invariant import (
            collect_eliminated_models,
            enforce_elimination_invariant,
        )

        eliminated_set = collect_eliminated_models(
            data_used=data_used,
            elimination_log=elimination_log,
            feasibility_map=feasibility_map,
            explicit_eliminated=eliminated_models,
        )
        pre_rank = [r.model for r in recommendations if not r.avoid]
        recommendations = enforce_elimination_invariant(
            [r for r in recommendations if not r.avoid],
            eliminated_set,
            context="pipeline_ranking",
        )
        try:
            from services.recommendation.hack_v1_constraint_kernel import (
                hack_v1_permanent_exclusions,
            )

            hack_excl = hack_v1_permanent_exclusions(data_used)
            if hack_excl:
                recommendations = [r for r in recommendations if r.model not in hack_excl]
        except Exception:
            pass
        stripped = [m for m in pre_rank if m not in {r.model for r in recommendations}]
        if stripped:
            s4.details["elimination_stripped_from_ranking"] = stripped

        recommendations = _attach_fleet_composition_plan(
            mission_profile=mission_profile_out,
            mission_state=mission_state,
            recommendations=recommendations,
            query=query,
            feasible_models=feasible_models,
            data_used=data_used,
            elimination_log=elimination_log,
            feasibility_map=feasibility_map,
            eliminated_models=eliminated_models,
        )
        recommendations = _sync_recommendations_with_fleet_plan(recommendations, data_used)

        s4.details = {
            "ranked_models": [r.model for r in recommendations],
            "feasible_count": len(feasible_models),
            "elimination_invariant_enforced": True,
        }
        try:
            from services.recommendation.clarification_decision import (
                recommendation_confidence_sufficient,
            )

            s4.confidence = 0.88 if recommendation_confidence_sufficient(recommendations) else 0.65
        except Exception:
            s4.confidence = 0.65

    graph_snapshot = evaluate_capability_graph(mission_profile_out, candidates)
    elimination_log = merge_elimination_log_with_audit(
        elimination_log, selection_audit
    )

    if isinstance(data_used, dict) and not data_used.get("fleet_composition_plan"):
        recommendations = _attach_fleet_composition_plan(
            mission_profile=mission_profile_out,
            mission_state=mission_state,
            recommendations=recommendations,
            query=query,
            feasible_models=feasible_models,
            data_used=data_used,
            elimination_log=elimination_log,
            feasibility_map=feasibility_map,
            eliminated_models=eliminated_models,
        )
        recommendations = _sync_recommendations_with_fleet_plan(recommendations, data_used)

    _finalize_reasoning_packet(
        data_used=data_used,
        recommendations=recommendations,
        elimination_log=elimination_log,
    )
    if isinstance(data_used, dict):
        pkt_summary = data_used.get("hye_reasoning_packet_summary")
        if isinstance(pkt_summary, dict):
            tr.record(
                "reasoning_packet",
                "ok",
                details=dict(pkt_summary),
            )
        audit_issues = data_used.get("fleet_packet_audit_issues")
        if isinstance(audit_issues, list) and audit_issues:
            tr.record(
                "fleet_packet_audit",
                "degraded",
                confidence=0.4,
                details={"issues": audit_issues[:6]},
            )

    result = AdvisoryPipelineResult(
        mission_profile=mission_profile_out,
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
    finalize_trace_confidence(tr, result, recommendations=recommendations)
    return result, tr


def _build_route_assessments(
    mission: MissionState,
    recommendations: List[AircraftRecommendation],
) -> List[RouteFeasibilityAssessment]:
    if not mission.routes or not recommendations:
        return []
    top = recommendations[0]
    prof = _AIRCRAFT_PROFILES.get(top.model) or {}
    route_assessments = assess_mission_routes(
        mission,
        aircraft_practical_nm=float(prof.get("practical_nm") or 3000),
        aircraft_brochure_nm=float(prof.get("brochure_nm") or 3500),
    )
    if len(recommendations) > 1 and len(mission.routes) > 1:
        route_assessments = []
        for rec in recommendations[:3]:
            p = _AIRCRAFT_PROFILES.get(rec.model) or {}
            route_assessments.extend(
                assess_mission_routes(
                    mission,
                    aircraft_practical_nm=float(p.get("practical_nm") or 3000),
                    aircraft_brochure_nm=float(p.get("brochure_nm") or 3500),
                )[: len(mission.routes)]
            )
    return route_assessments


def run_consultant_orchestration(
    query: str,
    *,
    llm_draft: str = "",
    conversation_state: Optional[Dict[str, Any]] = None,
    data_used: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    explicit_candidates: Optional[List[str]] = None,
    max_results: int = 3,
    query_intent: str = "",
    tavily_payload: Optional[Dict[str, Any]] = None,
    phly_rows: Optional[List[Dict[str, Any]]] = None,
    use_structured_formatter: Optional[bool] = None,
) -> ConsultantOrchestrationResult:
    """
    Full orchestration (stages 1–7) for mission advisory turns.

    Deterministic stages own feasibility and ranking; stages 5–7 format output.
    """
    from services.consultant.broker_advisory_layer import format_broker_advisory_response
    from services.consultant.llm_explanation_layer import build_pipeline_authority_block
    from services.recommendation.query_recommendation_intent import (
        QueryRecommendationIntent,
        apply_query_intent_metadata,
        classify_query_recommendation_intent,
        requires_ranked_aircraft_pipeline,
    )

    du: Dict[str, Any] = dict(data_used) if isinstance(data_used, dict) else {}
    du.update(advisory_image_context_patch(query))
    tr = OrchestrationTrace(mode=orchestration_mode(), decision_source=DECISION_SOURCE)
    patch: Dict[str, Any] = {"orchestration": tr.to_dict(), "recommendation_decision_source": DECISION_SOURCE}
    patch.update(advisory_image_context_patch(query))

    from services.preprocessing import attach_mission_preprocessing

    attach_mission_preprocessing(du, query)
    if isinstance(data_used, dict):
        data_used.update(
            {
                k: du[k]
                for k in ("mission_preprocessing", "mission_preprocessing_json", "mission_preprocessing_meta")
                if k in du
            }
        )

    from services.orchestration.orchestration_router_v2 import (
        OrchestrationQueryTypeV2,
        OrchestrationRendererV2,
        OrchestrationRouterV2Result,
        apply_orchestration_v2_metadata,
        orchestration_v2_locked_comparison_models,
        route_orchestration_v2,
    )

    v2_route = route_orchestration_v2(query, history=history)
    try:
        from services.orchestration.orchestration_stabilization import OrchestrationStabilizer

        _stab = OrchestrationStabilizer().stabilize(query, v2_route)
        v2_route = _stab.route
        du.update(_stab.to_patch())
        patch.update(_stab.to_patch())
    except Exception:
        pass
    apply_orchestration_v2_metadata(du, v2_route)
    apply_orchestration_v2_metadata(patch, v2_route)

    qri = classify_query_recommendation_intent(query, history=history)
    if query_intent:
        try:
            qri.intent = QueryRecommendationIntent(query_intent)
            qri.requires_ranked_pipeline = requires_ranked_aircraft_pipeline(qri.intent)
        except ValueError:
            pass
    apply_query_intent_metadata(patch, qri)
    apply_query_intent_metadata(du, qri)

    from services.orchestration.response_mode_classifier import (
        apply_orchestration_response_mode_metadata,
        classify_orchestration_response_mode,
    )

    orm = classify_orchestration_response_mode(query, history=history)
    # V2 router is authoritative — sync legacy response mode flags.
    if v2_route.query_type == OrchestrationQueryTypeV2.EXPLICIT_COMPARISON:
        from services.orchestration.response_mode_classifier import OrchestrationResponseMode

        orm.mode = OrchestrationResponseMode.COMPARISON_MODE
        orm.suppresses_aircraft_recommendations = False
        orm.explicit_aircraft_request = True
    elif v2_route.query_type == OrchestrationQueryTypeV2.RECOMMENDATION_REQUEST:
        from services.orchestration.response_mode_classifier import OrchestrationResponseMode

        orm.mode = OrchestrationResponseMode.RECOMMENDATION_MODE
        orm.suppresses_aircraft_recommendations = False
        orm.explicit_aircraft_request = True
    elif v2_route.query_type in (
        OrchestrationQueryTypeV2.STRATEGIC_FLEET_ANALYSIS,
        OrchestrationQueryTypeV2.NETWORK_STRUCTURE,
    ):
        from services.orchestration.response_mode_classifier import OrchestrationResponseMode

        orm.mode = OrchestrationResponseMode.STRUCTURE_MODE
        orm.suppresses_aircraft_recommendations = not v2_route.allow_recommendation_ranking
        orm.structural_first = True
    elif v2_route.query_type == OrchestrationQueryTypeV2.NAMED_AIRCRAFT_CAPABILITY:
        orm.suppresses_aircraft_recommendations = True
        orm.structural_first = False

    apply_orchestration_response_mode_metadata(patch, orm)
    apply_orchestration_response_mode_metadata(du, orm)
    if orm.suppresses_aircraft_recommendations or v2_route.suppress_generic_shortlist:
        qri.allows_acquisition_framing = False
        if not v2_route.allow_recommendation_ranking:
            qri.requires_ranked_pipeline = v2_route.requires_deterministic_pipeline
    else:
        if isinstance(du, dict):
            du.pop("orchestration_suppresses_aircraft", None)
            du["defer_global_shortlist"] = False

    if (
        not qri.requires_ranked_pipeline
        and v2_route.renderer != OrchestrationRendererV2.OWNERSHIP_ECONOMICS
    ):
        tr.record("mission_extraction", "skipped", details={"reason": qri.intent.value})
        for stage in ORCHESTRATION_STAGES[1:]:
            tr.record(stage, "skipped", details={"reason": "non_ranked_intent"})
        answer = (llm_draft or "").strip()
        if qri.intent == QueryRecommendationIntent.OWNERSHIP_ECONOMICS:
            from services.orchestration.ownership_economics import format_ownership_economics_response
            from services.mission.adapters import mission_profile_to_state
            from services.pipeline.run_pipeline import extract_mission_profile

            turn_profile, _, _ = extract_mission_profile(
                query,
                conversation_state=conversation_state,
                data_used=du,
            )
            mission_state_own = mission_profile_to_state(turn_profile)
            answer = format_ownership_economics_response(query, mission=mission_state_own)
            tr.record("ownership_economics", "ok", details={"mode": "dedicated_branch"})
        conf = finalize_trace_confidence(tr, None)
        if not answer.strip():
            from services.orchestration.ownership_economics import format_ownership_economics_response

            answer = format_ownership_economics_response(query)
        answer, low = apply_low_confidence_guidance(answer, conf)
        tr.low_confidence = low
        patch["orchestration"] = tr.to_dict()
        return ConsultantOrchestrationResult(
            answer=answer,
            mission_state=MissionState(),
            trace=tr,
            data_used_patch=patch,
            low_confidence=low,
        )

    pipeline, tr = run_deterministic_stages(
        query,
        conversation_state=conversation_state,
        data_used=du,
        explicit_candidates=explicit_candidates,
        max_results=max_results,
        trace=tr,
    )
    if isinstance(du, dict) and isinstance(du.get("mission_continuity_assessment"), dict):
        patch["mission_continuity_assessment"] = dict(du["mission_continuity_assessment"])
    # Persist broker memory + operational graph explicitly via patch (caller-applied state).
    for k in (
        "hye_broker_memory",
        "mission_operational_graph",
        "pre_ranking_representation",
        "pre_ranking_applied",
        "mission_route_graph",
        "mission_governance",
        "industrial_airport_profile",
        "mission_segment_graph",
        "structural_decomposition",
        "structural_representation",
        "fleet_strategy_required",
        "hack_v1",
        "hack_v1_constraint_empty",
        "hack_v1_feasible_aircraft",
        "hack_v1_permanent_exclusions",
        "hack_v2_ranking",
        "hack_v3_renderer",
        "hack_v3_renderer_locked",
        "freeze_frame",
        "multi_factor_ranking",
        "broker_narrative_authoritative",
        "tier_downgrade_blocked",
    ):
        if isinstance(du, dict) and k in du:
            patch[k] = du.get(k)
    patch.update(persist_mission_state_patch(pipeline.persistent_mission_state))
    try:
        from services.mission.mission_understanding_engine import (
            MISSION_UNDERSTANDING_KEY,
            load_mission_understanding,
        )

        _understanding_pkt = load_mission_understanding(du)
        if _understanding_pkt is not None:
            patch[MISSION_UNDERSTANDING_KEY] = _understanding_pkt.to_dict()
            patch["mission_understanding_confidence"] = _understanding_pkt.overall_confidence
            patch["recommend_aircraft_gated"] = 1 if _understanding_pkt.recommend_aircraft else 0
    except Exception:
        pass
    patch["deterministic_recommendation_pipeline"] = {
        "mission_category": pipeline.mission_category.value if pipeline.mission_category else "",
        "recommendations": [r.to_dict() for r in pipeline.recommendations],
        "feasible_models": list(pipeline.feasible_models),
        "eliminated_models": list(pipeline.eliminated_models),
    }
    if pipeline.mission_validation:
        patch["mission_state_validation"] = pipeline.mission_validation

    mission = pipeline.mission_state
    persistent = pipeline.persistent_mission_state
    clarifications_asked = max(
        0, int(getattr(persistent, "clarification_questions_asked", 0) or 0)
    )
    recommendations = list(pipeline.recommendations)
    route_assessments = _build_route_assessments(mission, recommendations)

    from services.orchestration.recommendation_gate import (
        apply_recommendation_gate_metadata,
        finalize_recommendations,
        strip_aircraft_from_response,
    )
    from services.mission.mission_understanding_engine import load_mission_understanding

    _gate_pkt = load_mission_understanding(du)
    rec_gate = finalize_recommendations(
        query,
        recommendations,
        mission,
        data_used=du,
        packet=_gate_pkt,
        max_results=max_results,
    )
    apply_recommendation_gate_metadata(patch, rec_gate)
    apply_recommendation_gate_metadata(du, rec_gate)
    if rec_gate.suppress_aircraft:
        recommendations = []
        if _gate_pkt is not None:
            _gate_pkt.recommend_aircraft = False
            patch["recommend_aircraft_gated"] = 0
    else:
        recommendations = list(rec_gate.filtered_recommendations)

    # Stabilization: shortlist validation gate.
    # EMPTY SHORTLIST != permission to recover via lower-tier fallback hallucinations.
    if isinstance(du, dict) and du.get("mission_hard_invalid"):
        recommendations = []

    # Alternatives query: if stabilization set a reference aircraft, do not "recommend" the reference.
    if recommendations and isinstance(du, dict):
        try:
            stab = du.get("orchestration_stabilization") or {}
            if isinstance(stab, dict):
                ref = str(stab.get("reference_aircraft") or "").strip()
                if ref:
                    recommendations = [r for r in recommendations if (r.model or "") != ref]
        except Exception:
            pass

    if recommendations and isinstance(du, dict):
        try:
            v1_feasible = set(du.get("hack_v1_feasible_aircraft") or [])
            if v1_feasible:
                recommendations = [r for r in recommendations if (r.model or "") in v1_feasible]
        except Exception:
            pass

    if not recommendations and isinstance(du, dict) and not rec_gate.suppress_aircraft:
        # If validation strips everything, do not invent a new shortlist.
        # Force strategic analysis renderer for this turn.
        du["tier_downgrade_blocked"] = du.get("tier_downgrade_blocked") or "shortlist_validation_empty"
        v2_route = OrchestrationRouterV2Result(
            query_type=OrchestrationQueryTypeV2.STRATEGIC_FLEET_ANALYSIS,
            renderer=OrchestrationRendererV2.STRATEGIC_ANALYSIS,
            confidence=0.90,
            signals=list((v2_route.signals or [])) + ["stabilizer:shortlist_validation_empty"],
            authoritative=True,
            allow_recommendation_ranking=False,
            allow_tier_fallback=False,
            allow_operational_synthesis=False,
            preserve_comparison_models=(),
            named_aircraft_models=(),
            suppress_generic_shortlist=True,
            requires_deterministic_pipeline=True,
            physics_first_priority=True,
        )
        from services.orchestration.orchestration_router_v2 import apply_orchestration_v2_metadata

        apply_orchestration_v2_metadata(du, v2_route)
        apply_orchestration_v2_metadata(patch, v2_route)

    mentioned = detect_models_from_text(query)
    ql = (query or "").lower()
    comparison = None
    locked_compare = orchestration_v2_locked_comparison_models(du)
    compare_models = locked_compare if locked_compare else mentioned
    if (
        len(compare_models) >= 2
        or v2_route.query_type == OrchestrationQueryTypeV2.EXPLICIT_COMPARISON
        or "compare" in ql
        or " vs " in ql
        or "versus" in ql
    ):
        comparison = build_structured_comparison(
            compare_models,
            mission,
            recommendations=recommendations,
            locked_models_only=bool(locked_compare),
        )
        patch["consultant_comparison"] = comparison.to_dict()

    visuals = build_visual_intelligence_bundle(
        mission,
        recommendations,
        route_assessments[:6],
        comparison=comparison,
    )
    patch["consultant_visual_models"] = visuals.to_dict()

    validation = pipeline.mission_validation or {}
    authority_block = build_pipeline_authority_block(
        pipeline,
        query=query,
        query_intent=qri.intent.value,
        data_used=du,
    )
    try:
        from services.telemetry.reasoning_packet import IMMUTABLE_PACKET_KEY

        if IMMUTABLE_PACKET_KEY in du:
            patch[IMMUTABLE_PACKET_KEY] = du[IMMUTABLE_PACKET_KEY]
    except Exception:
        pass
    patch["pre_llm_pipeline_authority"] = 1

    answer = ""
    image_confidence: Optional[float] = None
    aircraft_images: List[Dict[str, Any]] = []

    # --- Stage 5: Broker Narrative Generation (deterministic first) ---
    with StageRunner(tr, ORCHESTRATION_STAGES[4]) as s5:
        from services.mission.mission_understanding_engine import (
            format_understanding_first_advisory,
            load_mission_understanding,
        )

        _pkt = load_mission_understanding(du)
        if (
            v2_route.renderer == OrchestrationRendererV2.NAMED_CAPABILITY
            and v2_route.named_aircraft_models
        ):
            from services.consultant.named_aircraft_capability import (
                format_named_aircraft_capability_response,
            )

            answer = format_named_aircraft_capability_response(
                v2_route.named_aircraft_models,
                mission,
                mission_profile=pipeline.mission_profile,
                data_used=du,
                query=query,
            )
            recommendations = []
            du["broker_narrative_authoritative"] = True
            s5.details = {"mode": "named_aircraft_capability_v2"}
        elif v2_route.renderer == OrchestrationRendererV2.STRATEGIC_ANALYSIS:
            from services.consultant.strategic_analysis_renderer import (
                format_strategic_analysis_response,
            )
            from services.mission.mission_understanding_engine import load_mission_understanding

            answer = format_strategic_analysis_response(
                mission,
                load_mission_understanding(du),
                query=query,
                data_used=du,
            )
            recommendations = []
            s5.details = {"mode": "strategic_analysis_v2"}
        elif v2_route.renderer == OrchestrationRendererV2.NETWORK_TOPOLOGY:
            from services.consultant.network_topology_renderer import (
                format_network_topology_response,
            )

            answer = format_network_topology_response(
                mission,
                query=query,
                data_used=du,
                packet=_pkt,
            )
            recommendations = []
            du["broker_narrative_authoritative"] = True
            s5.details = {"mode": "network_topology_v2"}
        elif v2_route.renderer == OrchestrationRendererV2.STRATEGIC_COMPARISON:
            from services.consultant.strategic_comparison_renderer import (
                format_strategic_comparison_response,
            )

            answer = format_strategic_comparison_response(
                mission,
                query=query,
                data_used=du,
            )
            recommendations = []
            du["broker_narrative_authoritative"] = True
            s5.details = {"mode": "strategic_comparison_v2"}
        elif v2_route.query_type == OrchestrationQueryTypeV2.EXPLICIT_COMPARISON:
            from services.consultant.comparison_structured_output import (
                format_comparison_response,
            )

            locked_compare = orchestration_v2_locked_comparison_models(du)
            mentioned_models = detect_models_from_text(query)
            compare_models = locked_compare if locked_compare else mentioned_models
            answer = format_comparison_response(
                query=query,
                mission=mission,
                compare_models=compare_models,
                data_used=du,
            )
            recommendations = []
            du["broker_narrative_authoritative"] = True
            s5.details = {"mode": "comparison_structured_engine", "count": len(compare_models)}
        elif validation.get("needs_route_clarification") or (
            validation.get("synthesis_first_required")
            and v2_route.allow_operational_synthesis
        ):
            # Reasoning-first: provide operational synthesis + class band even when city pair is missing.
            clarifying = str(validation.get("clarifying_question") or "").strip()
            if _pkt is not None:
                answer = format_understanding_first_advisory(
                    mission,
                    _pkt,
                    recommendations=recommendations if recommendations else [],
                    query=query,
                    data_used=du,
                    route_certainty_degraded=bool(
                        validation.get("route_certainty_degraded")
                        or validation.get("route_blocks_ranking")
                    ),
                )
                if clarifying:
                    answer = f"{answer}\n\nNext, {clarifying}"
                s5.details = {"mode": "route_clarification_understanding_gate"}
            else:
                answer = format_route_clarification_response(
                    mission=mission,
                    clarifying_question=str(validation.get("clarifying_question") or ""),
                )
                s5.details = {"mode": "route_clarification"}
        elif recommendations:
            from services.orchestration.response_mode_classifier import (
                explicit_aircraft_request,
                load_orchestration_response_mode,
            )
            from services.orchestration.recommendation_gate import _comparative_economics_query
            from services.consultant.comparative_analysis_renderer import is_named_model_comparison_query

            force_broker = (
                explicit_aircraft_request(query)
                or _comparative_economics_query(query)
                or is_named_model_comparison_query(query)
            )
            mode_cached = load_orchestration_response_mode(du)
            if mode_cached and mode_cached.structural_first and not explicit_aircraft_request(query):
                force_broker = False
            if rec_gate.render_interpretation_only and _pkt is not None and not force_broker:
                from services.orchestration.recommendation_gate import (
                    render_interpretation_first_response,
                )

                answer = render_interpretation_first_response(
                    mission,
                    _pkt,
                    query=query,
                    data_used=du,
                    route_certainty_degraded=bool(validation.get("route_certainty_degraded")),
                )
                recommendations = []
                s5.degraded("interpretation_mode", confidence=_pkt.overall_confidence)
                s5.details = {"mode": "interpretation_first"}
            else:
                broker_body = format_broker_advisory_response(
                    mission,
                    [r for r in recommendations if not r.avoid][:3],
                    route_assessments=route_assessments,
                    eliminated_models=list(pipeline.eliminated_models),
                    data_used=du,
                    query=query,
                )
                answer = broker_body or ""
                du["broker_narrative_authoritative"] = True
                s5.details = {
                    "mode": "broker_deterministic",
                    "models": [r.model for r in recommendations[:3]],
                }
        else:
            if _pkt is not None:
                if rec_gate.render_interpretation_only:
                    from services.orchestration.recommendation_gate import (
                        render_interpretation_first_response,
                    )

                    answer = render_interpretation_first_response(
                        mission,
                        _pkt,
                        query=query,
                        data_used=du,
                        route_certainty_degraded=bool(validation.get("route_certainty_degraded")),
                    )
                    s5.details = {"mode": "interpretation_first_empty_shortlist"}
                else:
                    # Renderer contamination block: do not leak OPERATIONAL SYNTHESIS into
                    # broker/strategic flows when synthesis is suppressed by router/stabilizer.
                    if isinstance(du, dict) and du.get("kernel_synthesis_blocked"):
                        answer = degraded_empty_shortlist_guidance(mission, pipeline, query)
                        s5.details = {"mode": "empty_shortlist_degraded_no_synthesis"}
                    else:
                        answer = format_understanding_first_advisory(
                            mission,
                            _pkt,
                            query=query,
                            data_used=du,
                            route_certainty_degraded=bool(validation.get("route_certainty_degraded")),
                        )
            else:
                answer = degraded_empty_shortlist_guidance(mission, pipeline, query)
            s5.degraded("empty_shortlist", confidence=0.45)

    # --- Stage 6: Image Verification (on-demand only — never on advisory-only turns) ---
    with StageRunner(tr, ORCHESTRATION_STAGES[5]) as s6:
        from services.orchestration.image_trust_policy import should_activate_image_trust

        if not should_activate_image_trust(query):
            s6.skip("advisory_turn_no_explicit_visual_request")
            aircraft_images = []
        elif not recommendations:
            s6.skip("no_recommendations")
            aircraft_images = []
        else:
            try:
                from services.consultant_aircraft_images import build_consultant_aircraft_images

                gallery_meta: Dict[str, Any] = {}
                aircraft_images = build_consultant_aircraft_images(
                    tavily_payload or {},
                    phly_rows or [],
                    user_query=query,
                    history=history,
                    gallery_meta_out=gallery_meta,
                )
                patch["consultant_gallery_meta"] = gallery_meta
                confidences = [
                    float(im.get("confidence") or im.get("verification_confidence") or 0)
                    for im in aircraft_images
                    if im.get("url")
                ]
                if confidences:
                    image_confidence = sum(confidences) / len(confidences)
                    s6.confidence = image_confidence
                else:
                    image_confidence = None
                s6.details = {
                    "image_count": len(aircraft_images),
                    "avg_confidence": round(image_confidence, 3) if image_confidence else None,
                }
            except Exception as exc:
                logger.warning("orchestration image_verification failed: %s", exc)
                s6.degraded("image_pipeline_failed", confidence=0.4)

    # --- Stage 7: Final Response Formatting ---
    with StageRunner(tr, ORCHESTRATION_STAGES[6]) as s7:
        formatter_ok = use_structured_formatter
        if formatter_ok is None:
            formatter_ok = should_use_structured_formatter(du, mission, query)
        try:
            if validation.get("needs_route_clarification"):
                pass  # answer already set
            elif du.get("broker_narrative_authoritative") or du.get("hack_v3_renderer_locked"):
                pass  # broker / HACK v3 locked output — no kernel or formatter overwrite
            elif formatter_ok and recommendations:
                pipeline_body = format_consultant_response(
                    mission=mission,
                    recommendations=recommendations,
                    route_assessments=route_assessments,
                    comparison=comparison,
                    draft_answer="",
                    include_comparison_section=comparison is not None,
                    query=query,
                    turn_seed=query,
                    clarifications_already_asked=clarifications_asked,
                    eliminated_models=list(pipeline.eliminated_models),
                    data_used=du,
                    include_acquisition_intelligence=bool(
                        re.search(
                            r"\b(?:acquire|acquisition|buy|purchase|resale|liquidity)\b",
                            (query or "").lower(),
                        )
                    ),
                )
                try:
                    from services.recommendation.clarification_decision import (
                        mission_clarification_needs,
                    )
                    from services.state.mission_state import record_clarification_question_asked

                    follow_needs = mission_clarification_needs(
                        mission,
                        query,
                        recommendations=[r for r in recommendations if not r.avoid],
                        clarifications_already_asked=clarifications_asked,
                    )
                    if follow_needs.any and not follow_needs.needs_route:
                        persistent = record_clarification_question_asked(persistent)
                        patch.update(persist_mission_state_patch(persistent))
                except Exception:
                    pass
                hard_excluded: set = set()
                hard_ctx = None
                try:
                    from services.recommendation.hard_mission_elimination import (
                        detect_hard_elimination_context,
                        hard_excluded_model_set,
                    )

                    hard_ctx = detect_hard_elimination_context(pipeline.mission_profile)
                    hard_excluded = hard_excluded_model_set(pipeline.mission_profile)
                except Exception:
                    pass

                comp_models = (
                    list(comparison.models) if comparison and comparison.models else None
                )
                merged, regen = reconcile_answer_with_pipeline(
                    llm_draft,
                    mission=mission,
                    recommendations=recommendations,
                    route_assessments=route_assessments,
                    comparison=comparison,
                    query=query,
                    turn_seed=query,
                    comparison_models=comp_models,
                    hard_excluded=hard_excluded if hard_ctx is not None else None,
                    data_used=du,
                )
                if regen and (merged or "").strip():
                    answer = merged
                else:
                    answer = pipeline_body
                patch["pipeline_authority_enforced"] = 1
                patch["llm_narration_mode"] = "orchestration_authoritative"

                answer, fmt_report = ensure_validated_consultant_response(
                    answer,
                    mission=mission,
                    recommendations=recommendations,
                    route_assessments=route_assessments,
                    comparison=comparison,
                    query=query,
                    turn_seed=query,
                )
                patch["consultant_format_validation"] = fmt_report.to_dict()
                try:
                    from services.telemetry.reasoning_packet_enforcement import (
                        enforce_reasoning_packet_authority,
                    )

                    answer, pkt_enf = enforce_reasoning_packet_authority(
                        answer,
                        data_used=du,
                        recommendations=recommendations,
                        mission=mission,
                        route_assessments=route_assessments,
                        comparison_models=comp_models,
                        query=query,
                        turn_seed=query,
                    )
                    patch["reasoning_packet_enforcement"] = pkt_enf.to_dict()
                except Exception as exc:
                    logger.warning("reasoning_packet_enforcement failed (non-fatal): %s", exc)
                try:
                    from services.mission.mission_understanding_engine import (
                        load_mission_understanding,
                    )
                    from services.mission.mission_authority_kernel import (
                        attach_kernel_enforcement_report,
                        build_mission_authority_kernel,
                        enforce_kernel_authority,
                        filter_recommendations_by_kernel,
                        load_mission_authority_kernel,
                    )

                    _syn_pkt = load_mission_understanding(du)
                    if _syn_pkt is not None and not du.get("kernel_synthesis_blocked"):
                        _kernel = load_mission_authority_kernel(du) or build_mission_authority_kernel(
                            mission,
                            _syn_pkt,
                            pipeline.mission_profile,
                            recommendations=recommendations,
                            query=query,
                            data_used=du,
                            route_certainty_degraded=bool(
                                validation.get("route_certainty_degraded")
                                or validation.get("route_blocks_ranking")
                            ),
                        )
                        _filtered = filter_recommendations_by_kernel(recommendations, _kernel)
                        answer, _kr = enforce_kernel_authority(
                            answer,
                            _kernel,
                            _filtered,
                        )
                        attach_kernel_enforcement_report(du, _kr)
                        patch["kernel_authority_enforcement"] = _kr.to_dict()
                        from services.mission.mission_authority_kernel import (
                            MISSION_AUTHORITY_KERNEL_KEY,
                        )

                        patch[MISSION_AUTHORITY_KERNEL_KEY] = _kernel.to_dict()
                        patch["mission_authority_bound"] = 1
                except Exception as exc:
                    logger.warning("kernel_authority_enforcement failed (non-fatal): %s", exc)
                try:
                    from services.consultant.response_formatter import last_response_style

                    patch["consultant_response_style"] = last_response_style()
                except Exception:
                    pass
            elif not answer:
                answer = safe_stage_fallback(
                    ORCHESTRATION_STAGES[6],
                    query=query,
                    mission=mission,
                    pipeline=pipeline,
                    recommendations=recommendations,
                    data_used=du,
                )
                s7.degraded("empty_answer", confidence=0.4)

            # Comparison + capability outputs must preserve strict structure (tables/newlines).
            if isinstance(du, dict) and (
                du.get("comparison_v2")
                or du.get("comparison_structured_engine")
                or du.get("network_topology_renderer")
                or du.get("strategic_comparison_renderer")
                or du.get("strategic_analysis_renderer")
            ):
                answer = (answer or "").strip()
            else:
                suppressed = suppress_templates(answer)
                answer = sanitize_advisor_output(suppressed.text)
                try:
                    from services.consultant.response_cleanup import cleanResponseText

                    answer = cleanResponseText(answer)
                except Exception:
                    pass
            s7.details = {"formatter": bool(formatter_ok and recommendations)}
        except Exception as exc:
            logger.exception("orchestration final_response_formatting failed")
            answer = answer or safe_stage_fallback(
                ORCHESTRATION_STAGES[6],
                query=query,
                mission=mission,
                pipeline=pipeline,
                recommendations=recommendations,
                data_used=du,
            )
            s7.degraded(str(exc)[:120], confidence=0.35)

    answer = ensure_non_empty_answer(
        answer,
        query=query,
        mission=mission,
        pipeline=pipeline,
        recommendations=recommendations,
        data_used=du,
    )
    conf = finalize_trace_confidence(
        tr,
        pipeline,
        recommendations=recommendations,
        image_confidence=image_confidence,
    )
    # Comparison v2 contract: JSON-only output — no degradation prefix or prose wrapper.
    if (
        v2_route.query_type == OrchestrationQueryTypeV2.EXPLICIT_COMPARISON
        and isinstance(du, dict)
        and (du.get("comparison_v2") or du.get("comparison_structured_engine"))
    ):
        answer = (answer or "").strip()
        low = False
    else:
        answer, low = apply_low_confidence_guidance(answer, conf)
    tr.low_confidence = low

    if rec_gate.suppress_aircraft and (answer or "").strip():
        # Output safety: never strip aircraft identifiers from capability/comparison modes
        # (would corrupt structured tables and named-aircraft verdicts).
        try:
            from services.orchestration.orchestration_router_v2 import OrchestrationQueryTypeV2

            if v2_route.query_type not in (
                OrchestrationQueryTypeV2.EXPLICIT_COMPARISON,
                OrchestrationQueryTypeV2.NAMED_AIRCRAFT_CAPABILITY,
            ):
                answer = strip_aircraft_from_response(answer)
        except Exception:
            answer = strip_aircraft_from_response(answer)

    patch["orchestration"] = tr.to_dict()
    patch["consultant_recommendations"] = [r.to_dict() for r in recommendations]
    if route_assessments:
        patch["consultant_route_feasibility"] = [a.to_dict() for a in route_assessments[:8]]
    for _meta_key in (
        "hack_v1",
        "hack_v1_constraint_empty",
        "hack_v1_feasible_aircraft",
        "hack_v2_ranking",
        "hack_v3_renderer",
        "hack_v3_renderer_locked",
        "freeze_frame",
        "multi_factor_ranking",
        "broker_narrative_authoritative",
        "tier_downgrade_recovery",
        "orchestration_v2",
        "orchestration_v2_query_type",
        "orchestration_v2_renderer",
        "kernel_synthesis_blocked",
    ):
        if isinstance(du, dict) and _meta_key in du:
            patch[_meta_key] = du.get(_meta_key)

    return ConsultantOrchestrationResult(
        answer=answer,
        mission_state=mission,
        recommendations=recommendations,
        aircraft_images=aircraft_images,
        pipeline_result=pipeline,
        trace=tr,
        data_used_patch=patch,
        route_assessments=route_assessments,
        comparison=comparison,
        low_confidence=low,
        authority_block=authority_block,
    )
