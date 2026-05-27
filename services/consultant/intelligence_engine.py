"""
Consultant Intelligence Orchestrator — runs after generation, before containment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.consultant.comparison_engine import build_structured_comparison
from services.consultant.mission_state import MissionState
from services.mission.memory_bridge import extract_mission_with_memory
from services.mission.adapters import mission_profile_to_state
from services.consultant.prompt_hygiene import apply_prompt_hygiene
from services.consultant.recommendation_engine import detect_models_from_text
from services.consultant.recommendation_authority import reconcile_answer_with_pipeline
from services.consultant.response_format_validation import ensure_validated_consultant_response
from services.consultant.response_formatter import (
    format_consultant_response,
    format_route_clarification_response,
    sanitize_advisor_output,
    should_use_structured_formatter,
)
from services.state.mission_state import persist_mission_state_patch
from services.consultant.route_feasibility import assess_mission_routes
from services.consultant.recommendation_engine import _AIRCRAFT_PROFILES
from services.consultant.template_suppression import suppress_templates
from services.consultant.visual_models import build_visual_intelligence_bundle


def consultant_intelligence_enabled() -> bool:
    return (os.getenv("CONSULTANT_INTELLIGENCE_LAYER") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


@dataclass
class ConsultantIntelligenceResult:
    answer: str
    mission_state: MissionState
    data_used_patch: Dict[str, Any] = field(default_factory=dict)
    applied: bool = False


def _is_advisory_context(data_used: Dict[str, Any], query: str) -> bool:
    try:
        from rag.pinpoint_answer import is_pinpoint_factual_turn

        if is_pinpoint_factual_turn(query, data_used):
            return False
    except Exception:
        pass
    return True


def _merge_prior_answer(history: Optional[List[Dict[str, str]]]) -> str:
    if not history:
        return ""
    for turn in reversed(history):
        if isinstance(turn, dict) and str(turn.get("role")).lower() == "assistant":
            return str(turn.get("content") or "")
    return ""


def run_consultant_intelligence_layer(
    *,
    answer: str,
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    data_used: Optional[Dict[str, Any]] = None,
    conversation_state: Optional[Dict[str, Any]] = None,
) -> ConsultantIntelligenceResult:
    """
    Post-generation intelligence: mission state, scoring, formatting, suppression, visuals.

    Pipeline position: after LLM draft/review, before response_safety / containment.
    """
    du = dict(data_used) if isinstance(data_used, dict) else {}
    from services.preprocessing import attach_mission_preprocessing

    attach_mission_preprocessing(du, query)
    raw_answer = (answer or "").strip()

    if not consultant_intelligence_enabled():
        return ConsultantIntelligenceResult(
            answer=raw_answer,
            mission_state=MissionState(),
            applied=False,
        )

    # Turn-isolated extraction; optional stable memory merge (current turn always wins).
    turn_profile, profile, next_memory = extract_mission_with_memory(
        query,
        conversation_state=conversation_state,
        data_used=du,
    )
    mission = mission_profile_to_state(profile)

    # Always run suppression + hygiene on generated text
    prior_assistant = _merge_prior_answer(history)
    suppressed = suppress_templates(raw_answer)
    cleaned, hygiene = apply_prompt_hygiene(
        suppressed.text,
        prior_answer=prior_assistant,
        history=history,
        turn_seed=query,
    )
    working = cleaned

    patch: Dict[str, Any] = {
        "consultant_intelligence_layer": 1,
        "consultant_mission_profile": {
            **profile.to_dict(),
            "memory_merge_applied": profile != turn_profile,
        },
        "consultant_mission_turn_only": turn_profile.to_dict(),
        "consultant_mission_memory": next_memory.to_dict(),
        "consultant_mission_state": mission.to_dict(),
        "consultant_template_suppression": {
            "removed_blocks": suppressed.removed_blocks,
            "duplicate_paragraphs_removed": suppressed.duplicate_paragraphs_removed,
            "fallback_contamination_score": round(suppressed.fallback_contamination_score, 3),
        },
        "consultant_prompt_hygiene": hygiene.to_dict(),
    }

    try:
        from services.mission.mission_understanding_engine import (
            apply_understanding_to_mission_state,
            apply_understanding_to_profile,
            attach_mission_understanding,
            build_mission_understanding,
        )
        from services.session.broker_memory import (
            load_broker_memory,
            save_broker_memory,
            update_broker_memory_from_understanding,
        )

        _bm = load_broker_memory(du)
        _pkt = build_mission_understanding(
            query,
            profile,
            mission,
            broker_memory=_bm.to_dict(),
            history=history,
        )
        profile = apply_understanding_to_profile(profile, _pkt)
        mission = apply_understanding_to_mission_state(mission, _pkt)
        attach_mission_understanding(du, _pkt)
        _bm = update_broker_memory_from_understanding(_bm, _pkt)
        save_broker_memory(du, _bm)
        patch["mission_understanding_confidence"] = _pkt.overall_confidence
        patch["corridor_type"] = _pkt.corridor_type
    except Exception:
        pass

    if not _is_advisory_context(du, query):
        patch["consultant_intelligence_skipped"] = "pinpoint_factual"
        try:
            from services.consultant.phrase_repetition_guard import apply_phrase_repetition_guard

            working, phrase_report = apply_phrase_repetition_guard(
                working,
                history=history,
                prior_answer=prior_assistant,
                turn_seed=query,
            )
            patch["phrase_repetition_guard"] = phrase_report.to_dict()
        except Exception:
            pass
        return ConsultantIntelligenceResult(
            answer=working,
            mission_state=mission,
            data_used_patch=patch,
            applied=True,
        )

    from services.consultant.pre_llm_recommendation import resolve_query_recommendation_intent
    from services.recommendation.query_recommendation_intent import (
        QueryRecommendationIntent,
        apply_query_intent_metadata,
    )

    qri = resolve_query_recommendation_intent(query, history=history, data_used=du)
    apply_query_intent_metadata(patch, qri)
    apply_query_intent_metadata(du, qri)

    if qri.intent == QueryRecommendationIntent.VISUALIZATION_REQUEST:
        from services.consultant.visualization_handler import (
            build_visualization_authority_block,
            run_visualization_turn,
        )

        viz = run_visualization_turn(
            query,
            mission=mission,
            history=history,
            conversation_state=conversation_state,
            data_used=du,
        )
        patch["visualization_turn"] = viz.to_dict()
        patch["consultant_visual_models"] = viz.bundle.to_dict()
        patch["consultant_response_mode_canonical"] = "image_showcase"
        patch["consultant_response_mode"] = "image_showcase"
        if viz.recommendations:
            patch["consultant_recommendations"] = [r.to_dict() for r in viz.recommendations]
        if viz.followup_needed:
            working = viz.followup_message
            patch["consultant_structured_formatter"] = "visualization_followup"
        else:
            from services.consultant.visualization_render import (
                format_visualization_user_response,
            )

            rendered, render_patch = format_visualization_user_response(viz)
            working = rendered or viz.caption or working
            patch.update(render_patch)
            patch["consultant_structured_formatter"] = "visualization_direct"
        try:
            from services.consultant.phrase_repetition_guard import apply_phrase_repetition_guard

            working, phrase_report = apply_phrase_repetition_guard(
                working,
                history=history,
                prior_answer=prior_assistant,
                turn_seed=query,
            )
            patch["phrase_repetition_guard"] = phrase_report.to_dict()
        except Exception:
            pass
        patch["visualization_authority"] = build_visualization_authority_block(viz)
        return ConsultantIntelligenceResult(
            answer=working,
            mission_state=mission,
            data_used_patch=patch,
            applied=True,
        )

    if not qri.requires_ranked_pipeline:
        patch["consultant_intelligence_skipped_pipeline"] = qri.intent.value
        try:
            from services.consultant.phrase_repetition_guard import apply_phrase_repetition_guard

            working, phrase_report = apply_phrase_repetition_guard(
                working,
                history=history,
                prior_answer=prior_assistant,
                turn_seed=query,
            )
            patch["phrase_repetition_guard"] = phrase_report.to_dict()
        except Exception:
            pass
        return ConsultantIntelligenceResult(
            answer=working,
            mission_state=mission,
            data_used_patch=patch,
            applied=True,
        )

    mentioned = detect_models_from_text(query)
    ql = (query or "").lower()
    comparison_query = (
        qri.intent == QueryRecommendationIntent.AIRCRAFT_COMPARISON
        or (
            len(mentioned) >= 2
            and any(tok in ql for tok in ("compare", " versus ", " vs ", "versus"))
        )
    )

    from services.recommendation.fit_policy import recommendation_limit_from_query
    from services.orchestration.modes import orchestration_enabled
    from services.orchestration.pipeline_orchestrator import run_consultant_orchestration

    tavily_payload = du.get("tavily") if isinstance(du.get("tavily"), dict) else {}
    phly_rows = du.get("phlydata_rows") if isinstance(du.get("phlydata_rows"), list) else []

    if orchestration_enabled():
        orch = run_consultant_orchestration(
            query,
            llm_draft=working,
            conversation_state=conversation_state,
            data_used=du,
            history=history,
            explicit_candidates=mentioned if comparison_query else None,
            max_results=recommendation_limit_from_query(query),
            query_intent=qri.intent.value,
            tavily_payload=tavily_payload,
            phly_rows=phly_rows,
        )
        patch.update(orch.data_used_patch)
        working = orch.answer
        mission = orch.mission_state
        recommendations = orch.recommendations
        pipeline = orch.pipeline_result
        route_assessments = orch.route_assessments
        comparison = orch.comparison
        if orch.low_confidence:
            patch["orchestration_low_confidence"] = True
        if pipeline:
            patch["consultant_mission_category"] = (
                pipeline.mission_category.value if pipeline.mission_category else ""
            )
            patch["consultant_feasibility"] = {
                m: fr.to_dict() for m, fr in pipeline.feasibility_map.items()
            }
            patch["consultant_feasible_models"] = list(pipeline.feasible_models)
            patch["consultant_eliminated_models"] = list(pipeline.eliminated_models)
            patch["consultant_feasibility_elimination_log"] = pipeline.elimination_log
            if pipeline.capability_graph:
                patch["consultant_capability_graph"] = pipeline.capability_graph
            validation = pipeline.mission_validation or {}
            if validation.get("needs_route_clarification"):
                patch["consultant_structured_formatter"] = "route_clarification"
            elif recommendations:
                patch["consultant_structured_formatter"] = 1
        if orch.aircraft_images:
            patch["aircraft_images"] = orch.aircraft_images
    else:
        from services.recommendation.recommendation_pipeline import run_recommendation_pipeline

        pipeline, pipe_trace = run_recommendation_pipeline(
            query,
            conversation_state=conversation_state,
            data_used=du,
            explicit_candidates=mentioned if comparison_query else None,
            max_results=recommendation_limit_from_query(query),
            query_intent=qri.intent.value,
        )
        patch["recommendation_pipeline"] = pipe_trace.to_dict()
        patch["recommendation_decision_source"] = pipe_trace.decision_source
        mission = pipeline.mission_state
        patch.update(persist_mission_state_patch(pipeline.persistent_mission_state))
        if pipeline.mission_validation:
            patch["mission_state_validation"] = pipeline.mission_validation
        recommendations = pipeline.recommendations
        mission_category = pipeline.mission_category
        patch["consultant_mission_category"] = mission_category.value
        patch["consultant_feasibility"] = {
            m: fr.to_dict() for m, fr in pipeline.feasibility_map.items()
        }
        patch["consultant_feasible_models"] = list(pipeline.feasible_models)
        patch["consultant_eliminated_models"] = list(pipeline.eliminated_models)
        patch["consultant_feasibility_elimination_log"] = pipeline.elimination_log
        hard_ctx = None
        hard_excluded: set = set()
        try:
            from services.recommendation.hard_mission_elimination import (
                detect_hard_elimination_context,
                hard_excluded_model_set,
            )

            hard_ctx = detect_hard_elimination_context(pipeline.mission_profile)
            hard_excluded = hard_excluded_model_set(pipeline.mission_profile)
            if hard_ctx is not None:
                patch["hard_mission_elimination"] = {
                    "rule_id": hard_ctx.rule_id,
                    "summary": hard_ctx.summary,
                    "required_route_nm": hard_ctx.required_route_nm,
                    "excluded_models": sorted(hard_excluded),
                }
        except Exception:
            pass
        if pipeline.capability_graph:
            patch["consultant_capability_graph"] = pipeline.capability_graph

        route_assessments = []
        if mission.routes and recommendations:
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

        comparison = None
        ql = (query or "").lower()
        if len(mentioned) >= 2 or "compare" in ql or " vs " in ql or "versus" in ql:
            comparison = build_structured_comparison(
                mentioned, mission, recommendations=recommendations
            )
            patch["consultant_comparison"] = comparison.to_dict()

        visuals = build_visual_intelligence_bundle(
            mission,
            recommendations,
            route_assessments[:6],
            comparison=comparison,
        )
        patch["consultant_visual_models"] = visuals.to_dict()
        patch["consultant_recommendations"] = [r.to_dict() for r in recommendations]
        if route_assessments:
            patch["consultant_route_feasibility"] = [
                a.to_dict() for a in route_assessments[:8]
            ]

        validation = pipeline.mission_validation or {}
        if validation.get("needs_route_clarification"):
            working = format_route_clarification_response(
                mission=mission,
                clarifying_question=str(validation.get("clarifying_question") or ""),
            )
            patch["consultant_structured_formatter"] = "route_clarification"
        else:
            use_formatter = should_use_structured_formatter(du, mission, query)
            _pipeline_formatter_intents = frozenset(
                {
                    QueryRecommendationIntent.ACQUISITION_RECOMMENDATION,
                    QueryRecommendationIntent.MISSION_FEASIBILITY,
                    QueryRecommendationIntent.AIRCRAFT_COMPARISON,
                    QueryRecommendationIntent.OPERATIONAL_TRADEOFF_ANALYSIS,
                    QueryRecommendationIntent.SHORTLIST_RANKING,
                }
            )
            use_formatter = (
                use_formatter
                and bool(recommendations)
                and qri.intent in _pipeline_formatter_intents
            )
            if use_formatter:
                pipeline_body = format_consultant_response(
                    mission=mission,
                    recommendations=recommendations,
                    route_assessments=route_assessments,
                    comparison=comparison,
                    draft_answer="",
                    include_comparison_section=comparison is not None,
                    query=query,
                    turn_seed=query,
                )
                from services.consultant.response_formatter import last_response_style

                patch["consultant_response_style"] = last_response_style()
                patch["consultant_structured_formatter"] = 1
                patch["llm_narration_mode"] = "pipeline_authoritative"

                comp_models = (
                    list(comparison.models) if comparison and comparison.models else None
                )
                merged, regen = reconcile_answer_with_pipeline(
                    working,
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
                working = merged if regen and (merged or "").strip() else pipeline_body
                patch["pipeline_authority_enforced"] = 1

        if recommendations and patch.get("consultant_structured_formatter"):
            working, fmt_report = ensure_validated_consultant_response(
                working,
                mission=mission,
                recommendations=recommendations,
                route_assessments=route_assessments,
                comparison=comparison,
                query=query,
                turn_seed=query,
            )
            patch["consultant_format_validation"] = fmt_report.to_dict()
            if not patch.get("reasoning_packet_enforcement"):
                try:
                    from services.telemetry.reasoning_packet_enforcement import (
                        enforce_reasoning_packet_authority,
                    )

                    pkt_du = {**du, **patch}
                    working, pkt_enf = enforce_reasoning_packet_authority(
                        working,
                        data_used=pkt_du,
                        recommendations=recommendations,
                        mission=mission,
                        route_assessments=route_assessments,
                        comparison_models=(
                            list(comparison.models) if comparison and comparison.models else None
                        ),
                        query=query,
                        turn_seed=query,
                    )
                    patch["reasoning_packet_enforcement"] = pkt_enf.to_dict()
                except Exception:
                    pass

        final_suppressed = suppress_templates(working)
        working = sanitize_advisor_output(final_suppressed.text)
        try:
            from services.consultant.response_cleanup import cleanResponseText

            working = cleanResponseText(working)
        except Exception:
            pass

    regenerate_fn = None
    if recommendations and patch.get("consultant_structured_formatter"):

        def _regen_from_pipeline() -> str:
            return format_consultant_response(
                mission=mission,
                recommendations=recommendations,
                route_assessments=route_assessments,
                comparison=comparison,
                query=query,
                turn_seed=f"{query}|phrase_regen",
            )

        regenerate_fn = _regen_from_pipeline

    try:
        from services.consultant.phrase_repetition_guard import apply_phrase_repetition_guard

        working, phrase_report = apply_phrase_repetition_guard(
            working,
            history=history,
            prior_answer=prior_assistant,
            turn_seed=query,
            regenerate_fn=regenerate_fn,
        )
        patch["phrase_repetition_guard"] = phrase_report.to_dict()
    except Exception:
        pass

    return ConsultantIntelligenceResult(
        answer=working,
        mission_state=mission,
        data_used_patch=patch,
        applied=True,
    )
