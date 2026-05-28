"""
Recommendation gate — final orchestration discipline for aircraft output.

Combines response mode, suppression policy, mission understanding, and structural
state to decide whether aircraft models may appear in the response.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.mission.mission_understanding_engine import MissionUnderstandingPacket
from services.mission.recommendation_suppression import (
    RecommendationSuppressionPolicy,
    build_recommendation_suppression_policy,
    filter_suppressed_recommendations,
    is_generic_dump_model,
)
from services.orchestration.response_mode_classifier import (
    OrchestrationResponseMode,
    OrchestrationResponseModeResult,
    classify_orchestration_response_mode,
    load_orchestration_response_mode,
)

_GENERIC_AIRCRAFT_RE = re.compile(
    r"\b(?:"
    r"gulfstream\s+g?\s*\d+|g\s*\d{3,4}(?:er)?|"
    r"global\s+\d{4,5}|"
    r"falcon\s+\d+x?|"
    r"citation\s+(?:latitude|longitude|x)?|"
    r"phenom\s+\d+|"
    r"challenger\s+\d+|"
    r"legacy\s+\d+|"
    r"praetor\s+\d+"
    r")\b",
    re.I,
)

_LUXURY_MARKETING_RE = re.compile(
    r"\b(?:excellent\s+choice|luxury|flagship|prestige|world[\s-]class\s+cabin)\b",
    re.I,
)

_ECONOMICS_FIRST_RE = re.compile(
    r"\b(?:"
    r"operating\s+economics|economics\s+(?:than|over)\s+prestige|care.*economics|"
    r"ownership\s+economics|fixed\s+cost|charter\s+about\s+\d+\s+hours|"
    r"costs?\s+(?:are\s+)?too\s+high|previously\s+used\s+large|large\s+long[- ]range\s+jets?\s+but"
    r")\b",
    re.I,
)

_COMPARATIVE_ECONOMICS_RE = re.compile(
    r"\b(?:"
    r"converted\s+airliner|at\s+what\s+point|rational\s+than\s+a\s+large\s+business\s+jet|"
    r"large\s+business\s+jet\?|bbj|acj319|acj\s*319|airbus\s+acj|"
    r"which\s+structure\s+is\s+economically"
    r")\b",
    re.I,
)

_ULR_CATEGORIES = frozenset({"ultra-long", "ultra_long"})
_ULR_MODEL_RE = re.compile(
    r"\b(?:global\s*\d+|g\s*650|g\s*700|g\s*800|falcon\s*8x|falcon\s*10x)\b",
    re.I,
)


def _economics_over_prestige_query(query: str) -> bool:
    return bool(_ECONOMICS_FIRST_RE.search(query or ""))


def _comparative_economics_query(query: str) -> bool:
    return bool(_COMPARATIVE_ECONOMICS_RE.search(query or ""))


def _filter_economics_first_recommendations(
    recommendations: Sequence[AircraftRecommendation],
    *,
    query: str,
    packet: Optional[MissionUnderstandingPacket],
    data_used: Optional[Dict[str, Any]],
) -> List[AircraftRecommendation]:
    """Drop ULR glamour picks when economics dominate and utilization is regional."""
    if not recommendations or not _economics_over_prestige_query(query):
        return list(recommendations)

    ql = (query or "").lower()
    regional = any(
        w in ql
        for w in (
            "hawaii",
            "honolulu",
            "dallas",
            "new york",
            "boston",
            "chicago",
            "caribbean",
            "miami",
            "domestic",
            "corridor",
        )
    )
    occasional_ulr = "occasionally" in ql or "occasional" in ql

    if isinstance(data_used, dict):
        hw = data_used.get("hierarchy_weighting") or {}
        if isinstance(hw, dict) and hw.get("dominant_utilization"):
            dom = str(hw.get("dominant_utilization")).lower()
            regional = regional or any(w in dom for w in ("domestic", "corridor", "regional", "hawaii"))

    if packet is not None:
        ic = packet.inferred_constraints or {}
        regional = regional or bool(
            ic.get("domestic_utilization_dominant")
            or ic.get("domestic_utilization_dominates_except_founder_ulr")
        )

    if not (regional or occasional_ulr):
        return list(recommendations)

    filtered = [
        r
        for r in recommendations
        if not _ULR_MODEL_RE.search(r.model or "")
        and str(getattr(r, "category", "") or "").lower() not in ("ultra-long", "ultra_long")
    ]
    return filtered


@dataclass
class RecommendationGateResult:
    suppress_aircraft: bool
    reason: str
    render_interpretation_only: bool
    filtered_recommendations: List[AircraftRecommendation]
    anti_generic_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suppress_aircraft": self.suppress_aircraft,
            "reason": self.reason,
            "render_interpretation_only": self.render_interpretation_only,
            "filtered_count": len(self.filtered_recommendations),
            "anti_generic_applied": self.anti_generic_applied,
        }


def evaluate_recommendation_gate(
    query: str,
    recommendations: Sequence[AircraftRecommendation],
    *,
    data_used: Optional[Dict[str, Any]] = None,
    packet: Optional[MissionUnderstandingPacket] = None,
    response_mode: Optional[OrchestrationResponseModeResult] = None,
    suppression: Optional[RecommendationSuppressionPolicy] = None,
) -> RecommendationGateResult:
    """
    Decide whether aircraft recommendations may render for this turn.
    """
    from services.orchestration.response_mode_classifier import explicit_aircraft_request

    mode = response_mode or load_orchestration_response_mode(data_used)
    if mode is None:
        mode = classify_orchestration_response_mode(query)

    explicit = (
        explicit_aircraft_request(query)
        or _comparative_economics_query(query)
        or (
            mode.mode
            in (
                OrchestrationResponseMode.RECOMMENDATION_MODE,
                OrchestrationResponseMode.BUY_DECISION_MODE,
                OrchestrationResponseMode.COMPARISON_MODE,
            )
            and not mode.suppresses_aircraft_recommendations
        )
    )

    reasons: List[str] = []
    suppress = False
    interpretation_only = False

    if mode.suppresses_aircraft_recommendations:
        suppress = True
        interpretation_only = True
        reasons.append(f"response_mode:{mode.mode.value}")

    # Structural blockers apply even when user asks for aircraft explicitly.
    ic: Dict[str, Any] = {}
    if packet is not None:
        ic = dict(packet.inferred_constraints or {})
    if ic.get("incompatible_mission_bands") or ic.get("multi_hard_domain_mission"):
        if not explicit:
            suppress = True
            reasons.append("incompatible_domains")
    if ic.get("passenger_load_variable") and ic.get("cargo_over_cabin"):
        if not explicit:
            suppress = True
            reasons.append("planning_hierarchy_unstable")

    if isinstance(data_used, dict):
        resolution = data_used.get("mission_structure_resolution") or {}
        if isinstance(resolution, dict) and resolution.get("decomposition_required"):
            if not explicit:
                suppress = True
                reasons.append("structural_decomposition_required")

    if packet is not None and not packet.recommend_aircraft and not explicit:
        suppress = True
        reasons.append("mission_understanding_gate")

    if ic.get("defer_global_shortlist") and not explicit:
        suppress = True
        reasons.append("defer_global_shortlist")

    if isinstance(data_used, dict) and data_used.get("orchestration_suppresses_aircraft") and not explicit:
        suppress = True
        if "orchestration_suppresses_aircraft" not in reasons:
            reasons.append("orchestration_suppresses_aircraft")

    if suppression is None and isinstance(data_used, dict):
        rs = data_used.get("recommendation_suppression")
        if isinstance(rs, dict):
            suppression = RecommendationSuppressionPolicy(
                suppress_aircraft_specificity=bool(rs.get("suppress_aircraft_specificity")),
                permits_aircraft_specificity=bool(rs.get("permits_aircraft_specificity")),
                reason=str(rs.get("reason") or ""),
                render_class_bands_only=bool(rs.get("render_class_bands_only")),
            )

    if suppression and suppression.suppress_aircraft_specificity and not explicit:
        suppress = True
        if suppression.reason:
            reasons.append(suppression.reason)

    recs = list(recommendations)
    if suppress:
        recs = filter_suppressed_recommendations(recs, RecommendationSuppressionPolicy(
            suppress_aircraft_specificity=True,
            permits_aircraft_specificity=False,
            reason="gate",
            render_class_bands_only=True,
        ))
        if not explicit:
            interpretation_only = True
    else:
        recs = _filter_economics_first_recommendations(
            recs,
            query=query,
            packet=packet,
            data_used=data_used,
        )

    # Anti-generic dump: strip known default models when structure unresolved
    anti_generic = False
    if suppress or (packet and not packet.recommend_aircraft) or _economics_over_prestige_query(query):
        generic = [r for r in recs if is_generic_dump_model(r.model or "")]
        if generic and (len(recs) <= 3 or _economics_over_prestige_query(query)):
            recs = [r for r in recs if not is_generic_dump_model(r.model or "")]
            anti_generic = True
            reasons.append("anti_generic_aircraft_guard")

    # Economics / cost rejection: never leave ULR-only shortlist when user rejected large jets
    if _economics_over_prestige_query(query) and recs:
        ulr_only = all(
            _ULR_MODEL_RE.search(r.model or "")
            or str(getattr(r, "category", "") or "").lower() in _ULR_CATEGORIES
            for r in recs
        )
        if ulr_only:
            recs = []

    return RecommendationGateResult(
        suppress_aircraft=suppress,
        reason="; ".join(dict.fromkeys(reasons)) if reasons else "",
        render_interpretation_only=interpretation_only,
        filtered_recommendations=recs,
        anti_generic_applied=anti_generic,
    )


def finalize_recommendations(
    query: str,
    recommendations: Sequence[AircraftRecommendation],
    mission: MissionState,
    *,
    data_used: Optional[Dict[str, Any]] = None,
    packet: Optional[MissionUnderstandingPacket] = None,
    response_mode: Optional[OrchestrationResponseModeResult] = None,
    max_results: int = 5,
) -> RecommendationGateResult:
    """
    Gate + tier recovery + multi-factor enrichment — single orchestration entry point.
    """
    from services.recommendation.multi_factor_ranking import enrich_recommendations_multi_factor
    from services.recommendation.tier_downgrade_recovery import tier_downgrade_recovery

    gate = evaluate_recommendation_gate(
        query,
        recommendations,
        data_used=data_used,
        packet=packet,
        response_mode=response_mode,
    )

    recs = list(gate.filtered_recommendations)
    from services.orchestration.response_mode_classifier import explicit_aircraft_request
    from services.recommendation.hack_v1_constraint_kernel import hack_v1_constraint_empty

    try:
        from services.orchestration.orchestration_router_v2 import (
            load_orchestration_v2,
            OrchestrationQueryTypeV2,
        )

        v2 = load_orchestration_v2(data_used)
        if v2 is not None and v2.authoritative and not v2.allow_recommendation_ranking:
            gate.filtered_recommendations = []
            gate.suppress_aircraft = True
            if isinstance(data_used, dict):
                data_used["recommendation_gate_v2"] = v2.query_type.value
            return gate
    except Exception:
        pass

    # Stabilization lock: impossible structure must not trigger fallback recovery.
    if isinstance(data_used, dict) and data_used.get("mission_hard_invalid"):
        gate.filtered_recommendations = []
        gate.suppress_aircraft = True
        gate.render_interpretation_only = True
        data_used["tier_downgrade_blocked"] = data_used.get("tier_downgrade_blocked") or "mission_hard_invalid"
        return gate

    explicit = explicit_aircraft_request(query)
    if hack_v1_constraint_empty(data_used):
        gate.filtered_recommendations = []
        gate.suppress_aircraft = False
        if isinstance(data_used, dict):
            data_used["tier_downgrade_blocked"] = "hack_v1_empty"
        return gate
    if gate.suppress_aircraft and not explicit:
        gate.filtered_recommendations = []
        return gate
    if gate.suppress_aircraft and explicit:
        gate.suppress_aircraft = False
        gate.render_interpretation_only = False
        recs = []

    if not recs:
        from services.recommendation.hack_v1_constraint_kernel import load_hack_v1_result

        hack_loaded = load_hack_v1_result(data_used)
        feasible_seed = (
            list(hack_loaded.feasible_aircraft_list or [])
            if hack_loaded is not None
            else []
        )
        if feasible_seed:
            from services.recommendation.mission_ranker import rank_missions
            from services.recommendation.tier_downgrade_recovery import (
                _economics_exclude_ulr,
                _exclude_ulr_models,
            )

            if _economics_exclude_ulr(query, data_used):
                feasible_seed = _exclude_ulr_models(feasible_seed)

        if feasible_seed:
            _, hack_recs, _, _ = rank_missions(
                mission,
                candidate_models=feasible_seed,
                max_results=max_results,
                data_used=data_used,
                query=query,
            )
            recs = [r for r in hack_recs if not r.avoid]
            if isinstance(data_used, dict) and recs:
                data_used["tier_downgrade_recovery"] = {
                    "tier": "hack_v1_feasible",
                    "source": "finalize_hack_v1_seed",
                    "count": len(recs),
                }

    if not recs:
        recovered, _tier = tier_downgrade_recovery(
            mission,
            query,
            prior_recommendations=[],
            data_used=data_used,
            max_results=max_results,
        )
        recs = recovered
        if isinstance(data_used, dict):
            data_used["tier_downgrade_applied"] = True
            if not data_used.get("tier_downgrade_recovery"):
                data_used["tier_downgrade_recovery"] = {"tier": _tier, "source": "finalize_recommendations"}

    recs = enrich_recommendations_multi_factor(
        recs,
        mission=mission,
        packet=packet,
        query=query,
        data_used=data_used,
    )

    # HACK v2 — unify scoring + verdict labels + strict composite ranking
    from services.recommendation.hack_v1_constraint_kernel import load_hack_v1_result
    from services.recommendation.hack_v2_unified_ranking import (
        RankingIntegrityError,
        hack_v2_unify_rank_and_verdict,
    )

    if load_hack_v1_result(data_used) is not None:
        try:
            contract_rows = hack_v2_unify_rank_and_verdict(
                mission=mission,
                recommendations=recs,
                packet=packet,
                query=query,
                data_used=data_used,
                max_results=max_results,
            )
            if isinstance(data_used, dict):
                data_used["hack_v2_ranking"] = contract_rows
            if contract_rows:
                ordered_models = [r["aircraft_name"] for r in contract_rows]
                model_to_rec = {r.model: r for r in recs if getattr(r, "model", None)}
                ordered_recs = [model_to_rec[m] for m in ordered_models if m in model_to_rec]
                for i, rec in enumerate(ordered_recs, start=1):
                    rec.rank = i
                    rec.avoid = False
                recs = ordered_recs
            else:
                recs = []
        except RankingIntegrityError:
            raise
        except Exception:
            pass

    gate.filtered_recommendations = recs[:max_results]
    return gate


def strip_aircraft_from_response(text: str) -> str:
    """Final safety net — remove aircraft model names and luxury marketing from text."""
    if not (text or "").strip():
        return text
    cleaned = _GENERIC_AIRCRAFT_RE.sub("", text)
    cleaned = _LUXURY_MARKETING_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def apply_recommendation_gate_metadata(
    data_used: Dict[str, Any],
    gate: RecommendationGateResult,
) -> None:
    data_used["recommendation_gate"] = gate.to_dict()
    if gate.suppress_aircraft:
        data_used["recommend_aircraft_gated"] = 0
        data_used["orchestration_suppresses_aircraft"] = True


def render_interpretation_first_response(
    mission,
    packet: MissionUnderstandingPacket,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
    route_certainty_degraded: bool = False,
) -> str:
    """
    Structural-first renderer — interpretation before any aircraft discussion.
    """
    from services.mission.mission_authority_kernel import build_mission_authority_kernel
    from services.mission.mission_interpretation_formatter import format_mission_interpretation
    from services.orchestration.hierarchy_weighting import (
        attach_hierarchy_weighting_metadata,
        detect_dominant_mission,
        format_hierarchy_weighting_section,
    )

    hierarchy = detect_dominant_mission(packet, query=query, data_used=data_used)
    attach_hierarchy_weighting_metadata(data_used, hierarchy)

    projection_trace = None
    if isinstance(data_used, dict) and isinstance(data_used.get("ranking_projection_trace"), dict):
        try:
            from services.mission.mission_ranking_projection import RankingProjectionTrace

            raw = data_used["ranking_projection_trace"]
            projection_trace = RankingProjectionTrace(
                segment_isolated=bool(raw.get("segment_isolated")),
                suppressed_global_flags=list(raw.get("suppressed_global_flags") or []),
                peak_leg_nm=float(raw.get("peak_leg_nm") or 0),
                route_display_order=list(raw.get("route_display_order") or []),
            )
        except Exception:
            projection_trace = None

    kernel = build_mission_authority_kernel(
        mission,
        packet,
        recommendations=[],
        query=query or "",
        data_used=data_used,
        route_certainty_degraded=route_certainty_degraded,
        projection_trace=projection_trace,
    )

    body = format_mission_interpretation(
        mission,
        packet,
        kernel,
        query=query or "",
        data_used=data_used,
        hierarchy=hierarchy,
    )
    hierarchy_block = format_hierarchy_weighting_section(hierarchy)
    if hierarchy_block and hierarchy_block not in body:
        # Insert hierarchy after operational structure per structural-first contract
        parts = body.split("\n\nPrimary Utilization", 1)
        if len(parts) == 2:
            body = f"{parts[0]}\n\n{hierarchy_block}\n\nPrimary Utilization{parts[1]}"
        else:
            body = f"{body}\n\n{hierarchy_block}"

    from services.consultant.dispatch_conflict_renderer import format_dispatch_conflict_block

    conflict = format_dispatch_conflict_block(packet, data_used=data_used, query=query or "")
    if conflict:
        body = f"{body}\n\n{conflict}"

    return strip_aircraft_from_response(body)


__all__ = [
    "RecommendationGateResult",
    "apply_recommendation_gate_metadata",
    "evaluate_recommendation_gate",
    "finalize_recommendations",
    "render_interpretation_first_response",
    "strip_aircraft_from_response",
]
