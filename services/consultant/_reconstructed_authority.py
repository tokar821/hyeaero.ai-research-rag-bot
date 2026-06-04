"""
Enforce pipeline authority on user-facing answers — LLM/RAG cannot name aircraft
outside the deterministic shortlist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from services.consultant.comparison_engine import StructuredComparison
from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.consultant.route_feasibility import RouteFeasibilityAssessment

logger = logging.getLogger(__name__)

EMPTY_PIPELINE_AUTHORITY_MESSAGE = "No aircraft passed mission filters."
UNAUTHORIZED_LOG_EVENT = "UNAUTHORIZED_AIRCRAFT_REFERENCE"

_RANKED_INTENTS = frozenset(
    {
        "acquisition_recommendation",
        "mission_feasibility",
        "aircraft_comparison",
        "operational_tradeoff_analysis",
        "shortlist_ranking",
    }
)


@dataclass
class UnauthorizedAircraftReference:
    aircraft: str
    source: str

    def to_dict(self) -> Dict[str, str]:
        return {"aircraft": self.aircraft, "source": self.source}


@dataclass
class RecommendationAuthority:
    """Whitelist of aircraft the explanation layer may name."""

    approved_shortlist: Set[str] = field(default_factory=set)
    pipeline_candidates: Set[str] = field(default_factory=set)
    comparison_models: Set[str] = field(default_factory=set)
    hard_excluded: Set[str] = field(default_factory=set)

    @property
    def allowed_models(self) -> Set[str]:
        allowed = set(self.approved_shortlist) | set(self.comparison_models)
        allowed |= {m for m in self.pipeline_candidates if m in self.approved_shortlist}
        if self.hard_excluded:
            allowed -= set(self.hard_excluded)
        return allowed

    @property
    def strict_empty_shortlist(self) -> bool:
        return not self.approved_shortlist

    @classmethod
    def from_pipeline(
        cls,
        recommendations: Sequence[AircraftRecommendation],
        *,
        data_used: Optional[Dict[str, Any]] = None,
        comparison_models: Optional[Sequence[str]] = None,
        hard_excluded: Optional[Set[str]] = None,
    ) -> RecommendationAuthority:
        viable = [r.model for r in recommendations if not getattr(r, "avoid", False) and r.model]
        approved = set(viable)
        candidates: Set[str] = set()

        if isinstance(data_used, dict):
            for key in ("final_ranked_aircraft", "approved_shortlist"):
                raw = data_used.get(key)
                if isinstance(raw, list):
                    approved.update(str(m) for m in raw if m)
            raw_candidates = data_used.get("pipeline_candidates")
            if isinstance(raw_candidates, list):
                candidates.update(str(m) for m in raw_candidates if m)
            pipe = data_used.get("deterministic_recommendation_pipeline")
            if isinstance(pipe, dict):
                for m in pipe.get("feasible_models") or []:
                    candidates.add(str(m))
                for rec in pipe.get("recommendations") or []:
                    if isinstance(rec, dict) and rec.get("model") and not rec.get("avoid"):
                        approved.add(str(rec["model"]))
            trace = data_used.get("recommendation_pipeline")
            if isinstance(trace, dict):
                for m in trace.get("ranked_models") or []:
                    approved.add(str(m))

        comp = {str(m) for m in (comparison_models or []) if m}
        excluded = set(hard_excluded or [])
        return cls(
            approved_shortlist=approved,
            pipeline_candidates=candidates,
            comparison_models=comp,
            hard_excluded=excluded,
        )

    def detect_unauthorized(self, text: str) -> List[str]:
        if not (text or "").strip():
            return []
        try:
            from services.consultant.recommendation_engine import detect_models_from_text
        except Exception:
            return []

        mentioned = list(detect_models_from_text(text))
        allowed = self.allowed_models
        if self.strict_empty_shortlist:
            return mentioned
        if not allowed:
            return []
        return [m for m in mentioned if m not in allowed]

    def log_unauthorized(self, violations: List[str], *, source: str) -> List[UnauthorizedAircraftReference]:
        refs: List[UnauthorizedAircraftReference] = []
        for aircraft in violations:
            ref = UnauthorizedAircraftReference(aircraft=aircraft, source=source)
            refs.append(ref)
            logger.warning(
                "%s aircraft=%s source=%s",
                UNAUTHORIZED_LOG_EVENT,
                aircraft,
                source,
            )
        return refs

    def record_violations(
        self,
        violations: List[str],
        *,
        source: str,
        data_used: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not violations or not isinstance(data_used, dict):
            return
        refs = self.log_unauthorized(violations, source=source)
        log = data_used.setdefault("unauthorized_aircraft_references", [])
        if isinstance(log, list):
            log.extend(r.to_dict() for r in refs)

    def enforce(
        self,
        text: str,
        *,
        source: str = "llm_explanation",
        data_used: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[str]]:
        violations = self.detect_unauthorized(text)
        if violations:
            self.record_violations(violations, source=source, data_used=data_used)
        return text, violations


def is_ranked_recommendation_query(
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> bool:
    """Detect ranked recommendation intent from metadata or query classification."""
    if isinstance(data_used, dict):
        intent = str(data_used.get("query_recommendation_intent") or "").strip().lower()
        if intent in _RANKED_INTENTS:
            return True
        if data_used.get("query_recommendation_requires_pipeline"):
            return True
    if not (query or "").strip():
        return False
    try:
        from services.recommendation.query_recommendation_intent import (
            classify_query_recommendation_intent,
            requires_ranked_aircraft_pipeline,
        )

        qri = classify_query_recommendation_intent(query)
        return requires_ranked_aircraft_pipeline(qri.intent)
    except Exception:
        return False


def requires_recommendation_aircraft_authority(
    data_used: Optional[Dict[str, Any]],
    *,
    query: str = "",
) -> bool:
    """True when this turn must not name aircraft outside the deterministic pipeline."""
    if is_ranked_recommendation_query(query, data_used):
        return True
    if not isinstance(data_used, dict):
        return False
    if data_used.get("pre_llm_pipeline_authority") or data_used.get("pipeline_authority_enforced"):
        return True
    if data_used.get("block_aircraft_substitution"):
        intent = str(data_used.get("query_recommendation_intent") or "").strip().lower()
        if intent in _RANKED_INTENTS:
            return True
        if data_used.get("query_recommendation_requires_pipeline"):
            return True
    return False


def enforce_orchestration_recommendation_authority(
    data_used: Dict[str, Any],
    query: str = "",
) -> bool:
    """
    Orchestration-level law: set authority flags before any LLM stage when ranked intent applies.
    """
    if not is_ranked_recommendation_query(query, data_used):
        return False
    data_used["pipeline_authority_enforced"] = True
    data_used["block_aircraft_substitution"] = True
    try:
        from services.recommendation.query_recommendation_intent import (
            apply_query_intent_metadata,
            classify_query_recommendation_intent,
        )

        qri = classify_query_recommendation_intent(query)
        apply_query_intent_metadata(data_used, qri)
    except Exception:
        pass
    return True


def should_suppress_aviation_engines_catalog(
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
    *,
    fine_intent: str = "",
) -> bool:
    """Block RAG catalog aircraft lists for ranked recommendation workflows."""
    du = data_used if isinstance(data_used, dict) else {}
    if du.get("deterministic_pre_llm_executed") or du.get("pipeline_llm_facts"):
        return True
    if isinstance(du.get("deterministic_recommendation_pipeline"), dict):
        return True
    fi = (fine_intent or "").strip().lower()
    if fi not in ("aircraft_recommendation", "aviation_mission", "aircraft_comparison"):
        return False
    return is_ranked_recommendation_query(query, data_used)


def filter_recommendations_to_authority(
    recommendations: Sequence[AircraftRecommendation],
    *,
    data_used: Optional[Dict[str, Any]] = None,
    comparison_models: Optional[Sequence[str]] = None,
    query: str = "",
) -> List[AircraftRecommendation]:
    """Drop ranked rows that are not on the approved whitelist."""
    if not requires_recommendation_aircraft_authority(data_used, query=query):
        return list(recommendations)
    auth = RecommendationAuthority.from_pipeline(
        recommendations,
        data_used=data_used,
        comparison_models=comparison_models,
    )
    allowed = auth.allowed_models
    if not allowed:
        return []
    return [r for r in recommendations if r.model in allowed and not r.avoid]


def format_authority_interpretation_only_response(
    mission: MissionState,
    *,
    operational_synthesis: str = "",
    data_used: Optional[Dict[str, Any]] = None,
    query: str = "",
) -> str:
    """Mission interpretation only — no aircraft names, classes, or category hints."""
    del query
    lines = ["Mission Interpretation", ""]
    if operational_synthesis.strip():
        lines.append(f"- {operational_synthesis.strip()}")
    else:
        lines.append("- Operational read is constrained by stated route, passengers, and reserves.")
    route = mission.routes[0] if mission.routes else ""
    if route:
        pax = mission.passenger_count
        pax_bit = f" with {pax} passengers" if pax is not None else ""
        lines.append(f"- Primary stage: {route}{pax_bit}.")
    lines.extend(
        [
            "",
            "Constraint Summary",
            "",
        ]
    )
    lines.extend(_constraint_summary(data_used, omit_aircraft_names=True))
    lines.append("")
    lines.append(EMPTY_PIPELINE_AUTHORITY_MESSAGE)
    lines.append(
        "Re-run feasibility after a constraint changes — the deterministic pipeline "
        "is the only source for aircraft names on this turn."
    )
    from services.consultant.response_formatter import sanitize_advisor_output

    return sanitize_advisor_output("\n".join(lines))


def apply_final_answer_authority(
    answer: str,
    *,
    mission: MissionState,
    recommendations: List[AircraftRecommendation],
    data_used: Optional[Dict[str, Any]] = None,
    query: str = "",
    comparison_models: Optional[List[str]] = None,
    source: str = "final_response",
) -> str:
    """Last-line guard on user-visible text for ranked recommendation workflows."""
    if not requires_recommendation_aircraft_authority(data_used, query=query):
        return answer
    viable = [r for r in recommendations if not r.avoid]
    authority = RecommendationAuthority.from_pipeline(
        viable,
        data_used=data_used,
        comparison_models=comparison_models,
    )
    violations = authority.detect_unauthorized(answer or "")
    if not violations:
        return answer
    authority.record_violations(violations, source=source, data_used=data_used)
    if not viable:
        return format_empty_pipeline_authority_response(mission, data_used=data_used, query=query)
    final, _ = reconcile_answer_with_pipeline(
        answer,
        mission=mission,
        recommendations=viable,
        query=query,
        comparison_models=list(comparison_models) if comparison_models else None,
        data_used=data_used,
    )
    return final


def attach_recommendation_authority_metadata(
    data_used: Dict[str, Any],
    recommendations: Sequence[AircraftRecommendation],
    *,
    feasible_models: Optional[Sequence[str]] = None,
    query: str = "",
) -> None:
    """Persist whitelist keys consumed by :class:`RecommendationAuthority`."""
    enforce_orchestration_recommendation_authority(data_used, query)
    ranked = [r.model for r in recommendations if not getattr(r, "avoid", False) and r.model]
    data_used["final_ranked_aircraft"] = list(ranked)
    data_used["approved_shortlist"] = list(ranked)
    if feasible_models is not None:
        data_used["pipeline_candidates"] = list(feasible_models)
    elif not data_used.get("pipeline_candidates"):
        pipe = data_used.get("deterministic_recommendation_pipeline")
        if isinstance(pipe, dict) and pipe.get("feasible_models"):
            data_used["pipeline_candidates"] = list(pipe["feasible_models"])


def allowed_recommendation_models(
    recommendations: List[AircraftRecommendation],
    *,
    comparison_models: Optional[List[str]] = None,
    hard_excluded: Optional[Set[str]] = None,
) -> Set[str]:
    """Models the narrator may name as recommendations."""
    auth = RecommendationAuthority.from_pipeline(
        recommendations,
        comparison_models=comparison_models,
        hard_excluded=hard_excluded,
    )
    return auth.allowed_models


def detect_unauthorized_aircraft(
    text: str,
    allowed: Set[str],
    *,
    strict_empty: bool = False,
) -> List[str]:
    """Models mentioned in prose but not in the pipeline shortlist."""
    auth = RecommendationAuthority(
        approved_shortlist=set(allowed) if allowed else set(),
        comparison_models=set(),
    )
    if strict_empty:
        auth = RecommendationAuthority(approved_shortlist=set())
    return auth.detect_unauthorized(text)


def _constraint_summary(
    data_used: Optional[Dict[str, Any]],
    *,
    omit_aircraft_names: bool = False,
) -> List[str]:
    lines: List[str] = []
    if not isinstance(data_used, dict):
        return lines
    pipe = data_used.get("deterministic_recommendation_pipeline")
    if isinstance(pipe, dict):
        for entry in (pipe.get("elimination_log") or [])[-6:]:
            if not isinstance(entry, dict):
                continue
            aircraft = entry.get("aircraft_name") or entry.get("model")
            reason = entry.get("reason") or entry.get("mission_constraint_failed")
            constraint = str(entry.get("mission_constraint_failed") or "").strip()
            if omit_aircraft_names:
                if constraint:
                    lines.append(f"• {constraint.replace('_', ' ')}")
                elif entry.get("summary"):
                    lines.append(f"• {entry.get('summary')}")
                continue
            if aircraft and reason:
                lines.append(f"• {aircraft}: {reason}")
            elif entry.get("summary"):
                lines.append(f"• {entry.get('summary')}")
    audit = data_used.get("recommendation_audit")
    if isinstance(audit, dict):
        summary = audit.get("elimination_summary") or audit.get("primary_constraint")
        if summary:
            lines.append(f"• {summary}")
    if not lines:
        lines.append(
            "• Typical blockers: range with NBAA reserves, runway or hot/high limits, "
            "passenger/payload load, or winter westbound margin on long east-west legs."
        )
    return lines[:8]


def format_empty_pipeline_authority_response(
    mission: MissionState,
    *,
    data_used: Optional[Dict[str, Any]] = None,
    query: str = "",
) -> str:
    """Deterministic empty shortlist — no invented aircraft."""
    del query
    lines = [EMPTY_PIPELINE_AUTHORITY_MESSAGE, ""]
    route = ""
    if mission.routes:
        route = mission.routes[0]
    pax = mission.passenger_count
    if route:
        opener = f"For {route}"
        if pax is not None:
            opener += f" with {pax} passengers"
        lines.append(opener + ", every catalog candidate failed at least one hard gate:")
    else:
        lines.append("Every catalog candidate failed at least one hard gate:")
    lines.extend(_constraint_summary(data_used, omit_aircraft_names=True))
    lines.append("")
    lines.append(
        "If a constraint can move (nonstop requirement, payload, runway, season), "
        "state which one and the shortlist can be re-run — without substituting aircraft "
        "outside the feasibility pipeline."
    )
    from services.consultant.response_formatter import sanitize_advisor_output

    return sanitize_advisor_output("\n".join(lines))


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
    LLM may format prose only. Reject merges that name aircraft outside the pipeline shortlist.
    """
    viable = [r for r in recommendations if not r.avoid]
    authority = RecommendationAuthority.from_pipeline(
        viable,
        data_used=data_used,
        comparison_models=comparison_models,
        hard_excluded=hard_excluded,
    )
    ranked_authority = requires_recommendation_aircraft_authority(data_used)

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
                    if ranked_authority and not viable:
                        return format_empty_pipeline_authority_response(
                            mission, data_used=data_used, query=query
                        ), True
                    return enforced, True

            if not (answer or "").strip():
                return canonical, True
    except Exception:
        pass

    if not viable:
        if ranked_authority:
            empty_body = format_empty_pipeline_authority_response(
                mission, data_used=data_used, query=query
            )
            if (answer or "").strip():
                violations = authority.detect_unauthorized(answer)
                if violations:
                    authority.record_violations(
                        violations, source="reconcile_empty_shortlist", data_used=data_used
                    )
            return empty_body, True
        return answer, False

    violations = authority.detect_unauthorized(answer or "")
    if violations:
        authority.record_violations(violations, source="reconcile_llm_merge", data_used=data_used)

    from services.consultant.response_format_validation import validateResponseFormatting

    fmt = validateResponseFormatting(answer, recommendations=viable)
    needs_regen = bool(violations) or not fmt.ok

    try:
        from services.telemetry.reasoning_packet_enforcement import (
            enforce_reasoning_packet_authority,
            extract_reasoning_packet,
        )

        packet = extract_reasoning_packet(data_used)
        if packet and (violations or not fmt.ok):
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
                post_violations = authority.detect_unauthorized(answer)
                if post_violations:
                    authority.record_violations(
                        post_violations,
                        source="reasoning_packet_enforcement",
                        data_used=data_used,
                    )
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
                final_violations = authority.detect_unauthorized(answer)
                if final_violations:
                    authority.record_violations(
                        final_violations,
                        source="kernel_post_enforce",
                        data_used=data_used,
                    )
                    needs_regen = True
        except Exception:
            pass
        if not needs_regen:
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
