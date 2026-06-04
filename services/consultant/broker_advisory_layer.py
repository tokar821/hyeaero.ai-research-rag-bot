"""
Broker-style advisory layer — top-tier acquisition consultant voice.

Deterministic pipeline owns feasibility; this layer formats narration only.
The LLM receives mission facts + feasible aircraft + metadata — never invents feasibility.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from services.consultant.mission_state import MissionState, normalize_routes
from services.consultant.recommendation_engine import AircraftRecommendation, _AIRCRAFT_PROFILES
from services.consultant.route_feasibility import RouteFeasibilityAssessment
from services.recommendation.fit_policy import FIT_GOOD, FIT_PARTIAL, FIT_STRONG, STANDARD_RECOMMENDATION_LIMIT

MAX_BROKER_AIRCRAFT = STANDARD_RECOMMENDATION_LIMIT  # 3

FINAL_OUTPUT_MARKER = "===FINAL_BROKER_OUTPUT==="
_LLM_SANITIZE_FALLBACK = "INSUFFICIENT_DATA: No verified aircraft available."
_ADVISORY_CONTEXT_LEAK_RE = re.compile(r"\[BROKER\s+ADVISORY\s+CONTEXT", re.I)


def sanitize_llm_output(text: str) -> str:
    """
    Strip prompt-injected advisory blocks and return only post-marker client prose.
    """
    raw = (text or "").strip()
    if not raw:
        return _LLM_SANITIZE_FALLBACK
    if _ADVISORY_CONTEXT_LEAK_RE.search(raw) and FINAL_OUTPUT_MARKER not in raw:
        return _LLM_SANITIZE_FALLBACK
    if FINAL_OUTPUT_MARKER in raw:
        raw = raw.split(FINAL_OUTPUT_MARKER, 1)[-1].strip()
    if not raw or _ADVISORY_CONTEXT_LEAK_RE.search(raw):
        return _LLM_SANITIZE_FALLBACK
    return raw


def extract_final_broker_output(text: str) -> str:
    """Return client-facing segment after FINAL_OUTPUT_MARKER when present."""
    return sanitize_llm_output(text)


def _append_mission_portfolio_sections(
    body: str,
    mission: MissionState,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Legacy hook — narrative authority owns fleet/ownership; no duplicate injection."""
    du = data_used if isinstance(data_used, dict) else {}
    from services.mission.narrative_authority import NARRATIVE_AUTHORITY_KEY

    from services.mission.mission_authority_kernel import MISSION_AUTHORITY_KERNEL_KEY

    if (
        du.get("narrative_authority_built")
        or du.get("mission_authority_bound")
        or du.get(NARRATIVE_AUTHORITY_KEY)
        or du.get(MISSION_AUTHORITY_KERNEL_KEY)
    ):
        return body

    sections: List[str] = []

    fleet_raw = du.get("fleet_composition_plan")
    if isinstance(fleet_raw, dict) and (
        fleet_raw.get("multi_aircraft_required") or fleet_raw.get("single_aircraft_structurally_invalid")
    ):
        try:
            from services.fleet.fleet_composition import (
                FleetCompositionPlan,
                FleetRoleAssignment,
                MissionSegment,
                MissionSegmentRole,
                format_fleet_composition_block,
            )

            plan = FleetCompositionPlan(
                multi_aircraft_required=bool(fleet_raw.get("multi_aircraft_required")),
                doctrine=str(fleet_raw.get("doctrine") or ""),
                ownership_note=str(fleet_raw.get("ownership_note") or ""),
                single_aircraft_structurally_invalid=bool(
                    fleet_raw.get("single_aircraft_structurally_invalid")
                ),
            )
            for s in fleet_raw.get("segments") or []:
                if isinstance(s, dict):
                    try:
                        seg_role = MissionSegmentRole(str(s.get("role") or "coast_to_coast"))
                    except ValueError:
                        seg_role = MissionSegmentRole.COAST_TO_COAST
                    plan.segments.append(
                        MissionSegment(
                            role=seg_role,
                            label=str(s.get("label") or ""),
                            stage_nm=float(s.get("stage_nm") or 0),
                            required_nm=float(s.get("required_nm") or 0),
                            route_labels=list(s.get("route_labels") or []),
                        )
                    )
            for a in fleet_raw.get("assignments") or []:
                if isinstance(a, dict):
                    try:
                        role = MissionSegmentRole(str(a.get("role") or "coast_to_coast"))
                    except ValueError:
                        role = MissionSegmentRole.COAST_TO_COAST
                    plan.assignments.append(
                        FleetRoleAssignment(
                            role=role,
                            segment_label=str(a.get("segment_label") or ""),
                            primary_model=str(a.get("primary_model") or ""),
                            fit_verdict=str(a.get("fit_verdict") or ""),
                            rationale=str(a.get("rationale") or ""),
                            alternates=list(a.get("alternates") or []),
                        )
                    )
            fleet_block = format_fleet_composition_block(plan)
            if fleet_block:
                sections.append(fleet_block)
        except Exception:
            pass

    try:
        from services.mission.mission_understanding_engine import (
            load_mission_understanding,
            needs_ownership_overlay,
            needs_portfolio_synthesis,
            format_portfolio_synthesis,
            format_ownership_economics_overlay,
        )

        packet = load_mission_understanding(du)
        if packet is not None:
            if not sections and needs_portfolio_synthesis(query, packet):
                sections.append(format_portfolio_synthesis(query, packet))
            if needs_ownership_overlay(query, packet):
                sections.append(format_ownership_economics_overlay(query, mission))
    except Exception:
        pass

    if not sections:
        return body
    return body + "\n\n" + "\n\n".join(sections)


# User-forbidden broker clichés / middleware tone
BROKER_FORBIDDEN_PHRASES: tuple[str, ...] = (
    r"\bmission profile\b",
    r"\bmission score\b",
    r"\bconfidence score\b",
    r"\boperationally\b",
    r"\bworth considering\b",
    r"\bif priorities shift\b",
    r"\bstage length\b",
    r"\bbalanced capability\b",
    r"\bfits your mission pattern\b",
    r"\bmission requirements\b",
    r"\boperational capability\b",
    r"\boperationally balanced\b",
    r"\bcredible alternate if priorities shift\b",
    r"\bset the mission profile\b",
    r"\bframe the mission\b",
    r"\bpressure-test in that class\b",
    r"\bwhere I'?d narrow after\b",
    r"\bclass choice follows\b",
    r"\bthe band\b",
    r"\bshopping\b",
)

_BROKER_FORBIDDEN_RE = re.compile("|".join(BROKER_FORBIDDEN_PHRASES), re.I)

from services.broker.broker_verdicts import BrokerVerdict, normalize_broker_verdict

_VERDICT_PRIMARY = BrokerVerdict.PRIMARY_RECOMMENDATION.value
_VERDICT_VIABLE = BrokerVerdict.VIABLE_WITH_COMPROMISES.value
_VERDICT_RISKY = BrokerVerdict.MISSION_RISKY.value
_VERDICT_NOT = BrokerVerdict.NOT_OPERATIONALLY_CREDIBLE.value


@dataclass
class BrokerAircraftBrief:
    model: str
    category: str
    practical_nm: float
    pax_typical: int
    runway_ft: float
    fit_verdict: str
    critique: str = ""


@dataclass
class BrokerAdvisoryContext:
    """Inputs allowed for LLM narration — no scores, no eliminated aircraft as recommendations."""

    mission_summary: str
    route_label: str
    passengers: Optional[int]
    constraints: List[str] = field(default_factory=list)
    feasible_aircraft: List[BrokerAircraftBrief] = field(default_factory=list)

    def to_llm_block(self) -> str:
        """Structured facts for the LLM context layer — not a client-facing template."""
        lines = [
            "[VERIFIED MISSION FACTS — pre-validated shortlist; narrate in natural broker prose]",
            f"mission_summary: {self.mission_summary}",
        ]
        if self.route_label:
            lines.append(f"route: {self.route_label}")
        if self.passengers is not None:
            lines.append(f"passengers: {self.passengers}")
        if self.constraints:
            lines.append("constraints: " + "; ".join(self.constraints))
        lines.append(f"feasible_aircraft_max: {MAX_BROKER_AIRCRAFT}")
        for ac in self.feasible_aircraft[:MAX_BROKER_AIRCRAFT]:
            row = (
                f"- model={ac.model!r} class={ac.category!r} practical_nm={int(ac.practical_nm)} "
                f"pax_typical={ac.pax_typical} runway_ft={int(ac.runway_ft)} "
                f"broker_verdict={ac.fit_verdict!r}"
            )
            if ac.critique:
                row += f" note={ac.critique[:240]!r}"
            lines.append(row)
        lines.append(
            "rules: Use only these aircraft; do not add/remove/re-score. "
            "Write one cohesive answer — no Mission Fit / Aircraft Options / Verdict headings."
        )
        return "\n".join(lines).strip()


def broker_verdict_label(rec: AircraftRecommendation) -> str:
    """Map internal fit tier to broker verdict bucket."""
    fv = (rec.fit_verdict or "").strip()
    if fv:
        return normalize_broker_verdict(fv).value

    fit = (rec.fit or "").strip()
    if rec.avoid or fit == "Not Recommended":
        return _VERDICT_NOT
    if fit in (FIT_STRONG, FIT_GOOD, "Strong Fit", "Good Fit"):
        return _VERDICT_PRIMARY
    if fit in (FIT_PARTIAL, "Partial Fit"):
        return _VERDICT_VIABLE
    if rec.total_score >= 0.62:
        return _VERDICT_PRIMARY
    if rec.total_score >= 0.48:
        return _VERDICT_VIABLE
    if rec.total_score < 0.40:
        return _VERDICT_RISKY
    return _VERDICT_VIABLE


def _aircraft_metadata(model: str) -> Optional[Dict[str, Any]]:
    from services.aircraft_truth import validate_aircraft_truth

    truth = validate_aircraft_truth(model)
    if not truth.verified or not truth.facts:
        return None
    facts = truth.facts
    spec = dict(_AIRCRAFT_PROFILES.get(model) or {})
    return {
        "category": facts.operating_category,
        "practical_nm": facts.practical_range_nm,
        "pax_typical": facts.max_passengers,
        "runway_ft": float(spec.get("runway_ft") or 0),
        "runway_class": facts.runway_class,
        "baggage_volume_cu_ft": facts.baggage_volume_cu_ft,
    }


def _route_phrase(mission: MissionState) -> str:
    routes = normalize_routes(mission.routes)
    if not routes:
        return ""
    return routes[0] if len(routes) == 1 else "; ".join(routes[:2])


def _mission_constraints(mission: MissionState) -> List[str]:
    bits: List[str] = []
    if mission.nonstop_requirement:
        bits.append("nonstop required")
    if mission.westbound:
        bits.append("westbound-sensitive")
    if (mission.seasonal_constraints or "").lower().find("winter") >= 0:
        bits.append("winter ops")
    if mission.mountain_airport_requirement:
        bits.append("mountain / hot-high")
    if mission.runway_constraints:
        bits.append(f"runway: {mission.runway_constraints}")
    return bits


def _category_territory_label(recs: Sequence[AircraftRecommendation]) -> str:
    cats: List[str] = []
    for r in recs[:3]:
        meta = _aircraft_metadata(r.model)
        if not meta:
            continue
        cat = meta["category"]
        label = {
            "light": "light jet",
            "super-midsize": "super-mid",
            "large": "large-cabin",
            "ultra-long": "ultra-long-range",
        }.get(cat, cat)
        if label and label not in cats:
            cats.append(label)
    if not cats:
        return "the right cabin class"
    if len(cats) == 1:
        return f"{cats[0]} territory"
    return f"{' / '.join(cats[:2])} territory"


def _opening_line(mission: MissionState, recs: Sequence[AircraftRecommendation]) -> str:
    route = _route_phrase(mission)
    pax = mission.passenger_count
    names = [r.model for r in recs[:3]]

    if route and names:
        if len(names) >= 2:
            pair = " / ".join(names[:2])
            if len(names) > 2:
                pair += f" (and {names[2]})"
            return f"For {route}, you're realistically in {pair} territory."
        return f"For {route}, I'd start with {names[0]}."

    territory = _category_territory_label(recs)
    if pax is not None:
        return f"With {pax} passengers, you're in {territory}."
    return f"For this trip, you're in {territory}."


def _critique_for_aircraft(
    rec: AircraftRecommendation,
    mission: MissionState,
    route_assessments: Sequence[RouteFeasibilityAssessment],
) -> str:
    meta = _aircraft_metadata(rec.model)
    if not meta:
        return ""
    cat = meta["category"]
    parts: List[str] = []

    if rec.explanation:
        for src in (rec.explanation.operational_compromises, rec.explanation.penalties):
            for item in src or []:
                t = (item or "").strip()
                if t and "score" not in t.lower():
                    parts.append(t.rstrip("."))
                    break
            if parts:
                break

    route = _route_phrase(mission)
    if not parts and route and mission.nonstop_requirement:
        if cat in ("super-midsize", "light", "midsize"):
            worst = next((a for a in route_assessments if not a.reliably_nonstop), None)
            if worst:
                parts.append(f"I would not trust it for consistent {route} nonstop with full payload")

    if not parts and rec.rank > 1 and rec.explanation and rec.explanation.why_alternatives_lost:
        parts.append(rec.explanation.why_alternatives_lost[0][:140].rstrip("."))

    if not parts:
        if rec.rank == 1:
            parts.append("best day-to-day match in this shortlist")
        else:
            parts.append("viable alternate if cabin or cost tradeoffs matter")

    return f"{rec.model} — {parts[0]}."


def _fit_footer(recs: Sequence[AircraftRecommendation]) -> str:
    primary: List[str] = []
    viable: List[str] = []
    risky: List[str] = []
    for rec in recs[:MAX_BROKER_AIRCRAFT]:
        bucket = broker_verdict_label(rec)
        if bucket == _VERDICT_PRIMARY:
            primary.append(rec.model)
        elif bucket == _VERDICT_RISKY:
            risky.append(rec.model)
        elif bucket == _VERDICT_VIABLE:
            viable.append(rec.model)
    lines: List[str] = []
    if primary:
        lines.append(f"{_VERDICT_PRIMARY}: {', '.join(primary)}")
    if viable:
        lines.append(f"{_VERDICT_VIABLE}: {', '.join(viable)}")
    if risky:
        lines.append(f"{_VERDICT_RISKY}: {', '.join(risky)}")
    if not lines:
        lines.append(f"{_VERDICT_VIABLE}: {', '.join(r.model for r in recs[:MAX_BROKER_AIRCRAFT])}")
    return "\n\n".join(lines)


def sanitize_broker_prose(text: str) -> str:
    """Strip forbidden broker phrases and internal labels."""
    from services.broker.broker_language import sanitize_broker_language

    if not (text or "").strip():
        return ""
    out = sanitize_llm_output(text)
    if out == _LLM_SANITIZE_FALLBACK:
        return out
    out = _BROKER_FORBIDDEN_RE.sub("", out)
    out = re.sub(r"\bMission Summary\b.*", "", out, flags=re.I)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"  +", " ", out)
    return sanitize_broker_language(out.strip())


def build_broker_advisory_context(
    mission: MissionState,
    recommendations: Sequence[AircraftRecommendation],
    route_assessments: Optional[Sequence[RouteFeasibilityAssessment]] = None,
) -> BrokerAdvisoryContext:
    """Build LLM-safe context: mission + feasible list + metadata only."""
    viable = [r for r in recommendations if not r.avoid][:MAX_BROKER_AIRCRAFT]
    ra = list(route_assessments or [])
    route = _route_phrase(mission)
    constraints = _mission_constraints(mission)

    summary_parts: List[str] = []
    if mission.passenger_count is not None:
        summary_parts.append(f"{mission.passenger_count} passengers")
    if route:
        summary_parts.append(route)
    if constraints:
        summary_parts.append(", ".join(constraints))
    mission_summary = "; ".join(summary_parts) if summary_parts else "Advisory mission intake"

    briefs: List[BrokerAircraftBrief] = []
    for rec in viable:
        meta = _aircraft_metadata(rec.model)
        if not meta:
            continue
        briefs.append(
            BrokerAircraftBrief(
                model=rec.model,
                category=meta["category"],
                practical_nm=meta["practical_nm"],
                pax_typical=meta["pax_typical"],
                runway_ft=meta["runway_ft"],
                fit_verdict=broker_verdict_label(rec),
                critique=_critique_for_aircraft(rec, mission, ra),
            )
        )

    return BrokerAdvisoryContext(
        mission_summary=mission_summary,
        route_label=route,
        passengers=mission.passenger_count,
        constraints=constraints,
        feasible_aircraft=briefs,
    )


def _detail_line(
    rec: AircraftRecommendation,
    mission: MissionState,
    route_assessments: Sequence[RouteFeasibilityAssessment],
) -> str:
    """One factual sentence on the lead aircraft — no marketing."""
    if rec.explanation:
        for src in (rec.explanation.why_it_fits, rec.explanation.strengths):
            for item in src or []:
                t = sanitize_broker_prose((item or "").strip().rstrip("."))
                if t and len(t) > 12 and "score" not in t.lower():
                    return t + "."
    route = _route_phrase(mission)
    if route and mission.nonstop_requirement:
        return f"Handles {route} nonstop with realistic payload and reserve margin."
    return ""


def format_broker_advisory_response(
    mission: MissionState,
    recommendations: Sequence[AircraftRecommendation],
    route_assessments: Optional[Sequence[RouteFeasibilityAssessment]] = None,
    *,
    query: str = "",
    eliminated_models: Optional[Sequence[str]] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Deterministic advisory response — fixed Mission Fit / Aircraft Options / Verdict layout.
    """
    seed = str(query or "").strip()
    from services.elimination.elimination_invariant import (
        collect_eliminated_models,
        enforce_elimination_invariant,
    )

    eliminated = collect_eliminated_models(
        data_used=data_used,
        explicit_eliminated=eliminated_models,
    )
    recommendations = enforce_elimination_invariant(
        recommendations,
        eliminated,
        context="broker_advisory",
    )
    viable = [r for r in recommendations if not r.avoid][:MAX_BROKER_AIRCRAFT]
    from services.mission.mission_understanding_engine import (
        format_understanding_first_advisory,
        load_mission_understanding,
    )

    packet = load_mission_understanding(data_used)
    route_degraded = bool(
        isinstance(data_used, dict) and data_used.get("route_blocks_ranking")
    )
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

    if not viable:
        if packet is not None:
            # Interpretation-only prompts: structure first, no aircraft output.
            if query:
                try:
                    from services.mission.mission_interpretation_formatter import (
                        format_mission_interpretation,
                        is_interpretation_only_query,
                    )
                    from services.mission.mission_authority_kernel import (
                        build_mission_authority_kernel,
                    )

                    if is_interpretation_only_query(query) or bool(
                        packet.inferred_constraints.get("defer_global_shortlist")
                    ):
                        kernel = build_mission_authority_kernel(
                            mission,
                            packet,
                            recommendations=[],
                            query=query or "",
                            data_used=data_used,
                            route_certainty_degraded=route_degraded,
                            projection_trace=projection_trace,
                        )
                        return format_mission_interpretation(
                            mission,
                            packet,
                            kernel,
                            query=query or "",
                            data_used=data_used,
                        )
                except Exception:
                    pass
            return format_understanding_first_advisory(
                mission,
                packet,
                recommendations=recommendations,
                query=query or "",
                data_used=data_used,
                route_certainty_degraded=route_degraded,
            )
        from services.broker.graceful_degradation import degraded_empty_shortlist_guidance

        guidance = degraded_empty_shortlist_guidance(mission, None, query)
        if guidance.strip():
            return guidance
        route = ", ".join(mission.routes or []) or "stated route"
        pax = mission.passenger_count if mission.passenger_count is not None else "—"
        return (
            "INSUFFICIENT_DATA: No verified aircraft meet stated mission constraints.\n\n"
            f"Mission Fit:\n* Route: {route}\n* Pax: {pax}\n"
            "* Constraints: hard feasibility eliminated all catalog candidates.\n\n"
            "Aircraft Options:\n(none)\n\n"
            "Verdict:\n* NOT A FIT: No verified shortlist for this mission as stated."
        )

    if packet is not None:
        from services.mission.mission_synthesis_contract import (
            attach_synthesis_contract_metadata,
            compose_ranked_advisory_response,
        )

        openers = [
            "Ranked options below follow the operational structure above — not a generic shortlist.",
            "Aircraft options are subordinate to the band and fleet doctrine already stated.",
        ]
        if seed:
            digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % len(openers)
        else:
            idx = 0
        from services.mission.mission_authority_kernel import (
            build_mission_authority_kernel,
            filter_recommendations_by_kernel,
            project_kernel_advisory,
        )

        kernel = build_mission_authority_kernel(
            mission,
            packet,
            recommendations=viable,
            query=query or "",
            data_used=data_used,
            route_certainty_degraded=route_degraded,
            projection_trace=projection_trace,
        )
        filtered = filter_recommendations_by_kernel(viable, kernel)
        body = project_kernel_advisory(kernel, filtered, opener=openers[idx])
        attach_synthesis_contract_metadata(
            data_used,
            prefix=body[:2000],
            projection_trace=projection_trace,
        )
        return body

    from services.consultant.response_architecture import format_recommendation_architecture

    body = format_recommendation_architecture(
        mission,
        viable,
        route_assessments=route_assessments,
    )
    return _append_mission_portfolio_sections(
        body,
        mission,
        query=query or "",
        data_used=data_used,
    )


def build_broker_llm_context_block(
    mission: MissionState,
    recommendations: Sequence[AircraftRecommendation],
    route_assessments: Optional[Sequence[RouteFeasibilityAssessment]] = None,
    *,
    eliminated_models: Optional[Sequence[str]] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Mandatory pre-LLM block — broker context without feasibility invention rights."""
    from services.elimination.elimination_invariant import (
        collect_eliminated_models,
        enforce_elimination_invariant,
    )

    eliminated = collect_eliminated_models(
        data_used=data_used,
        explicit_eliminated=eliminated_models,
    )
    recommendations = enforce_elimination_invariant(
        recommendations,
        eliminated,
        context="llm_context",
    )
    ctx = build_broker_advisory_context(mission, recommendations, route_assessments)
    try:
        from services.telemetry.reasoning_packet_enforcement import (
            extract_reasoning_packet,
            format_immutable_reasoning_packet_block,
            packet_verdict_sources,
        )

        packet = extract_reasoning_packet(data_used)
        if packet:
            verdicts = packet_verdict_sources(packet)
            for i, ac in enumerate(ctx.feasible_aircraft):
                key = (ac.model or "").strip().lower()
                if key in verdicts:
                    ctx.feasible_aircraft[i].fit_verdict = verdicts[key]
            block = ctx.to_llm_block() + "\n\n" + format_immutable_reasoning_packet_block(packet)
            return block
    except Exception:
        pass
    return ctx.to_llm_block()
