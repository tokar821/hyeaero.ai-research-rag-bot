"""
Standardized broker recommendation response renderer.

Structure:
1. Mission interpretation (1–3 lines)
2. Constraint summary (+ dispatch conflict when applicable)
3. Ranked shortlist (max 3–5) with multi-factor scores
4. Why each fits (constraint-based)
5. Final verdict (GOOD FIT / CONDITIONAL FIT / NOT A FIT)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from services.consultant.comparative_analysis_renderer import (
    format_comparative_analysis_table,
    format_three_way_model_comparison,
    is_comparative_economics_query,
    is_named_model_comparison_query,
)
from services.consultant.dispatch_conflict_renderer import format_dispatch_conflict_block
from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.consultant.route_feasibility import RouteFeasibilityAssessment
from services.broker.broker_language import sanitize_broker_language


def _mission_interpretation_lines(
    mission: MissionState,
    packet: Any,
    *,
    query: str = "",
) -> List[str]:
    lines: List[str] = []
    if packet is not None and getattr(packet, "operational_synthesis", None):
        syn = str(packet.operational_synthesis).strip()
        if syn:
            lines.append(syn[:280])
    elif mission.routes:
        route = mission.routes[0]
        pax = mission.passenger_count
        if pax:
            lines.append(f"Primary stage: {route} with {pax} passengers — economics and runway drive the band.")
        else:
            lines.append(f"Primary stage: {route} — constraint-based shortlist below.")
    else:
        lines.append("Mission intake is partial — shortlist reflects stated constraints only.")
    return lines[:3]


def _constraint_summary(
    mission: MissionState,
    packet: Any,
    *,
    data_used: Optional[Dict[str, Any]] = None,
) -> List[str]:
    lines: List[str] = ["Constraint Summary"]
    if mission.passenger_count:
        lines.append(f"- passengers: {mission.passenger_count}")
    if mission.routes:
        lines.append(f"- primary route: {mission.routes[0]}")
        if len(mission.routes) > 1:
            lines.append(f"- secondary legs: {len(mission.routes) - 1} additional corridor(s)")
    if (mission.operating_cost_priority or "").lower() == "high":
        lines.append("- operating cost sensitivity: high")
    if mission.mountain_airport_requirement:
        lines.append("- runway: short-field / mountain access required")
    if isinstance(data_used, dict):
        hw = data_used.get("hierarchy_weighting") or {}
        if isinstance(hw, dict) and hw.get("dominant_utilization"):
            lines.append(f"- dominant utilization: {hw['dominant_utilization']}")
        recovery = data_used.get("tier_downgrade_recovery") or {}
        if isinstance(recovery, dict) and recovery.get("tier"):
            lines.append(
                f"- shortlist recovery: tier-downgraded to {recovery['tier']} after filter pass"
            )
    return lines


def _verdict_label(rec: AircraftRecommendation) -> str:
    """Display-only mapping — verdict must come from HACK v2 (fit_verdict)."""
    v = (rec.fit_verdict or "").strip().upper()
    if not v:
        return "CONDITIONAL FIT"
    if "NOT" in v or rec.avoid:
        return "NOT A FIT"
    if "GOOD" in v:
        return "GOOD FIT"
    if "CONDITIONAL" in v:
        return "CONDITIONAL FIT"
    return v


def _why_fits(rec: AircraftRecommendation) -> str:
    if rec.explanation:
        for src in (rec.explanation.why_it_fits, rec.explanation.strengths):
            for item in src or []:
                t = (item or "").strip()
                if t and len(t) > 10:
                    return t.rstrip(".") + "."
    for s in rec.scores or []:
        if s.note and len(s.note) > 12:
            return s.note.rstrip(".") + "."
    return "Constraint alignment on stage length, runway, and operating economics."


def _multi_factor_line(rec: AircraftRecommendation) -> str:
    parts: List[str] = []
    if getattr(rec, "suitability_score", 0) > 0:
        parts.append(f"suitability={rec.suitability_score:.2f}")
    if getattr(rec, "economics_score", 0) > 0:
        parts.append(f"economics={rec.economics_score:.2f}")
    if getattr(rec, "operational_flexibility_score", 0) > 0:
        parts.append(f"flex={rec.operational_flexibility_score:.2f}")
    if getattr(rec, "mission_conflict_penalty", 0) > 0:
        parts.append(f"conflict_penalty={rec.mission_conflict_penalty:.2f}")
    return " | ".join(parts) if parts else ""


def format_broker_recommendation_response(
    mission: MissionState,
    recommendations: Sequence[AircraftRecommendation],
    *,
    query: str = "",
    packet: Any = None,
    route_assessments: Optional[Sequence[RouteFeasibilityAssessment]] = None,
    data_used: Optional[Dict[str, Any]] = None,
    max_results: int = 5,
) -> str:
    """Render standardized broker-grade recommendation output."""
    del route_assessments

    from services.recommendation.hack_v3_renderer_lock import (
        render_hack_v3_locked_response,
        should_use_hack_v3_renderer,
    )

    if should_use_hack_v3_renderer(data_used):
        return render_hack_v3_locked_response(
            data_used,
            recommendations=recommendations,
        )

    viable = [r for r in recommendations if not r.avoid][:max_results]
    sections: List[str] = []

    sections.append("Mission Interpretation")
    for line in _mission_interpretation_lines(mission, packet, query=query):
        sections.append(f"- {line}")

    sections.append("")
    sections.extend(_constraint_summary(mission, packet, data_used=data_used))

    conflict = format_dispatch_conflict_block(packet, data_used=data_used, query=query)
    if conflict:
        sections.append("")
        sections.append(conflict)

    if is_comparative_economics_query(query):
        sections.append("")
        sections.append(format_comparative_analysis_table(mission, query=query))

    if is_named_model_comparison_query(query):
        from services.consultant.recommendation_engine import detect_models_from_text

        named = detect_models_from_text(query)
        hours = None
        m = re.search(r"(\d{2,3})\s*[-–]\s*(\d{2,3})\s*hours", (query or "").lower())
        if m:
            hours = int(m.group(2))
        sections.append("")
        sections.append(format_three_way_model_comparison(named, annual_hours=hours))

    sections.append("")
    sections.append("Ranked Aircraft Shortlist")
    if not viable:
        sections.append("- No model cleared all filters — tier recovery produced class-band guidance only.")
        sections.append("- Re-run with dominant route hours split to tighten the band.")
    else:
        for rec in viable:
            mf = _multi_factor_line(rec)
            header = f"* {rec.model} — {_verdict_label(rec)}"
            if mf:
                header += f" ({mf})"
            sections.append(header)
            sections.append(f"  Why: {_why_fits(rec)}")

    sections.append("")
    sections.append("Final Verdict")
    if viable:
        lead = viable[0]
        sections.append(
            f"- {_verdict_label(lead)}: {lead.model} leads on composite mission economics and operational fit."
        )
        if len(viable) > 1:
            sections.append(
                f"- Alternates: {', '.join(r.model for r in viable[1:3])} — conditional paths if constraints shift."
            )
    else:
        sections.append("- CONDITIONAL FIT: structure-first — specify dominant annual hour split before model lock.")

    return sanitize_broker_language("\n".join(sections).strip())


__all__ = ["format_broker_recommendation_response"]
