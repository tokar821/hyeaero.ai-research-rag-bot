"""
Fixed response architecture — recommendations and comparisons.

Recommendations:
  Mission Fit → Route, Pax, Priorities
  Aircraft Options → Aircraft, Why it fits, Key compromise
  Verdict → PRIMARY RECOMMENDATION | VIABLE WITH COMPROMISES | MISSION-RISKY | NOT OPERATIONALLY CREDIBLE

Comparisons (only):
  range, cabin, operating cost, runway capability, liquidity
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from services.broker.broker_language import sanitize_broker_language
from services.broker.broker_verdicts import BrokerVerdict
from services.consultant.broker_advisory_layer import broker_verdict_label, sanitize_broker_prose
from services.consultant.mission_state import MissionState, normalize_routes
from services.consultant.recommendation_engine import AircraftRecommendation, _AIRCRAFT_PROFILES
from services.consultant.route_feasibility import RouteFeasibilityAssessment

_VERDICT_PRIMARY = BrokerVerdict.PRIMARY_RECOMMENDATION.value
_VERDICT_VIABLE = BrokerVerdict.VIABLE_WITH_COMPROMISES.value
_VERDICT_RISKY = BrokerVerdict.MISSION_RISKY.value
_VERDICT_NOT = BrokerVerdict.NOT_OPERATIONALLY_CREDIBLE.value

_MARKETING_RE = re.compile(
    r"\b(?:best[- ]in[- ]class|unparalleled|world[- ]class|game[- ]changing|"
    r"flagship|industry[- ]leading|premium experience|luxury redefined)\b",
    re.I,
)
_GENERIC_PROSE_RE = re.compile(
    r"\b(?:worth considering|if priorities shift|balanced capability|"
    r"operationally balanced|clearest fit|great option)\b",
    re.I,
)


def _clean_line(text: str, *, max_len: int = 220) -> str:
    s = (text or "").strip()
    s = _MARKETING_RE.sub("", s)
    s = _GENERIC_PROSE_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;.")
    if len(s) > max_len:
        s = s[: max_len - 3].rstrip() + "..."
    return s


def _route_line(mission: MissionState) -> str:
    normalized = normalize_routes(mission.routes)
    if not normalized:
        return "Not stated"
    if len(normalized) == 1:
        return normalized[0]
    return "; ".join(normalized[:2])


def _priorities_line(mission: MissionState) -> str:
    parts: List[str] = []
    if mission.nonstop_requirement:
        parts.append("nonstop")
    if (mission.operating_cost_priority or "").lower() == "high":
        parts.append("operating cost")
    if (mission.cabin_priority or "").lower() == "high":
        parts.append("cabin")
    if (mission.baggage_priority or "").lower() == "high":
        parts.append("baggage")
    if mission.mountain_airport_requirement:
        parts.append("runway / hot-high")
    if mission.westbound:
        parts.append("westbound margin")
    if (mission.seasonal_constraints or "").lower().find("winter") >= 0:
        parts.append("winter ops")
    if mission.runway_constraints:
        parts.append(f"runway ({mission.runway_constraints})")
    if mission.budget_usd:
        parts.append(f"budget ~${int(mission.budget_usd):,}")
    if (mission.acquisition_strategy or "").strip():
        parts.append(mission.acquisition_strategy.replace("_", " "))
    return ", ".join(parts) if parts else "Standard mission — no special constraints stated"


def _why_it_fits(rec: AircraftRecommendation) -> str:
    if rec.explanation:
        for src in (rec.explanation.why_it_fits, rec.explanation.strengths):
            for item in src or []:
                line = _clean_line(item)
                if line and "score" not in line.lower():
                    return line
        if rec.explanation.summary:
            s = _clean_line(rec.explanation.summary)
            if s and "NOT OPERATIONALLY" not in s.upper():
                return s
    return f"Matches route length and passenger load for this leg."


def _key_compromise(rec: AircraftRecommendation) -> str:
    if rec.explanation:
        for src in (
            rec.explanation.operational_compromises,
            rec.explanation.operational_caveats,
            rec.explanation.penalties,
        ):
            for item in src or []:
                line = _clean_line(item)
                if line:
                    return line
    return "None material on this leg."


def _format_mission_fit_section(mission: MissionState) -> str:
    pax = mission.passenger_count
    pax_str = str(pax) if pax is not None else "Not stated"
    lines = [
        "Mission Fit:",
        "",
        f"* Route: {_route_line(mission)}",
        f"* Pax: {pax_str}",
        f"* Priorities: {_priorities_line(mission)}",
    ]
    return "\n".join(lines)


def _format_aircraft_options_section(
    recommendations: Sequence[AircraftRecommendation],
) -> str:
    lines = ["Aircraft Options:", ""]
    for rec in recommendations:
        why = _why_it_fits(rec)
        compromise = _key_compromise(rec)
        lines.append(
            f"* {rec.model} — Why it fits: {why} Key compromise: {compromise}"
        )
    return "\n".join(lines)


def _format_verdict_section(
    recommendations: Sequence[AircraftRecommendation],
) -> str:
    primary: List[str] = []
    viable: List[str] = []
    risky: List[str] = []
    not_credible: List[str] = []

    for rec in recommendations:
        bucket = broker_verdict_label(rec)
        if bucket == _VERDICT_PRIMARY:
            primary.append(rec.model)
        elif bucket == _VERDICT_VIABLE:
            viable.append(rec.model)
        elif bucket == _VERDICT_RISKY:
            risky.append(rec.model)
        elif bucket == _VERDICT_NOT:
            not_credible.append(rec.model)
        else:
            viable.append(rec.model)

    lines = ["Verdict:", ""]
    if primary:
        lines.append(f"* {_VERDICT_PRIMARY}: {', '.join(primary)}")
    if viable:
        lines.append(f"* {_VERDICT_VIABLE}: {', '.join(viable)}")
    if risky:
        lines.append(f"* {_VERDICT_RISKY}: {', '.join(risky)}")
    if not_credible:
        lines.append(f"* {_VERDICT_NOT}: {', '.join(not_credible)}")
    if len(lines) == 2:
        lines.append(f"* {_VERDICT_VIABLE}: {', '.join(r.model for r in recommendations)}")
    return "\n".join(lines)


def format_recommendation_options_and_verdict(
    recommendations: Sequence[AircraftRecommendation],
    *,
    route_assessments: Optional[Sequence[RouteFeasibilityAssessment]] = None,
) -> str:
    """Aircraft options + verdict only — synthesis block is composed upstream."""
    del route_assessments
    viable = [r for r in recommendations if not r.avoid]
    if not viable:
        return ""
    blocks = [
        _format_aircraft_options_section(viable),
        "",
        _format_verdict_section(viable),
    ]
    return sanitize_broker_language(sanitize_broker_prose("\n".join(blocks).strip()))


def format_recommendation_architecture(
    mission: MissionState,
    recommendations: Sequence[AircraftRecommendation],
    *,
    route_assessments: Optional[Sequence[RouteFeasibilityAssessment]] = None,
) -> str:
    """
    Fixed three-block recommendation layout — no territory opener or generic prose.
    """
    del route_assessments
    viable = [r for r in recommendations if not r.avoid]
    if not viable:
        return ""

    blocks = [
        _format_mission_fit_section(mission),
        "",
        _format_aircraft_options_section(viable),
        "",
        _format_verdict_section(viable),
    ]
    return sanitize_broker_language(sanitize_broker_prose("\n".join(blocks).strip()))


def _qualitative_cost(operating_index: float) -> str:
    if operating_index <= 0.45:
        return "Lower"
    if operating_index <= 0.65:
        return "Mid"
    return "Higher"


def _qualitative_cabin(cabin_score: float) -> str:
    if cabin_score >= 0.85:
        return "Large-cabin comfort"
    if cabin_score >= 0.72:
        return "Stand-up / super-mid comfort"
    if cabin_score >= 0.55:
        return "Mid cabin"
    return "Compact cabin"


def _qualitative_runway(runway_ft: float, short_field: float) -> str:
    if short_field >= 0.65 and runway_ft <= 4200:
        return f"Shorter fields (~{int(runway_ft)} ft roll)"
    if runway_ft >= 5000:
        return f"Longer runway footprint (~{int(runway_ft)} ft)"
    return f"Typical runway (~{int(runway_ft)} ft)"


def _qualitative_liquidity(resale_score: float) -> str:
    if resale_score >= 0.82:
        return "Strong"
    if resale_score >= 0.72:
        return "Solid"
    return "Thinner"


def comparison_dimension_row(model: str) -> Dict[str, str]:
    """Five allowed comparison dimensions only — verified catalog facts."""
    from services.aircraft_truth import (
        UNVERIFIED_AIRCRAFT_MESSAGE,
        format_verified_comparison_snippets,
        validate_aircraft_truth,
    )

    truth = validate_aircraft_truth(model)
    if not truth.verified or not truth.facts:
        return {dim: UNVERIFIED_AIRCRAFT_MESSAGE for dim in (
            "range",
            "cabin",
            "operating cost",
            "runway capability",
            "liquidity",
        )}
    return format_verified_comparison_snippets(truth.facts)


def format_comparison_architecture(models: Sequence[str]) -> str:
    """
    Side-by-side comparison — only range, cabin, operating cost, runway, liquidity.
    """
    from services.aircraft_truth import filter_truth_verified_models

    uniq = filter_truth_verified_models(
        [m for m in dict.fromkeys(models) if m and m in _AIRCRAFT_PROFILES]
    )[:4]
    if len(uniq) < 2:
        return ""

    title = " vs ".join(uniq[:3])
    if len(uniq) > 3:
        title += " (+others)"

    dims = ("range", "cabin", "operating cost", "runway capability", "liquidity")
    rows_by_model = {m: comparison_dimension_row(m) for m in uniq}

    lines = [f"Comparison: {title}", ""]
    for dim in dims:
        parts = [f"* {dim.capitalize() if dim != 'operating cost' else 'Operating cost'}:"]
        for model in uniq:
            parts.append(f"  {model}: {rows_by_model[model][dim]}")
        lines.append("\n".join(parts))
        lines.append("")

    return sanitize_broker_prose("\n".join(lines).strip())


__all__ = [
    "comparison_dimension_row",
    "format_comparison_architecture",
    "format_recommendation_architecture",
]
