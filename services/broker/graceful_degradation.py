"""
Broker graceful degradation — bounded uncertainty instead of hard refusal.

Never make false operational claims; always provide the closest honest broker guidance.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation

# Phrases that collapse UX — rewritten or stripped before user sees them.
_COLLAPSE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"I\s+don'?t\s+have\s+enough\s+verified\s+field[- ]performance\s+data[^.]*\.?",
            re.I,
        ),
        "",
    ),
    (
        re.compile(r"I\s+don'?t\s+have\s+reliable\s+data\s+for\s+this\.?", re.I),
        "",
    ),
    (
        re.compile(
            r"I\s+couldn'?t\s+complete\s+the\s+full\s+advisory\s+pipeline[^.]*\.?",
            re.I,
        ),
        "",
    ),
    (
        re.compile(
            r"Share\s+the\s+primary\s+city\s+pair\s+and\s+passenger\s+count\s+and\s+I'?ll\s+size[^.]*\.?",
            re.I,
        ),
        "",
    ),
)

_REFUSAL_TO_DEGRADED: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"I\s+would\s+not\s+advise\s+this\s+with\s+confidence\s+based\s+on\s+verified\s+performance\s+data\.?",
            re.I,
        ),
        "I would treat this as a conditional mission — here is the closest credible class guidance:",
    ),
    (
        re.compile(
            r"I\s+can'?t\s+position\s+this\s+aircraft\s+as\s+operationally\s+credible[^.]*\.?",
            re.I,
        ),
        "I would not sell this as a clean operational fit without assumptions stated — closest read:",
    ),
    (
        re.compile(
            r"I\s+would\s+not\s+position\s+this\s+aircraft\s+as\s+a\s+credible\s+nonstop\s+solution\.?",
            re.I,
        ),
        "I would not sell this as a reliable year-round nonstop without payload and seasonal caveats. Realistically:",
    ),
    (
        re.compile(
            r"This\s+mission\s+exceeds\s+realistic\s+payload[- ]range\s+margins\.?",
            re.I,
        ),
        "Payload-range is tight on this profile — expect tech-stop or cabin tradeoffs:",
    ),
)


def transform_refusal_prose(text: str) -> str:
    """Rewrite refusal-heavy phrasing into bounded broker guidance."""
    out = (text or "").strip()
    if not out:
        return out
    for pat, repl in _REFUSAL_TO_DEGRADED + _COLLAPSE_PATTERNS:
        out = pat.sub(repl, out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    return out.strip()


def broker_degraded_message(*, context: str = "general", **details: Any) -> str:
    """
    Operational honesty without dead-ends — use instead of ``broker_refusal_message`` for user text.
    """
    route = str(details.get("route") or "").strip()
    pax = details.get("passengers")
    models = details.get("models") or details.get("class_band") or []

    if context == "nonstop_not_credible":
        r = route or "this leg"
        return (
            f"I would not sell {r} as a reliable year-round nonstop without stating payload, "
            "season, and reserve assumptions. The credible band is usually ultra-long or "
            "large-cabin with winter westbound margin — not a super-mid marketed as guaranteed nonstop."
        )
    if context == "payload_range":
        return (
            "Payload-range is the constraint here — not aircraft preference. "
            "Expect either a tech stop, fewer passengers with bags, or stepping up one cabin class."
        )
    if context == "field_performance":
        r = route or "this strip"
        return (
            f"Exact hot/high numbers depend on temperature and weight, but operationally {r} "
            "sits in short-field / high-performance territory — not a heavy cabin jet day-in, day-out."
        )
    if context == "geodesic_corridor_only":
        return (
            "I only have corridor-class distance here, not a verified catalog nonstop stage. "
            "I can size the aircraft class from that envelope, but I would not certify nonstop until "
            "the city pair resolves to catalog mileage."
        )
    if context == "aircraft_specs":
        model = str(details.get("model") or "this aircraft")
        return (
            f"I do not have verified brochure-grade numbers loaded for {model}, but I can still "
            "compare it qualitatively to peers in the same operational band if you name the mission."
        )
    if context == "conflicting_requirements":
        return (
            "You are stacking requirements that do not live on one airframe — I would split the "
            "mission by domain (range leg vs field-performance leg) rather than forcing one jet."
        )
    if context == "empty_shortlist":
        return degraded_empty_shortlist_guidance(
            details.get("mission"),
            details.get("pipeline"),
            str(details.get("query") or ""),
        )
    if context == "no_feasible":
        return degraded_no_feasible_guidance(
            details.get("mission"),
            details.get("pipeline"),
            details.get("data_used"),
        )

    if models:
        band = ", ".join(str(m) for m in models[:3])
        opener = f"For {route}, " if route else ""
        return (
            f"{opener}I would start in {band} territory"
            + (f" for {pax} passengers" if pax else "")
            + ". Exact dispatch margins still depend on payload and season, but that is the right class conversation."
        )
    return (
        "I would treat this as a conditional advisory — not a guaranteed spec sheet. "
        "Here is the closest operationally honest guidance I can support:"
    )


def _route_phrase(mission: Optional[MissionState]) -> str:
    if mission is None:
        return ""
    routes = list(mission.routes or [])
    if routes:
        return routes[0]
    return ""


def degraded_empty_shortlist_guidance(
    mission: Any,
    pipeline: Any,
    query: str = "",
) -> str:
    """Rank produced nothing — still guide by class band and next data needed."""
    ms = mission if isinstance(mission, MissionState) else None
    route = _route_phrase(ms)
    pax = getattr(ms, "passenger_count", None) if ms else None
    lines = [
        "Nothing in the catalog passed every hard gate as stated — that usually means range with reserves, "
        "runway, or passenger load is fighting the request, not that aviation is unknowable.",
    ]
    if route:
        lines.append(f"For {route}" + (f" with {pax} passengers" if pax else "") + ":")
    lines.append(
        "• If you can relax nonstop, the feasible band often widens by one cabin class."
    )
    lines.append(
        "• If runway or hot/high is the limiter, prioritize field-performance super-mid / light-jet bands "
        "over range-first transcon setups."
    )
    lines.append(
        "• Tell me which constraint can move (payload, date, nonstop requirement) and I will re-run the shortlist."
    )
    if pipeline is not None and getattr(pipeline, "feasible_models", None):
        feas = list(pipeline.feasible_models or [])[:4]
        if feas:
            lines.append(
                "Closest survivors before late elimination: " + ", ".join(feas) + "."
            )
    return "\n".join(lines)


def degraded_no_feasible_guidance(
    mission: Any,
    pipeline: Any,
    data_used: Optional[Dict[str, Any]],
) -> str:
    """Global feasibility empty — multi-domain or class-band guidance."""
    du = data_used or {}
    fp = du.get("fleet_composition_plan") if isinstance(du, dict) else None
    if isinstance(fp, dict) and fp.get("multi_aircraft_required"):
        lines = [
            "No single aircraft clears every leg of this mission as stated — that is a structural outcome, "
            "not missing data.",
            fp.get("doctrine") or "",
        ]
        for a in fp.get("assignments") or []:
            if isinstance(a, dict) and a.get("primary_model"):
                lines.append(
                    f"• {a.get('segment_label', 'Domain')}: {a.get('primary_model')} "
                    f"({a.get('fit_verdict', 'domain fit')})"
                )
        return "\n".join(l for l in lines if l).strip()

    return degraded_empty_shortlist_guidance(mission, pipeline, "")


def degraded_comparison_note(model: str) -> str:
    """Side-by-side row when specs are not verified — still useful."""
    return (
        f"{model}: I do not have verified brochure numbers loaded in-band, but you can still "
        "compare mission fit qualitatively — treat range and payload figures as directional only."
    )


def degraded_low_confidence_prefix() -> str:
    return (
        "Some details below are directional rather than catalog-verified — I am narrowing scope "
        "instead of overstating certainty:"
    )


def safe_broker_fallback_response(
    query: str = "",
    *,
    mission: Optional[MissionState] = None,
    pipeline: Any = None,
    recommendations: Optional[Sequence[AircraftRecommendation]] = None,
    route_assessments: Any = None,
    data_used: Optional[Dict[str, Any]] = None,
    failure_stage: str = "",
) -> str:
    """
    Last-resort broker answer — NEVER empty, NEVER generic retry-only text.
    """
    del route_assessments, failure_stage
    recs = [r for r in (recommendations or []) if not getattr(r, "avoid", False)]
    if recs:
        try:
            from services.consultant.broker_advisory_layer import format_broker_advisory_response

            body = format_broker_advisory_response(
                mission or MissionState(),
                recs[:3],
                data_used=data_used,
            )
            if (body or "").strip():
                return apply_graceful_degradation_to_answer(body, confidence=0.55)
        except Exception:
            pass

    if pipeline is not None:
        mv = getattr(pipeline, "mission_validation", None) or {}
        if isinstance(mv, dict) and mv.get("multi_domain_operational_decomposition"):
            text = degraded_no_feasible_guidance(mission, pipeline, data_used)
            return apply_graceful_degradation_to_answer(text, confidence=0.5)

    q = (query or "").strip()
    if q and re.search(r"(?is)\b(?:aggressive|fair|cheap|listed\s+at|asking)\b", q) and re.search(
        r"\$\s*\d", q
    ):
        return apply_graceful_degradation_to_answer(
            "For listing-price questions, compare the ask to recent comps for that year-model, "
            "program status, and time-on-airframe. If you share the exact model and year, "
            "I can frame whether the price looks aggressive, fair, or cheap relative to market.",
            confidence=0.6,
        )

    if mission is not None:
        try:
            from services.broker_execution.mission_broker_answer import is_mission_shaped_query

            if not is_mission_shaped_query(q):
                pass
            else:
                from services.consultant.broker_advisory_layer import (
                    _category_territory_label,
                    _opening_line,
                )

                territory = _category_territory_label(recs) if recs else "the right cabin class"
                opener = _opening_line(mission, recs) if recs else f"For this trip, you are in {territory}."
                return apply_graceful_degradation_to_answer(
                    f"{opener}\n\n"
                    "I hit a formatting fault on the last pass, but the mission sizing still points to "
                    f"{territory}. Name the city pair if you want a refreshed shortlist with tradeoffs.",
                    confidence=0.52,
                )
        except Exception:
            pass

    return broker_degraded_message(
        context="general",
        route=_route_phrase(mission),
        passengers=getattr(mission, "passenger_count", None) if mission else None,
        query=query,
    )


def apply_graceful_degradation_to_answer(
    answer: str,
    *,
    confidence: float = 1.0,
    prefix_if_low_confidence: bool = True,
) -> str:
    """Normalize prose and optionally prefix low-confidence guidance."""
    text = transform_refusal_prose(answer)
    if not text:
        return text
    if prefix_if_low_confidence and confidence < 0.55:
        prefix = degraded_low_confidence_prefix()
        if prefix.lower() not in text.lower():
            text = f"{prefix}\n\n{text}"
    return text.strip()


def ensure_non_empty_answer(
    answer: Optional[str],
    *,
    query: str = "",
    mission: Optional[MissionState] = None,
    pipeline: Any = None,
    recommendations: Optional[Sequence[AircraftRecommendation]] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Guarantee a user-visible broker response."""
    if (answer or "").strip():
        return transform_refusal_prose(answer.strip())
    return safe_broker_fallback_response(
        query,
        mission=mission,
        pipeline=pipeline,
        recommendations=recommendations,
        data_used=data_used,
    )
