"""
Private aviation advisor response formatting — direct, operational, conversational.

Output structure:
  1. Direct recommendation (1–2 lines)
  2. Short list of aircraft (bullets)
  3. Practical closing note (plain prose)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from services.consultant.comparison_engine import StructuredComparison
from services.consultant.mission_state import MissionState, format_routes_for_display
from services.consultant.recommendation_engine import AircraftRecommendation
from services.consultant.route_feasibility import RouteFeasibilityAssessment
from services.consultant.response_variation import (
    AdvisorPhraseBundle,
    VariationContext,
    compose_varied_response,
    select_response_style,
    _bullet_phrase,
)
from services.recommendation.clarification_decision import (
    build_clarification_questions,
    route_truly_missing,
)
from services.recommendation.fit_policy import (
    mission_clarification_needs,
    mission_maps_to_category,
    mission_well_defined,
)


_LAST_RESPONSE_STYLE: str = ""

# Internal / dev headers and metadata leaked into drafts
_STRIP_HEADER_RE = re.compile(
    r"^\s*(?:Mission Summary|Best Fit Aircraft|Why They Fit|Operational Tradeoffs|"
    r"Why Alternatives Ranked Lower|Alternatives(?:\s+scored\s+lower)?|"
    r"Bottom-Line Recommendation|Side-by-Side Comparison|Top Aircraft Options|"
    r"Conditional options|The tradeoffs to keep in view|Mission type|Route\(s\)|Passengers|"
    r"Nonstop required|Westbound|Mountain or hot/high|Ownership posture|Budget anchor)\s*:?\s*$",
    re.I | re.M,
)

_SCORE_METADATA_RE = re.compile(
    r"\s*[—–-]\s*mission[- ]?fit score\s*[\d.]+\s*"
    r"|\s*\(confidence\s*[\d.]+%?\)\s*"
    r"|\s*\(score\s*[\d.]+\s*\)"
    r"|\bmission score\s*[\d.]+\s*"
    r"|\bconfidence\s*[\d.]+%?\b"
    r"|\btotal_score\s*[=:]\s*[\d.]+\b"
    r"|\b\d\.\d{2,4}\s*(?:fit\s+)?score\b"
    r"|\b(?:rank|#)\s*[#]?\s*\d+\b"
    r"|\bFit:\s*(?:High|Medium|Low|Strong|Good|Partial)\b",
    re.I,
)

_ROBOTIC_PHRASE_RE = re.compile(
    r"\b(?:conditional options|conditional paths|mission firms up|"
    r"lock a single winner|until then|clearest fit|clearest starting point|"
    r"forced ranking|not a forced|before we lock|pin the city pair|"
    r"may not stay the right|that answer changes if)\b",
    re.I,
)

_HEDGING_PHRASE_RE = re.compile(
    r"\b(?:probably the smartest|might want to|could work if)\b",
    re.I,
)

_TECH_ROUTE_RE = re.compile(
    r"~?\d+\s*nm|brochure-capable only|practical_restricted|"
    r"reliably_nonstop|not feasible nonstop|NBAA-style margin|"
    r"westbound margin applied|headwind[s]? applied|versus super-midsize",
    re.I,
)

_INTERNAL_TAXONOMY_RE = re.compile(
    r"\b(?:super[- ]?mid\b(?!size)|mission[- ]?fit|point_to_point|"
    r"acquisition_advisory|eliminated before scoring)\b",
    re.I,
)


def _clean_bullet(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"^\s*[-•]\s*", "", s)
    s = _SCORE_METADATA_RE.sub("", s)
    s = _INTERNAL_TAXONOMY_RE.sub("", s)
    s = _ROBOTIC_PHRASE_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _natural_strength(rec: AircraftRecommendation) -> str:
    if not rec.explanation:
        return ""
    for src in (rec.explanation.why_it_fits, rec.explanation.strengths):
        for item in src or []:
            line = _clean_bullet(item)
            if line and "score" not in line.lower() and "mission-fit" not in line.lower():
                return line
    return ""


def _tradeoff_snippets(recs: List[AircraftRecommendation], limit: int = 2) -> List[str]:
    lines: List[str] = []
    for rec in recs[:2]:
        if not rec.explanation:
            continue
        for src in (
            rec.explanation.operational_compromises,
            rec.explanation.operational_caveats,
            rec.explanation.penalties,
        ):
            for item in src or []:
                line = _clean_bullet(item)
                if not line:
                    continue
                low = line.lower()
                if _TECH_ROUTE_RE.search(line):
                    continue
                if any(
                    skip in low
                    for skip in (
                        "eliminated",
                        "ranked lower",
                        "lower range",
                        "lower overbuying",
                        "vs ",
                        "operating cost runs high versus",
                    )
                ):
                    continue
                if line not in lines:
                    lines.append(line.rstrip("."))
                if len(lines) >= limit:
                    return lines
    return lines


def _route_phrase(mission: MissionState) -> str:
    routes = format_routes_for_display(mission.routes)
    if routes:
        return routes.replace(" -> ", " to ")
    return ""


def _operational_context(mission: MissionState) -> str:
    """Plain-language trip shape for advisor copy."""
    routes = _route_phrase(mission).lower()
    if mission.nonstop_requirement and mission.westbound and any(
        x in routes for x in ("tokyo", "seoul", "hong kong")
    ):
        return "regular westbound transpacific runs with a true nonstop requirement"
    if mission.nonstop_requirement and any(
        x in routes for x in ("london", "paris", "geneva", "europe")
    ):
        return "transatlantic nonstop work"
    if "caribbean" in routes or ("miami" in routes and "caribbean" in routes):
        return "South Florida and Caribbean hops"
    if mission.mountain_airport_requirement:
        return "short runways and hot-and-high fields"
    if mission.operating_cost_priority == "high":
        return "high-utilization flying where hourly cost and maintenance downtime show up fast"
    if mission.cabin_priority == "high":
        return "longer sectors where cabin altitude, lav/galley practicality, and baggage volume matter"
    if routes:
        return "the stage lengths you're flying"
    return "your typical mission pattern"


def _lead_reason(rec: AircraftRecommendation, index: int) -> str:
    raw = _natural_strength(rec)
    low = (raw or "").lower()
    if index == 0:
        if "runway" in low or "field" in low:
            return "keeps runway access and usable payload intact without turning every trip into a weight-and-balance exercise"
        if "operating" in low or "cost" in low or "economics" in low:
            return "keeps hourly cost and ownership friction sensible when you actually fly the airplane, not just shop it"
        if "range" in low or "nonstop" in low:
            return "covers the leg with real-world fuel/reserve margin and a cabin you can actually load with people and bags"
        if "cabin" in low:
            return "gives you a cabin that works in practice — not just a brochure length"
        return "fits the stage length and passenger load without stretching the operation"
    if index == 1:
        if "range" in low or "nonstop" in low:
            return "more range margin if the legs get longer"
        if "cabin" in low:
            return "more cabin if passenger comfort is the priority"
        if "runway" in low:
            return "better runway flexibility if airport access tightens"
        return "a solid backup if the mission changes"
    if "operating" in low or "cost" in low:
        return "lighter operating burn if hours stay high"
    if "runway" in low:
        return "useful if runway access becomes the deciding factor"
    return "worth keeping in view if your longest leg grows or the payload requirement tightens"


def _direct_answer(mission: MissionState, top: AircraftRecommendation) -> str:
    """Lead recommendation — confident and specific."""
    route = _route_phrase(mission)
    ctx = _operational_context(mission)
    reason = _lead_reason(top, 0)

    if route and mission.passenger_count:
        opener = (
            f"With {mission.passenger_count} passengers on {route}, "
            f"I'd start with the {top.model} — {reason}."
        )
    elif route:
        opener = f"On {route}, I'd start with the {top.model} — {reason}."
    elif mission.passenger_count:
        opener = (
            f"For {mission.passenger_count} passengers and {ctx}, "
            f"I'd start with the {top.model}."
        )
    else:
        opener = f"For {ctx}, I'd start with the {top.model}."

    if mission.budget_usd and mission.budget_usd >= 5_000_000:
        millions = mission.budget_usd / 1_000_000
        opener += f" That keeps you in a realistic envelope near ${millions:.0f}M."
    return opener


def _partial_mission_opener(mission: MissionState) -> str:
    route = _route_phrase(mission)
    pax = mission.passenger_count
    if not route and pax:
        return (
            f"With {pax} passengers in mind, I can give you a practical shortlist — "
            "tell me the primary city pair when you have it and I'll tighten this to one lead aircraft."
        )
    if not route:
        return (
            "Give me the primary origin and destination and I'll pin this to one aircraft — "
            "here's how I'd think about it with what we know so far."
        )
    if pax and route:
        return (
            f"On {route} with {pax} passengers, here's how I'd line up the aircraft — "
            "priority depends on whether range, runway, or cabin drives the decision."
        )
    return (
        "Here's how I'd line up the aircraft — "
        "the right pick depends on whether range, runway, or cabin drives the decision."
    )


def _bullet_clause(
    rec: AircraftRecommendation,
    mission: MissionState,
    index: int,
    *,
    partial: bool,
) -> str:
    if partial:
        return _scenario_clause(rec, mission, index)
    ctx = _operational_context(mission)
    if index == 0:
        return ctx or "this mission profile"
    reason = _lead_reason(rec, index)
    out = reason[0].lower() + reason[1:] if reason else (ctx or "this mission profile")
    return out.strip() or "this mission profile"


def _scenario_clause(rec: AircraftRecommendation, mission: MissionState, index: int) -> str:
    raw = _natural_strength(rec)
    if raw and len(raw) < 90 and "score" not in raw.lower():
        cleaned = raw[0].lower() + raw[1:] if raw else ""
        if cleaned:
            return cleaned
    ctx = _operational_context(mission)
    if index == 0 and mission.operating_cost_priority == "high":
        return "runway flexibility and operating cost matter more than cabin luxury"
    if index == 0 and mission.mountain_airport_requirement:
        return "short-field and hot-and-high performance are non-negotiable"
    if index == 0 and mission.cabin_priority == "high":
        return "cabin quality matters more than minimizing operating cost"
    if index == 1:
        return "you want more cabin or range without jumping to an ultra-long-range jet"
    return f"your priorities align with {ctx}"


def _build_tradeoff_block(
    mission: MissionState,
    top: AircraftRecommendation,
    alts: List[AircraftRecommendation],
    route_assessments: List[RouteFeasibilityAssessment],
    comparison: Optional[StructuredComparison],
) -> str:
    """Tradeoff / closing prose for variation engine."""
    snippets: List[str] = []

    if route_assessments:
        a = route_assessments[0]
        caveats = [c for c in a.caveats if c and not _TECH_ROUTE_RE.search(c)]
        if a.classification == "not_feasible":
            snippets.append(
                f"{a.route_label} is a demanding leg on fuel and payload — "
                "plan conservatively even in a large-cabin jet."
            )
        elif caveats:
            snippets.append(f"On {a.route_label}, {caveats[0].rstrip('.').lower()}.")

    snippets.extend(_tradeoff_snippets([top] + alts[:1], limit=2))

    if comparison and comparison.operational_tradeoffs:
        line = _clean_bullet(comparison.operational_tradeoffs[0])
        if line and line not in snippets:
            snippets.append(line.rstrip("."))

    if not snippets and alts:
        alt = alts[0].model
        return (
            f"If runway flexibility and operating cost matter more than cabin luxury, "
            f"{alt} is the smarter pivot from {top.model}."
        )

    if not snippets:
        return (
            f"For this utilization pattern, {top.model} is the most balanced day-to-day "
            "operator in this group."
        )

    text = " ".join(snippets[:2])
    if not text.endswith("."):
        text += "."
    return text


def _build_route_ops_block(route_assessments: List[RouteFeasibilityAssessment]) -> str:
    if not route_assessments:
        return ""
    a = route_assessments[0]
    caveats = [c for c in a.caveats if c and not _TECH_ROUTE_RE.search(c)]
    if a.classification == "not_feasible":
        return (
            f"{a.route_label} is a demanding leg on fuel and payload — "
            "plan conservatively even in a large-cabin jet."
        )
    if caveats:
        return f"On {a.route_label}, {caveats[0].rstrip('.')}."
    return ""


def _build_phrase_bundle(
    mission: MissionState,
    viable: List[AircraftRecommendation],
    route_assessments: List[RouteFeasibilityAssessment],
    comparison: Optional[StructuredComparison],
    *,
    partial: bool,
    style: str,
    seed: str,
) -> AdvisorPhraseBundle:
    from services.consultant.recommendation_framing import (
        build_category_framing_line,
        build_model_transition_line,
        build_operational_reality_block,
        should_anchor_single_model,
        use_tiered_advisor_framing,
    )
    from services.recommendation.mission_ranker import classify_mission_category

    top = viable[0]
    alts = viable[1:3]
    mission_category = classify_mission_category(mission)
    op_ctx = _operational_context(mission)
    route = _route_phrase(mission)

    comparison_models: List[str] = []
    if comparison and comparison.models:
        comparison_models = list(comparison.models[:4])
    elif len(viable) >= 2:
        comparison_models = [r.model for r in viable[:2]]

    tiered = use_tiered_advisor_framing(
        mission, viable, route_assessments, mission_category=mission_category
    )
    anchor = should_anchor_single_model(mission, viable, mission_category)

    bullet_lines = [
        _bullet_phrase(
            rec.model,
            idx,
            _bullet_clause(rec, mission, idx, partial=partial),
            style,
            seed,
            partial=partial,
        )
        for idx, rec in enumerate(viable)
    ]

    budget_m = None
    if mission.budget_usd and mission.budget_usd >= 5_000_000:
        budget_m = mission.budget_usd / 1_000_000

    category_framing = ""
    operational_reality = ""
    model_transition = ""
    if tiered:
        category_framing = build_category_framing_line(
            mission,
            mission_category,
            viable,
            route_phrase=route,
            operational_context=op_ctx,
        )
        operational_reality = build_operational_reality_block(
            mission,
            route_assessments,
            operational_context=op_ctx,
        )
        model_transition = build_model_transition_line(
            [r.model for r in viable],
            seed=seed,
        )

    return AdvisorPhraseBundle(
        route=route,
        operational_context=op_ctx,
        top_model=top.model,
        lead_reason=_lead_reason(top, 0),
        partial=partial,
        passenger_count=mission.passenger_count,
        budget_millions=budget_m,
        viable_models=[r.model for r in viable],
        bullet_lines=bullet_lines,
        tradeoff_block=_build_tradeoff_block(mission, top, alts, route_assessments, comparison),
        route_ops_block=_build_route_ops_block(route_assessments),
        comparison_models=comparison_models,
        use_tiered_framing=tiered,
        anchor_single_model=anchor,
        category_framing=category_framing,
        operational_reality=operational_reality,
        model_transition=model_transition,
    )


def sanitize_advisor_output(text: str) -> str:
    """Remove internal labels, scores, and robotic consultant phrasing."""
    if not (text or "").strip():
        return ""

    try:
        from rag.pinpoint_answer import strip_advisory_boilerplate

        text = strip_advisory_boilerplate(text)
    except Exception:
        pass

    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        if _STRIP_HEADER_RE.match(line):
            continue
        if line in ("Mission Fit:", "Aircraft Options:", "Verdict:"):
            lines.append(line)
            continue
        if line.lower() in ("top options:", "conditional options:"):
            continue
        line = _SCORE_METADATA_RE.sub("", line)
        if not line.startswith(("Mission Fit:", "Aircraft Options:", "Verdict:", "Comparison:")):
            line = _INTERNAL_TAXONOMY_RE.sub("", line)
        line = _ROBOTIC_PHRASE_RE.sub("", line)
        line = _HEDGING_PHRASE_RE.sub("", line)
        line = re.sub(r"\s*—\s*watch:\s*", " — ", line, flags=re.I)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)

    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = out.strip()
    try:
        from services.consultant.broker_advisory_layer import sanitize_broker_prose

        out = sanitize_broker_prose(out)
    except Exception:
        pass
    try:
        from services.consultant.response_cleanup import cleanResponseText

        out = cleanResponseText(out)
    except Exception:
        pass
    return out


def _append_clarification_question(body: str, needs) -> str:
    """One focused follow-up — only for allowed clarification gaps."""
    if not (body or "").strip():
        return body
    extras = build_clarification_questions(needs)
    if not extras:
        return body
    tail = " ".join(extras)
    if tail.lower() in body.lower():
        return body
    return f"{body.rstrip()}\n\n{tail}"


def _is_ownership_economics_query(query: str) -> bool:
    """Fractional vs full ownership / DOC — not a mission shortlist turn."""
    ql = (query or "").lower()
    if not any(
        tok in ql
        for tok in (
            "fractional",
            "full ownership",
            "ownership vs",
            "own vs",
            "own or charter",
            "hours a year",
            "hours per year",
            "cost of ownership",
            "leaning fractional",
            "overbuying",
        )
    ):
        return False
    # Route-first mission asks stay in the aircraft pipeline
    if re.search(r"\b(?:from|to)\s+\w+.*\b(?:nonstop|pax|passengers)\b", ql):
        return False
    if re.search(r"\bnew\s+york\s+to\s+|\bsfo\s+to\s+|\b\d+\s+pax\b", ql) and "fractional" not in ql:
        return False
    return True


def format_ownership_advisory_response(
    query: str,
    *,
    mission: MissionState,
    anchor_model: str = "",
) -> str:
    """Ownership / fractional economics — no ranked shortlist template."""
    ql = (query or "").lower()
    model = (anchor_model or "").strip()
    if not model:
        mentioned = []
        try:
            from services.consultant.recommendation_engine import detect_models_from_text

            mentioned = detect_models_from_text(query)
        except Exception:
            pass
        model = mentioned[0] if mentioned else ""

    hours_match = re.search(r"(\d{2,3})\s+hours?\s+(?:a|per)\s+year", ql)
    hours = hours_match.group(1) if hours_match else ""

    lines: List[str] = []
    if "fractional" in ql and model:
        lines.append(
            f"At roughly {hours or '200–300'} hours a year on a {model}, fractional usually wins on "
            "capital at risk and dispatch simplicity — you are buying access, not managing maintenance downtime."
        )
        lines.append(
            "Full ownership starts to make sense when utilization is steady, you want tail control, "
            "and you are willing to carry crew, mx programs, and residual risk yourself."
        )
    elif model:
        lines.append(
            f"On a {model}, the real decision is utilization: below ~400–500 hours a year, "
            "fractional or charter often carries the airplane cheaper than owning."
        )
        lines.append(
            "Above that band, ownership can work — but only if you budget for mx downtime, "
            "crew, and the months the jet sits."
        )
    else:
        lines.append(
            "Based on what you have described so far, I would size the ownership question to "
            "annual hours and how much dispatch control you need — not brochure performance."
        )

    lines.append(
        "In practice, most buyers overweight acquisition price and underweight fixed cost, "
        "mx reserves, and liquidity when they try to exit."
    )
    return sanitize_advisor_output("\n\n".join(lines))


def format_route_clarification_response(
    *,
    mission: MissionState,
    clarifying_question: str,
) -> str:
    """Ask for route only — direct, no hedging."""
    route = _route_phrase(mission)
    pax = mission.passenger_count
    if route:
        return sanitize_advisor_output(
            f"On {route}, tell me passenger count and I'll name the lead aircraft."
        )
    if pax:
        return sanitize_advisor_output(
            f"With {pax} passengers, what's the primary city pair — origin and destination?"
        )
    q = (clarifying_question or "").strip()
    if q:
        return sanitize_advisor_output(q)
    return sanitize_advisor_output(
        "What's the primary route — origin and destination? "
        "I'll match you to the right aircraft class for that leg."
    )


def format_consultant_response(
    *,
    mission: MissionState,
    recommendations: List[AircraftRecommendation],
    route_assessments: List[RouteFeasibilityAssessment],
    comparison: Optional[StructuredComparison] = None,
    draft_answer: str = "",
    include_comparison_section: bool = False,
    query: str = "",
    turn_seed: str = "",
    response_style: Optional[str] = None,
    clarifications_already_asked: int = 0,
    eliminated_models: Optional[List[str]] = None,
    data_used: Optional[Dict[str, Any]] = None,
    include_acquisition_intelligence: bool = False,
) -> str:
    """
    Build varied advisor response — structure and phrasing change by style and turn.
    """
    del draft_answer, include_comparison_section

    from services.elimination.elimination_invariant import (
        assert_elimination_invariant,
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
        context="formatter",
    )

    if _is_ownership_economics_query(query):
        anchor = viable[0].model if (viable := [r for r in recommendations if not r.avoid]) else ""
        body = format_ownership_advisory_response(query, mission=mission, anchor_model=anchor)
        try:
            from services.consultant.phrase_repetition_guard import apply_phrase_repetition_guard

            body, _ = apply_phrase_repetition_guard(body, turn_seed=query)
        except Exception:
            pass
        return body

    viable = [r for r in recommendations if not r.avoid]
    try:
        assert_elimination_invariant([r.model for r in viable], eliminated)
    except AssertionError:
        viable = [
            r
            for r in viable
            if (r.model or "").strip().lower() not in eliminated
        ]

    needs = mission_clarification_needs(
        mission,
        query,
        recommendations=viable,
        clarifications_already_asked=clarifications_already_asked,
    )

    if needs.needs_route:
        return format_route_clarification_response(
            mission=mission,
            clarifying_question="",
        )

    if not viable:
        route = _route_phrase(mission) or ""
        pax = mission.passenger_count
        nonstop = bool(mission.nonstop_requirement)

        if _is_ownership_economics_query(query):
            return format_ownership_advisory_response(query, mission=mission)

        if query and (mission.routes or mission.mountain_airport_requirement):
            try:
                from services.recommendation.recommendation_pipeline import (
                    run_recommendation_pipeline,
                )

                pipe, _ = run_recommendation_pipeline(query)
                fallback = [r for r in (pipe.recommendations or []) if not r.avoid]
                if fallback:
                    return format_consultant_response(
                        mission=pipe.mission_state or mission,
                        recommendations=fallback,
                        route_assessments=route_assessments,
                        comparison=comparison,
                        query=query,
                        turn_seed=turn_seed or query,
                        response_style=response_style,
                        clarifications_already_asked=clarifications_already_asked,
                    )
            except Exception:
                pass

        # Long-haul / ULR realism only when stage length actually demands it.
        if route:
            from services.consultant.route_feasibility import estimate_route_distance_nm

            leg_nm = estimate_route_distance_nm(route)
            if leg_nm >= 3500 and (nonstop or leg_nm >= 4500):
                pax_phrase = f" with {pax} passengers" if pax is not None else ""
                nonstop_phrase = " nonstop" if nonstop else ""
                return sanitize_advisor_output(
                    f"I wouldn’t try to force a smaller aircraft into {route}{pax_phrase}{nonstop_phrase}. "
                    "Once you layer in NBAA IFR reserves, a realistic passenger payload/bags, and winter westbound margin when it applies, "
                    "the mission either moves up a class or it needs a tech stop.\n\n"
                    "If nonstop is non‑negotiable, we should talk in terms of the next aircraft class up — not a \"close on paper\" midsize/super‑mid. "
                    "If you’re open to one stop, I can narrow the realistic options quickly."
                )
            if isinstance(data_used, dict) and data_used.get("mission_authority_bound"):
                from services.mission.mission_understanding_engine import (
                    format_understanding_first_advisory,
                    load_mission_understanding,
                )

                pkt = load_mission_understanding(data_used)
                if pkt is not None:
                    return format_understanding_first_advisory(
                        mission,
                        pkt,
                        query=query,
                        data_used=data_used,
                    )
            try:
                from services.consultant.recommendation_engine import rank_aircraft_recommendations

                fallback = rank_aircraft_recommendations(mission, max_results=3)
                fallback = [r for r in fallback if not r.avoid]
                if fallback:
                    return format_consultant_response(
                        mission=mission,
                        recommendations=fallback,
                        route_assessments=route_assessments,
                        comparison=comparison,
                        query=query,
                        turn_seed=turn_seed or query,
                        response_style=response_style,
                        clarifications_already_asked=clarifications_already_asked,
                    )
            except Exception:
                pass

        if needs.needs_category_usage or needs.needs_runway_detail or needs.needs_budget:
            followups = build_clarification_questions(needs)
            if followups:
                return sanitize_advisor_output(followups[0])

        if needs.needs_passenger_count:
            return sanitize_advisor_output(
                "What’s the primary route and passenger count? "
                "That’s enough for me to put you in the right aircraft class without guessing."
            )
        if needs.needs_route or route_truly_missing(mission, query):
            return sanitize_advisor_output(
                "What’s the primary city pair? Origin and destination are enough — I’ll size the aircraft class to that leg."
            )
        if mission.routes and mission_well_defined(
            mission,
            query,
            clarifications_already_asked=clarifications_already_asked,
        ):
            route = _route_phrase(mission) or "this route"
            pax = mission.passenger_count
            pax_phrase = f" with {pax} passengers" if pax is not None else ""
            return sanitize_advisor_output(
                f"On {route}{pax_phrase}, the realistic band is large-cabin or ultra-long-range — "
                "not a smaller jet stretched to the limit. I can name specific models once hard feasibility "
                "is re-run on your longest leg."
            )
        return sanitize_advisor_output(
            "Tell me the typical city pair or mission pattern and I’ll narrow this to the right aircraft class."
        )

    partial = False
    seed = turn_seed or f"{query}|{mission.passenger_count}|{'|'.join(mission.routes)}"

    global _LAST_RESPONSE_STYLE

    from services.consultant.broker_advisory_layer import format_broker_advisory_response
    from services.consultant.response_architecture import format_comparison_architecture

    if comparison and len(comparison.models or []) >= 2:
        ql = (query or "").lower()
        compare_only = bool(
            re.search(r"\bcompare\b|\bvs\.?\b|versus", ql)
            and not re.search(r"\brecommend\b", ql)
        )
        body = format_comparison_architecture(comparison.models)
        if viable and not compare_only:
            body = (
                body
                + "\n\n"
                + format_broker_advisory_response(
                    mission,
                    viable[:3],
                    route_assessments,
                    query=query,
                    eliminated_models=eliminated_models,
                    data_used=data_used,
                )
            )
    else:
        body = format_broker_advisory_response(
            mission,
            viable[:3],
            route_assessments,
            query=query,
            eliminated_models=eliminated_models,
            data_used=data_used,
        )
        if isinstance(data_used, dict):
            fleet_raw = data_used.get("fleet_composition_plan")
            if isinstance(fleet_raw, dict) and fleet_raw.get("multi_aircraft_required"):
                try:
                    from services.fleet.fleet_composition import (
                        FleetCompositionPlan,
                        FleetRoleAssignment,
                        MissionSegment,
                        MissionSegmentRole,
                        format_fleet_composition_block,
                    )

                    plan = FleetCompositionPlan(
                        multi_aircraft_required=True,
                        doctrine=str(fleet_raw.get("doctrine") or ""),
                        ownership_note=str(fleet_raw.get("ownership_note") or ""),
                    )
                    for s in fleet_raw.get("segments") or []:
                        if isinstance(s, dict):
                            try:
                                seg_role = MissionSegmentRole(
                                    str(s.get("role") or "coast_to_coast")
                                )
                            except ValueError:
                                seg_role = MissionSegmentRole.COAST_TO_COAST
                            plan.segments.append(
                                MissionSegment(
                                    role=seg_role,
                                    label=str(s.get("label") or ""),
                                    stage_nm=float(s.get("stage_nm") or 0),
                                    required_nm=float(s.get("required_nm") or 0),
                                    route_labels=list(s.get("route_labels") or []),
                                    notes=list(s.get("notes") or []),
                                )
                            )
                    for a in fleet_raw.get("assignments") or []:
                        if isinstance(a, dict):
                            try:
                                assign_role = MissionSegmentRole(
                                    str(a.get("role") or "coast_to_coast")
                                )
                            except ValueError:
                                assign_role = MissionSegmentRole.COAST_TO_COAST
                            plan.assignments.append(
                                FleetRoleAssignment(
                                    role=assign_role,
                                    segment_label=str(a.get("segment_label") or ""),
                                    primary_model=str(a.get("primary_model") or ""),
                                    fit_verdict=str(a.get("fit_verdict") or ""),
                                    rationale=str(a.get("rationale") or ""),
                                    alternates=list(a.get("alternates") or []),
                                )
                            )
                    fleet_block = format_fleet_composition_block(plan)
                    if fleet_block:
                        body = f"{body}\n\n{fleet_block}"
                except Exception:
                    pass
    if not body:
        from services.consultant.recommendation_framing import use_tiered_advisor_framing
        from services.recommendation.mission_ranker import classify_mission_category

        mission_category = classify_mission_category(mission)
        tiered = use_tiered_advisor_framing(
            mission,
            viable,
            route_assessments,
            mission_category=mission_category,
        )
        vctx = VariationContext(
            mission=mission,
            query=query,
            partial_mission=partial,
            comparison=comparison,
            route_assessments=route_assessments,
            turn_seed=seed,
            use_tiered_framing=tiered,
        )
        style = response_style or select_response_style(vctx)
        bundle = _build_phrase_bundle(
            mission,
            viable,
            route_assessments,
            comparison,
            partial=partial,
            style=style,
            seed=seed,
        )
        body, style_used = compose_varied_response(bundle, vctx, style=style)
        _LAST_RESPONSE_STYLE = style_used
    else:
        from services.consultant.response_variation import STYLE_BROKER_ADVISORY

        _LAST_RESPONSE_STYLE = STYLE_BROKER_ADVISORY

    body = sanitize_advisor_output(body)
    ql_acq = (query or "").lower()
    if (
        include_acquisition_intelligence
        or re.search(r"\b(?:acquire|acquisition|buy|purchase|resale|liquidity)\b", ql_acq)
    ) and viable:
        try:
            from services.acquisition.acquisition_advisory import (
                format_acquisition_intelligence_block,
            )

            body = (
                f"{body.rstrip()}\n\n"
                f"{format_acquisition_intelligence_block([r.model for r in viable[:3]])}"
            )
        except Exception:
            pass

    if (
        needs.needs_passenger_count
        or needs.needs_budget
        or needs.needs_category_usage
        or needs.needs_runway_detail
    ):
        body = _append_clarification_question(body, needs)
    elif mission_maps_to_category(mission, query):
        body = _strip_trailing_clarification_footers(body)
    try:
        from services.consultant.phrase_repetition_guard import apply_phrase_repetition_guard

        body, _ = apply_phrase_repetition_guard(body, turn_seed=query or turn_seed)
    except Exception:
        pass
    return body


def _strip_trailing_clarification_footers(text: str) -> str:
    """Remove generic 'tell me more' closers when the mission is already decisive."""
    lines = text.splitlines()
    while lines:
        last = lines[-1].strip().lower()
        if not last:
            lines.pop()
            continue
        if any(
            p in last
            for p in (
                "tell me passenger",
                "how many passengers",
                "city pair",
                "origin and destination",
                "before we lock",
                "pin the city pair",
            )
        ):
            lines.pop()
            continue
        break
    return "\n".join(lines).strip()


def last_response_style() -> str:
    """Style used on the most recent ``format_consultant_response`` call."""
    return _LAST_RESPONSE_STYLE


def should_use_structured_formatter(
    data_used: Optional[Dict[str, Any]],
    mission: MissionState,
    query: str,
) -> bool:
    du = data_used if isinstance(data_used, dict) else {}
    qri = str(du.get("query_recommendation_intent") or "").strip().lower()
    if qri in (
        "aircraft_critique",
        "ownership_economics",
        "payload_range_analysis",
        "visualization_request",
    ):
        return False
    if qri == "aircraft_comparison":
        return True
    mode = str(
        du.get("consultant_response_mode_canonical") or du.get("consultant_response_mode") or ""
    ).lower()
    if mode in (
        "advisory_mode",
        "advisory",
        "mission_advisory",
        "client_decision_scenarios",
        "comparison_mode",
        "followup_continuation",
    ):
        return True
    ql = (query or "").lower()
    if any(
        w in ql
        for w in (
            "recommend",
            "best jet",
            "best aircraft",
            "compare",
            " versus ",
            " vs ",
            "which should",
            "fractional",
            "ownership",
            "nonstop",
            "westbound",
            "range map",
        )
    ):
        return True
    if mission.routes or mission.passenger_count or mission.budget_usd:
        return True
    return False
