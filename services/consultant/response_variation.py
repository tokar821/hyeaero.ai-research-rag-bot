"""
Response variation engine — varied structure and phrasing per turn.

Styles: concise executive, advisor narrative, comparison-driven, operational analysis,
recommendation-first, tradeoff-first.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional

from services.consultant.comparison_engine import StructuredComparison
from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.consultant.route_feasibility import RouteFeasibilityAssessment

# --- Allowed response styles ---
STYLE_CONCISE_EXECUTIVE = "concise_executive"
STYLE_ADVISOR_NARRATIVE = "advisor_narrative"
STYLE_COMPARISON_DRIVEN = "comparison_driven"
STYLE_OPERATIONAL_ANALYSIS = "operational_analysis"
STYLE_RECOMMENDATION_FIRST = "recommendation_first"
STYLE_TRADEOFF_FIRST = "tradeoff_first"
STYLE_BROKER_ADVISORY = "broker_advisory"

ALL_STYLES = (
    STYLE_CONCISE_EXECUTIVE,
    STYLE_ADVISOR_NARRATIVE,
    STYLE_COMPARISON_DRIVEN,
    STYLE_OPERATIONAL_ANALYSIS,
    STYLE_RECOMMENDATION_FIRST,
    STYLE_TRADEOFF_FIRST,
    STYLE_BROKER_ADVISORY,
)

_BANNED_OPENING_PHRASES = (
    "on my list",
    "starts the conversation",
    "worth revisiting",
    "if the trip stays in this profile",
    "most balanced operator",
)


@dataclass
class VariationContext:
    mission: MissionState
    query: str = ""
    partial_mission: bool = False
    comparison: Optional[StructuredComparison] = None
    route_assessments: List[RouteFeasibilityAssessment] = field(default_factory=list)
    turn_seed: str = ""
    use_tiered_framing: bool = False


@dataclass
class AdvisorPhraseBundle:
    """Resolved phrases for one response — built by response_formatter helpers."""

    route: str
    operational_context: str
    top_model: str
    lead_reason: str
    partial: bool
    passenger_count: Optional[int]
    budget_millions: Optional[float]
    viable_models: List[str]
    bullet_lines: List[str]
    tradeoff_block: str
    route_ops_block: str
    comparison_models: List[str]
    use_tiered_framing: bool = False
    anchor_single_model: bool = True
    category_framing: str = ""
    operational_reality: str = ""
    model_transition: str = ""


def _stable_index(key: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def _pick(key: str, options: List[str]) -> str:
    if not options:
        return ""
    return options[_stable_index(key, len(options))]


def select_response_style(ctx: VariationContext) -> str:
    """
    Choose a response style from context — deterministic per turn, varied across turns.
    """
    ql = (ctx.query or "").lower()
    route_key = "|".join(ctx.mission.routes or [])
    seed = ctx.turn_seed or f"{ql}|{route_key}|{ctx.mission.passenger_count}"

    if ctx.comparison and len(ctx.comparison.models or []) >= 2:
        pool = [STYLE_COMPARISON_DRIVEN, STYLE_ADVISOR_NARRATIVE, STYLE_TRADEOFF_FIRST]
        return _pick(f"{seed}:cmp", pool)

    if re.search(r"\bcompare|versus|vs\.?\b", ql) and _comparison_models_hint(ql):
        return _pick(f"{seed}:vs", [STYLE_COMPARISON_DRIVEN, STYLE_TRADEOFF_FIRST, STYLE_ADVISOR_NARRATIVE])

    if ctx.route_assessments and any(
        a.classification in ("not_feasible", "practical_restricted") for a in ctx.route_assessments
    ):
        return _pick(
            f"{seed}:ops",
            [STYLE_OPERATIONAL_ANALYSIS, STYLE_TRADEOFF_FIRST, STYLE_ADVISOR_NARRATIVE],
        )

    if re.search(r"\b(runway|payload|feasibility|nonstop|range|westbound|operating cost)\b", ql):
        return _pick(
            f"{seed}:opq",
            [STYLE_OPERATIONAL_ANALYSIS, STYLE_TRADEOFF_FIRST, STYLE_RECOMMENDATION_FIRST],
        )

    if ctx.partial_mission:
        return _pick(
            f"{seed}:partial",
            [STYLE_RECOMMENDATION_FIRST, STYLE_CONCISE_EXECUTIVE, STYLE_ADVISOR_NARRATIVE],
        )

    if re.search(r"\b(ceo|executive|brief|short|quick|tldr|bottom line)\b", ql):
        return STYLE_CONCISE_EXECUTIVE

    if ctx.use_tiered_framing:
        return _pick(
            f"{seed}:tiered_style",
            [
                STYLE_ADVISOR_NARRATIVE,
                STYLE_OPERATIONAL_ANALYSIS,
                STYLE_TRADEOFF_FIRST,
            ],
        )

    if re.search(r"\b(recommend|best|which|should i|what jet|what aircraft)\b", ql):
        pool = [
            STYLE_RECOMMENDATION_FIRST,
            STYLE_ADVISOR_NARRATIVE,
            STYLE_CONCISE_EXECUTIVE,
            STYLE_OPERATIONAL_ANALYSIS,
            STYLE_TRADEOFF_FIRST,
        ]
        return _pick(f"{seed}:rec", pool)

    return _pick(f"{seed}:default", list(ALL_STYLES))


def _comparison_models_hint(query: str) -> bool:
    return bool(re.search(r"\b(gulfstream|falcon|citation|challenger|global|phenom|praetor|pc-24)\b", query, re.I))


def _bullet_phrase(
    model: str,
    index: int,
    clause: str,
    style: str,
    seed: str,
    *,
    partial: bool,
) -> str:
    """Varied bullet phrasing by style."""
    clause = (clause or "").strip() or "this mission profile"
    key = f"{seed}:bullet:{index}:{model}"
    if partial:
        if index == 0:
            templates = [
                f"- {model} — lead option when {clause}",
                f"- {model} — I'd start here when {clause}",
                f"- {model} — first call when {clause}",
            ]
        else:
            templates = [
                f"- {model} — strong alternate when {clause}",
                f"- {model} — next look when {clause}",
                f"- {model} — backup when {clause}",
            ]
        return _pick(key, templates)

    if style == STYLE_CONCISE_EXECUTIVE:
        templates = [
            f"- {model} — {clause}",
            f"- {model}: {clause}",
        ]
        return _pick(key, templates)

    if style == STYLE_COMPARISON_DRIVEN:
        templates = [
            f"- {model} — {clause}",
            f"- {model}: {clause}",
            f"- {model} — compares well when {clause}",
        ]
        return _pick(key, templates)

    if style == STYLE_OPERATIONAL_ANALYSIS:
        templates = [
            f"- {model} — operationally, {clause}",
            f"- {model} — {clause}",
            f"- {model}: strong on {clause}",
        ]
        return _pick(key, templates)

    if index == 0:
        templates = [
            f"- {model} — my first call when {clause}",
            f"- {model} — lead option for {clause}",
            f"- {model} — I'd put at the top for {clause}",
            f"- {model} — the clean dispatch answer when {clause}",
        ]
    elif index == 1:
        templates = [
            f"- {model} — {clause}",
            f"- {model} — alternate if {clause}",
            f"- {model} — next look if {clause}",
        ]
    else:
        templates = [
            f"- {model} — {clause}",
            f"- {model} — also consider if {clause}",
        ]
    return _pick(key, templates)


def _opening_phrases(bundle: AdvisorPhraseBundle, style: str, seed: str) -> str:
    route, top, reason, ctx = bundle.route, bundle.top_model, bundle.lead_reason, bundle.operational_context
    pax, budget = bundle.passenger_count, bundle.budget_millions

    if bundle.use_tiered_framing and not bundle.anchor_single_model:
        return ""

    # Opening archetype diversification (deterministic per turn).
    # These are intentionally different in cadence/shape to reduce the “list generator” feel.
    archetype = _stable_index(f"{seed}:open_archetype", 7)
    if not bundle.partial and style not in (STYLE_CONCISE_EXECUTIVE, STYLE_TRADEOFF_FIRST):
        elim = _pick(
            f"{seed}:elim_open",
            [
                (
                    f"I’d be careful trying to force a smaller jet onto {route}."
                    if route
                    else "I’d be careful trying to force a smaller jet onto this mission."
                ),
                (
                    f"On {route}, brochure range can make a few airplanes look close; in practice the margin disappears fast."
                    if route
                    else "Brochure range can make a few airplanes look close; in practice the margin disappears fast."
                ),
                "The first decision here isn’t a model — it’s whether you want clean dispatch margin or a \"close on paper\" solution.",
            ],
        )
        dispatch = _pick(
            f"{seed}:dispatch_open",
            [
                (
                    f"Dispatch reality on {route}: plan this like an operator, not a marketing chart."
                    if route
                    else "Dispatch reality: plan this like an operator, not a marketing chart."
                ),
                "If you want this to go nonstop reliably, you need payload-and-reserve margin, not a brochure number.",
                "Most operators who fly this hard learn quickly that the mission is won or lost on reserves, winds, and usable payload.",
            ],
        )
        psych = _pick(
            f"{seed}:psych_open",
            [
                "The ownership question underneath this is whether you’re buying for the 80% mission or the one trip you refuse to compromise on.",
                "If you’re shopping this category, you’re really trading cabin experience against operating friction and dispatch margin.",
                "Owners rarely regret buying a little too much margin; they do regret buying the airplane that needs \"perfect day\" planning.",
            ],
        )
        strategic = _pick(
            f"{seed}:strat_open",
            [
                (
                    f"For {pax} passengers on {route}, I’d frame this as a capability bracket first, then pick models."
                    if route and pax
                    else (
                        f"For {route}, I’d frame this as a capability bracket first, then pick models."
                        if route
                        else "I’d frame this as a capability bracket first, then pick models."
                    )
                ),
                "Let’s size the airplane to how aggressively you want to fly the mission — conservative dispatch margin vs. \"it usually makes it\".",
                "The clean way to do this is to decide what you’re unwilling to give up (nonstop? runway? cabin?) and let the shortlist fall out.",
            ],
        )
        if archetype == 0:
            return elim
        if archetype == 1:
            return dispatch
        if archetype == 2:
            return psych
        if archetype == 3:
            return strategic

    if bundle.partial:
        partial_opts = [
            (
                f"With {pax} passengers, what's the primary city pair? "
                "Here's the aircraft class I'd line up for that trip."
            )
            if pax
            else (
                "What's the primary route — origin and destination? "
                "Here's the aircraft class I'd line up for that trip."
            ),
            (
                f"With {pax} passengers, name the city pair and I'll name the lead aircraft — "
                "here's where I'd start on class."
            )
            if pax
            else (
                "Name the city pair and I'll name the lead aircraft — here's where I'd start on class."
            ),
        ]
        return _pick(f"{seed}:partial_open", [o for o in partial_opts if o])

    if style == STYLE_CONCISE_EXECUTIVE:
        opts = [
            f"{top} — best match for {route}." if route else f"{top} — best match for {ctx}.",
            (
                f"For {pax} pax on {route}, {top}." if route and pax else f"{top} for {ctx}."
            ),
            f"Lead aircraft: {top} on {route}." if route else f"Lead aircraft: {top}.",
        ]
        return _pick(f"{seed}:exec_open", opts)

    if style == STYLE_COMPARISON_DRIVEN and bundle.comparison_models:
        a, b = bundle.comparison_models[0], bundle.comparison_models[1]
        opts = [
            f"Between {a} and {b}, the mission profile favors {top} — {reason}.",
            f"If you're cross-shopping {a} and {b}, I'd lean {top} for {ctx}.",
            f"Head-to-head, {top} wins on {ctx} — {a} and {b} trade different strengths.",
        ]
        return _pick(f"{seed}:cmp_open", opts)

    if style == STYLE_OPERATIONAL_ANALYSIS:
        opts = [
            (
                f"Operationally on {route}, {top} is where I'd land — {reason}."
            )
            if route
            else f"On the operational side, {top} is where I'd land — {reason}.",
            (
                f"For {ctx}, {top} handles the leg cleanly — {reason}."
            ),
            f"From a fuel-and-reserves standpoint, {top} is the clean answer — {reason}.",
        ]
        return _pick(f"{seed}:ops_open", opts)

    if style == STYLE_TRADEOFF_FIRST:
        return ""  # tradeoff block leads; opening may be omitted

    if style == STYLE_ADVISOR_NARRATIVE:
        opts = [
            (
                f"If you're flying {pax} passengers {route}, I've seen operators do well with "
                f"{top} — {reason}."
            )
            if route and pax
            else (
                f"For {ctx}, I've seen operators do well with {top} — {reason}."
            ),
            (
                f"On {route}, most owners in your position end up circling {top} first — {reason}."
            )
            if route
            else f"Most owners in your position end up circling {top} first — {reason}.",
            (
                f"What usually works on {route} is {top} — {reason}."
            )
            if route
            else f"What usually works here is {top} — {reason}.",
        ]
        return _pick(f"{seed}:narr_open", opts)

    # recommendation_first + default
    opts = [
        (
            f"With {pax} passengers on {route}, I'd start with {top} — {reason}."
        )
        if route and pax
        else f"On {route}, I'd start with {top} — {reason}." if route else f"I'd start with {top} — {reason}.",
        (
            f"For {ctx}, {top} is my first call — {reason}."
        ),
        (
            f"Realistically, {top} is the cleanest operator on {route} — {reason}."
            if route
            else f"Realistically, {top} is the cleanest operator here — {reason}."
        ),
        (
            f"On {route}, {top} is the airplane I'd put at the top of the list — {reason}."
        )
        if route
        else f"{top} is the airplane I'd put at the top of the list — {reason}.",
    ]
    text = _pick(f"{seed}:rec_open", opts)
    if budget and budget >= 5:
        suffix = _pick(
            f"{seed}:budget",
            [
                f" That keeps you near ${budget:.0f}M.",
                f" Realistic around ${budget:.0f}M.",
                f" Should fit a ${budget:.0f}M envelope.",
            ],
        )
        text += suffix
    return text


def _transition_phrases(style: str, seed: str) -> str:
    if style == STYLE_CONCISE_EXECUTIVE:
        return ""  # no transition — tight list follows
    if style == STYLE_ADVISOR_NARRATIVE:
        return _pick(
            f"{seed}:trans",
            [
                "Aircraft I'd keep on the desk:",
                "Names I'd pressure-test:",
                "Others in the conversation:",
            ],
        )
    if style == STYLE_COMPARISON_DRIVEN:
        return _pick(
            f"{seed}:trans",
            [
                "How the alternates stack up:",
                "Side-by-side:",
                "Also in the mix:",
            ],
        )
    if style == STYLE_OPERATIONAL_ANALYSIS:
        return _pick(
            f"{seed}:trans",
            [
                "From an operating standpoint:",
                "Worth weighing on capability:",
                "On paper for this trip:",
            ],
        )
    return _pick(
        f"{seed}:trans",
        [
            "Where I’d focus first:",
            "The short list I’d pressure‑test:",
            "A practical set to evaluate:",
            "If you want this to dispatch cleanly, look at:",
            "The few I’d actually put in front of an operator:",
        ],
    )


def _conclusion_phrases(bundle: AdvisorPhraseBundle, style: str, seed: str) -> str:
    if bundle.use_tiered_framing and not bundle.anchor_single_model:
        from services.consultant.recommendation_framing import build_tiered_conclusion

        return build_tiered_conclusion(
            bundle.viable_models,
            bundle.tradeoff_block,
            seed=seed,
        )

    if bundle.partial:
        names = bundle.viable_models
        if len(names) >= 2:
            lead, alt = names[0], names[1]
            return _pick(
                f"{seed}:partial_close",
                [
                    f"Longer legs push me toward {alt}; shorter hops keep {lead} in the sweet spot.",
                    f"I'd default to {lead} — move to {alt} if range becomes the driver.",
                ],
            )
        if names:
            return _pick(
                f"{seed}:partial_close1",
                [
                    f"{names[0]} is the balanced pick for this profile.",
                    f"I'd put {names[0]} at the top until the route is confirmed.",
                ],
            )
        return ""

    top = bundle.top_model
    alts = bundle.viable_models[1:3]
    trade = bundle.tradeoff_block

    if style == STYLE_TRADEOFF_FIRST and trade:
        return trade

    if style == STYLE_CONCISE_EXECUTIVE:
        if trade:
            return trade
        if alts:
            alt = alts[0]
            return _pick(
                f"{seed}:exec_close",
                [
                    f"If cost matters more than cabin, consider {alt}.",
                    f"Pivot to {alt} if runway or burn becomes the driver.",
                ],
            )
        return ""

    if trade:
        closes = [
            trade,
            f"{trade} The only thing that really changes the answer is if your legs get longer or your runway access tightens.",
            f"{trade} If you want to fly this more aggressively (winter, westbound, full cabin), buy margin up front.",
        ]
        return _pick(f"{seed}:close_trade", [c for c in closes if c])

    if alts:
        alt = alts[0]
        return _pick(
            f"{seed}:close",
            [
                f"If runway and operating cost matter more than cabin, {alt} is the smarter pivot from {top}.",
                f"Cabin over cost? Stay with {top}. Cost over cabin? Look hard at {alt}.",
                f"That answer changes if your legs get longer — then {alt} deserves a serious look.",
                f"For day-to-day flying in this profile, {top} is the most balanced operator in the group.",
            ],
        )

    return _pick(
        f"{seed}:close_default",
        [
            f"That’s the realistic bracket. The rest is preference: cabin vs cost, and how conservative you want dispatch margin.",
            f"If you tell me whether you care more about runway access or cabin comfort, I can tighten this without guessing.",
        ],
    )


def _paragraph_list(bundle: AdvisorPhraseBundle, *, seed: str) -> str:
    """Non-bulleted shortlist for a more human, written feel."""
    models = bundle.viable_models[:3]
    if not models:
        return ""
    route = bundle.route
    pax = bundle.passenger_count
    ctx = bundle.operational_context
    lines: List[str] = []
    seen: set = set()
    idx = 0
    for m in models:
        if m in seen:
            continue
        seen.add(m)
        key = f"{seed}:para:{idx}:{m}"
        lead = _pick(
            key,
            [
                f"{m} is the clean dispatch answer here",
                f"{m} is the low-drama way to fly this",
                f"{m} tends to keep you out of the \"perfect day\" planning trap",
            ],
        )
        if idx == 0 and route and pax:
            lines.append(f"{lead} for {pax} pax on {route}.")
        elif idx == 0 and route:
            lines.append(f"{lead} on {route}.")
        else:
            lines.append(
                f"{m} is the next place I’d look if priorities shift (cabin vs cost vs runway)."
            )
        idx += 1
    return " ".join(lines).strip()


def _compose_tiered_advisor_response(
    bundle: AdvisorPhraseBundle,
    ctx: VariationContext,
    chosen: str,
    seed: str,
) -> tuple[str, str]:
    """Category → operational reality → models (senior advisor framing)."""
    blocks: List[str] = []
    if bundle.category_framing:
        blocks.append(bundle.category_framing)
    ops = bundle.operational_reality or bundle.route_ops_block
    if ops:
        blocks.append(ops)
    bullets = "\n".join(bundle.bullet_lines)
    if bullets:
        if bundle.model_transition:
            blocks.append(bundle.model_transition)
        blocks.append(bullets)
    conclusion = _conclusion_phrases(bundle, chosen, seed)
    if conclusion:
        blocks.append(conclusion)
    body = "\n\n".join(b for b in blocks if b and b.strip())
    return body, chosen


def compose_varied_response(
    bundle: AdvisorPhraseBundle,
    ctx: VariationContext,
    *,
    style: Optional[str] = None,
) -> tuple[str, str]:
    """
    Assemble response sections in a style-dependent order.

    Returns ``(body_text, style_used)``.
    """
    chosen = style or select_response_style(ctx)
    seed = ctx.turn_seed or "default"

    if bundle.use_tiered_framing and not bundle.anchor_single_model:
        return _compose_tiered_advisor_response(bundle, ctx, chosen, seed)

    opening = _opening_phrases(bundle, chosen, seed)
    transition = _transition_phrases(chosen, seed)
    # Occasionally drop bullets entirely for a written, advisor feel.
    use_paragraphs = (
        chosen in (STYLE_ADVISOR_NARRATIVE, STYLE_OPERATIONAL_ANALYSIS, STYLE_COMPARISON_DRIVEN)
        and _stable_index(f"{seed}:shape", 5) == 0
    )
    bullets = "" if use_paragraphs else "\n".join(bundle.bullet_lines)
    para = _paragraph_list(bundle, seed=seed) if use_paragraphs else ""
    conclusion = _conclusion_phrases(bundle, chosen, seed)

    blocks: List[str] = []

    if chosen == STYLE_TRADEOFF_FIRST:
        if bundle.tradeoff_block:
            blocks.append(bundle.tradeoff_block)
        if bundle.route_ops_block:
            blocks.append(bundle.route_ops_block)
        if opening:
            blocks.append(opening)
        if bullets:
            if transition:
                blocks.append(transition)
            blocks.append(bullets)
        elif transition:
            blocks.append(transition)
        if conclusion and conclusion != bundle.tradeoff_block:
            blocks.append(conclusion)

    elif chosen == STYLE_OPERATIONAL_ANALYSIS:
        if bundle.route_ops_block:
            blocks.append(bundle.route_ops_block)
        if opening:
            blocks.append(opening)
        if para:
            blocks.append(para)
        elif bullets:
            if transition:
                blocks.append(transition)
            blocks.append(bullets)
        if conclusion:
            blocks.append(conclusion)

    elif chosen == STYLE_RECOMMENDATION_FIRST:
        if opening:
            blocks.append(opening)
        if para:
            blocks.append(para)
        elif bullets:
            if transition:
                blocks.append(transition)
            blocks.append(bullets)
        if conclusion:
            blocks.append(conclusion)

    elif chosen == STYLE_CONCISE_EXECUTIVE:
        if opening:
            blocks.append(opening)
        if para:
            blocks.append(para)
        elif bullets:
            blocks.append(bullets)
        if conclusion:
            blocks.append(conclusion)

    elif chosen == STYLE_COMPARISON_DRIVEN:
        if opening:
            blocks.append(opening)
        if para:
            blocks.append(para)
        elif bullets:
            if transition:
                blocks.append(transition)
            blocks.append(bullets)
        if conclusion:
            blocks.append(conclusion)

    elif chosen == STYLE_ADVISOR_NARRATIVE:
        if opening:
            blocks.append(opening)
        if para:
            blocks.append(para)
        elif bullets:
            if transition:
                blocks.append(transition)
            blocks.append(bullets)
        if conclusion:
            blocks.append(conclusion)

    else:
        if opening:
            blocks.append(opening)
        if para:
            blocks.append(para)
        elif bullets:
            if transition:
                blocks.append(transition)
            blocks.append(bullets)
        if conclusion:
            blocks.append(conclusion)

    body = "\n\n".join(b for b in blocks if b and b.strip())
    low = body.lower()
    for bad in _BANNED_OPENING_PHRASES:
        if bad in low:
            body = re.sub(re.escape(bad), " ", body, flags=re.I)
    body = re.sub(r"\s{2,}", " ", body).strip()
    return body, chosen
