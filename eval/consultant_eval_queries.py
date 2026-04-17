"""
Qualitative / adversarial evaluation query set for the aviation consultant.

This is evaluation-only: no production code import side-effects required.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence


class EvalCategory(str, Enum):
    FACTUAL_ACCURACY = "factual_accuracy"
    REASONING_DEPTH = "reasoning_depth"
    ADVISORY_QUALITY = "advisory_quality"
    COMPARISON_QUALITY = "comparison_quality"
    TAIL_NUMBER_PRECISION = "tail_number_precision"
    VISUAL_QUERY_HANDLING = "visual_query_handling"
    HALLUCINATION_RESISTANCE = "hallucination_resistance"
    EDGE_INVALID_INPUT = "edge_invalid_input"
    CLIENT_DECISION_SCENARIOS = "client_decision_scenarios"
    TONE_PROFESSIONALISM = "tone_professionalism"


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: EvalCategory
    query: str
    notes: str = ""


def build_default_eval_cases() -> List[EvalCase]:
    """
    10–15 queries per category, designed to surface:
    - hallucinations (fake models/specs)
    - weak answers (avoidance, link-only)
    - tail/image precision errors
    - shallow comparisons and advisory logic
    """
    cases: List[EvalCase] = []

    def add(cat: EvalCategory, q: str, *, notes: str = "") -> None:
        cases.append(EvalCase(id=f"{cat.value}:{len([c for c in cases if c.category == cat]) + 1}", category=cat, query=q, notes=notes))

    # 1) FACTUAL ACCURACY
    for q in (
        "What’s the range of a Falcon 2000LXS vs Falcon 2000?",
        "Is the Phenom 300 faster than a Challenger 350?",
        "What engines does the Challenger 350 use, and what does that imply for maintenance programs?",
        "How many passengers is a Gulfstream G650 typically comfortable for on long legs?",
        "Is the Citation Latitude a true stand-up cabin? Answer like a broker, not a brochure.",
        "Compare typical baggage practicality: Falcon 2000 vs Challenger 350.",
        "Does a Global 6000 have a different cabin cross-section than a Global 5000?",
        "What’s a realistic still-air range band for a PC-12, and what does that mean for dispatch?",
        "How does the G280 cabin feel vs a Challenger 350 in real use?",
        "What’s the difference between G650 and G650ER in plain language?",
        "What is the typical cruise speed class difference: light jet vs super-midsize?",
        "Do you generally need a second pilot for a Phenom 300 in corporate use?",
    ):
        add(EvalCategory.FACTUAL_ACCURACY, q)

    # 2) REASONING DEPTH (ask for nuance / constraints / tradeoffs)
    for q in (
        "I need 2,400 nm with 8 passengers. What matters more: brochure range or usable range? Explain.",
        "If I care about dispatch reliability, what’s your broker checklist beyond specs?",
        "What are the real reasons buyers pick Falcon 2000 over Challenger 350 (or vice versa)?",
        "What should I worry about with aging avionics / parts support when buying a 15–20 year old jet?",
        "If I’m often flying short legs, why might a larger cabin jet be a worse experience?",
        "Explain why 'more range' can be a trap for first-time buyers.",
        "What’s the risk of buying the cheapest airframe on the market?",
        "How would you sanity-check an aircraft’s claimed maintenance status from a listing?",
        "If two jets can do the mission, what tends to drive resale more than spec sheets?",
        "What are the operational implications of a single-point vs dual-point fueling capability (high level)?",
    ):
        add(EvalCategory.REASONING_DEPTH, q)

    # 3) ADVISORY QUALITY (real client missions)
    for q in (
        "I fly 6 people NYC to LA twice a week — what should I buy?",
        "I fly 150 hours/year, should I own or charter?",
        "Best jet for 8 passengers Chicago to London?",
        "We do 4 passengers Dallas to Teterboro weekly, plus 2 Europe trips per year. Recommend options.",
        "I want the quietest cabin experience for executives, 6 pax, 1,800–2,200 nm legs. What fits?",
        "We need reliable winter ops into Aspen with 6 pax and bags. What should we avoid?",
        "We have $12–$18M and want a modern cabin with Wi‑Fi that feels premium. What models are realistic?",
        "We’re considering chartering but hate cancellations. What ownership structure actually improves dispatch?",
        "I’m upgrading from a King Air. I want faster but not a huge cost jump. What’s the step-up path?",
        "I’m buying for resale first, mission second. What families hold liquidity best and why?",
        "We do 10 passengers but only occasionally. Should we size for the peak or charter for peaks?",
    ):
        add(EvalCategory.ADVISORY_QUALITY, q)

    # 4) COMPARISON QUALITY (force verdict + nuance)
    for q in (
        "Falcon 2000 vs Challenger 350 — which is actually better? Don’t be vague.",
        "G650 vs Global 7500 — don’t give me marketing, tell me what matters.",
        "Latitude vs Praetor 600 — who wins for a buyer that values cabin feel?",
        "Phenom 300 vs CJ4 — which is the better owner-flown adjacent step and why?",
        "PC-12 vs TBM 960 — where do owners regret their choice?",
        "Challenger 650 vs Global 5000 — if I don’t need ultra-long range, what’s the rational choice?",
        "Falcon 7X vs G550 — which is more forgiving operationally?",
        "Global 6000 vs Global 6500 — is it worth paying up and for whom?",
        "G280 vs Challenger 350 — if I do mostly 2-hour legs, what changes?",
        "Vision Jet vs Phenom 100 — what’s the honest tradeoff?",
    ):
        add(EvalCategory.COMPARISON_QUALITY, q)

    # 5) TAIL NUMBER PRECISION (strictness + no wrong aircraft)
    for q in (
        "Show me N807JS",
        "What does N628TS look like?",
        "Find images of N123AB",
        "Do you have photos of N000ZZZ?",
        "Show me this aircraft: N807JS — and confirm it’s the same tail in each photo.",
        "Any pictures for N628TS interior?",
        "Is N807JS currently for sale? Keep it factual; don’t guess.",
        "I think N807JS is a Citation Excel — confirm or tell me you can’t verify.",
        "Show N123AB and don’t mix it with N123AC.",
        "I saw N628TS on a listing — show it, but only if you can match the reg.",
    ):
        add(EvalCategory.TAIL_NUMBER_PRECISION, q)

    # 6) VISUAL QUERY HANDLING (cabin/cockpit/exterior specificity)
    for q in (
        "Show me a Challenger 350 cockpit",
        "Let me see inside a Global 7500 bedroom",
        "Exterior of a Falcon 2000 at night",
        "Show me a Phenom 300 cabin layout",
        "What does the G650 cockpit look like?",
        "Show me the baggage compartment of a Challenger 350",
        "Find walkaround photos of a Falcon 2000LXS",
        "Show exterior vs interior differences for a Latitude (briefly).",
        "I want pictures of a G280 cabin — not exterior shots.",
        "Cockpit photos of a Global 6000, please.",
    ):
        add(EvalCategory.VISUAL_QUERY_HANDLING, q)

    # 7) HALLUCINATION RESISTANCE (fake models / fake certainty)
    for q in (
        "Tell me about the Falcon 9000",
        "What’s the range of a Gulfstream G750?",
        "What’s the range of a Gulfstream G6500?",
        "Do you have verified photos of N000ZZZ? Be honest.",
        "What’s the operating cost per hour of a G700? Give me an exact number.",
        "Who owns N123AB? If you don’t know, say so.",
        "Is the Global 10000 a good buy?",
        "What’s the list price of a brand-new Falcon 6X today? Don’t guess.",
        "Can a Phenom 300 cross the Atlantic nonstop in winter? Answer carefully.",
        "Is N807JS definitely a Citation Excel? Don’t assume.",
    ):
        add(EvalCategory.HALLUCINATION_RESISTANCE, q)

    # 8) EDGE / INVALID INPUT (ambiguous, short, pronouns)
    for q in (
        "Show me it",
        "What jet is best?",
        "Is private aviation worth it?",
        "I need something that can do it — you know what I mean.",
        "How much does it cost?",
        "Can it fly nonstop?",
        "Do you have any pictures?",
        "Falcon 2000LXS cockpit vs cabin — show me",
        "N807JS vs N807JT — show me the right one",
        "Global 7500 interior — but I’m on a budget",
    ):
        add(EvalCategory.EDGE_INVALID_INPUT, q)

    # 9) CLIENT-LIKE DECISION SCENARIOS (high expectation)
    for q in (
        "I’m a first-time buyer. I want one aircraft that covers 90% of missions: 6–8 pax, 1,800–2,500 nm. What’s the smart play?",
        "We’re replacing NetJets hours. We want control, but don’t want staffing pain. What structure do you recommend?",
        "I need to justify this to a CFO. What’s the decision logic to pick charter vs own vs fractional?",
        "We fly 2 pilots already. We care about cabin experience and dispatch reliability more than max range. What families are safest bets?",
        "We’re sensitive to resale. Which segment is the most liquid right now and why (conceptually)?",
        "I want to avoid buyer’s remorse. What are the top 5 questions you’d force me to answer before shopping?",
        "We often book last minute. How should that change the aircraft choice and ownership choice?",
        "We’ll do occasional international but mostly domestic. Should we buy for the exceptions?",
        "We need an aircraft that feels premium for clients, but we hate the 'empty cost'. What’s the pragmatic compromise?",
        "We’re looking at a 2008–2012 large cabin jet. What are the hidden costs that blow up budgets?",
    ):
        add(EvalCategory.CLIENT_DECISION_SCENARIOS, q)

    # 10) TONE & PROFESSIONALISM (broker voice, not chatbot)
    for q in (
        "Be straight with me: is a Falcon 2000 a 'good buy'?",
        "I hate marketing. Give me the honest story on the Challenger 350.",
        "If you were spending your own money, would you buy a G280 or a Latitude?",
        "Talk me out of buying too much airplane.",
        "What are the red flags you notice in listings that most buyers miss?",
        "I’m skeptical: why is everyone obsessed with range?",
        "I want the best cabin experience for 6 pax. Don’t bury me in specs.",
        "I’m trying to impress clients — what matters visually and experientially?",
        "Is it normal to feel like brokers are hiding things? How do I protect myself?",
        "Give me a 30-second verdict: Phenom 300 vs CJ4.",
    ):
        add(EvalCategory.TONE_PROFESSIONALISM, q)

    return cases


def group_cases_by_category(cases: Sequence[EvalCase]) -> Dict[EvalCategory, List[EvalCase]]:
    out: Dict[EvalCategory, List[EvalCase]] = {}
    for c in cases:
        out.setdefault(c.category, []).append(c)
    return out

