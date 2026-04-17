"""
Optional "gold" answers for qualitative comparison.

Keep these short and structured; they are not used for automatic grading beyond light checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class GoldAnswer:
    query: str
    ideal: str
    tags: Optional[List[str]] = None


def gold_answers() -> List[GoldAnswer]:
    return [
        GoldAnswer(
            query="Falcon 2000 vs Challenger 350 — which is actually better? Don’t be vague.",
            ideal=(
                "Verdict: If you prioritize cabin volume and a more “big-jet” feel, I’d lean Falcon 2000; if you want a "
                "newer-generation super-midsize with strong efficiency and broad mission flexibility, Challenger 350 is hard to beat.\n\n"
                "Key differences:\n"
                "- Cabin feel: Falcon 2000 generally feels wider/more spacious; Challenger 350 feels modern and efficient.\n"
                "- Mission fit: both cover most U.S. missions; pick based on typical stage length, payload, and cabin priorities.\n"
                "- Operating reality: the cheaper airplane to buy is rarely the cheaper airplane to own—condition and programs matter.\n\n"
                "When to choose each:\n"
                "- Choose Falcon 2000 if the cabin experience is the point.\n"
                "- Choose Challenger 350 if you want a very balanced modern platform.\n\n"
                "Consultant Insight: Most buyers who fly 2–3 hour legs regret paying for max range; they’re happier when the airplane is easy to dispatch and feels great every week."
            ),
            tags=["comparison", "consultant_insight"],
        ),
        GoldAnswer(
            query="I fly 6 people NYC to LA twice a week — what should I buy?",
            ideal=(
                "Assuming 6 passengers with bags, year-round reliability, and a preference for nonstop: you’re shopping in the super‑midsize / large‑cabin edge.\n\n"
                "Required capability: NYC–LA is ~2,150 nm; I’d size for ~2,600–2,900 nm practical margin.\n\n"
                "Shortlist (why it fits):\n"
                "- Challenger 350: balanced nonstop capability + strong dispatch reputation.\n"
                "- Praetor 600: excellent range margin and modern cabin for the segment.\n"
                "- G280: great speed/cabin for transcon if you value efficiency.\n\n"
                "Tradeoffs: stepping up to a true large cabin improves cabin experience but adds ownership complexity and cost.\n\n"
                "Consultant Insight: Two transcons a week often makes buyers over-index on range; the real win is a platform you can keep in a simple, predictable maintenance/program posture."
            ),
            tags=["advisory", "assumptions", "consultant_insight"],
        ),
        GoldAnswer(
            query="Tell me about the Falcon 9000",
            ideal=(
                "There isn’t a Dassault “Falcon 9000” production model. Most people mean one of these:\n"
                "- Falcon 900 (900EX/EASy variants)\n"
                "- Falcon 2000 family\n"
                "- Falcon 7X / 8X / 6X / 10X\n\n"
                "If you tell me your typical passengers and longest leg, I’ll point you to the closest real fit.\n\n"
                "Consultant Insight: When a model name is off by a digit, listings and photos get mixed fast—best move is to lock the exact variant before comparing price or range."
            ),
            tags=["invalid", "sanity", "consultant_insight"],
        ),
        GoldAnswer(
            query="Do you have photos of N000ZZZ?",
            ideal=(
                "No verified images found for this aircraft. If you can share the model or operator, I can pull representative photos of the type, but I won’t guess the tail match.\n\n"
                "Consultant Insight: Tail-only photo hunting is noisy because CDNs often strip the registration—multi-source agreement is what makes a match trustworthy."
            ),
            tags=["tail", "honesty", "consultant_insight"],
        ),
    ]


def gold_index() -> Dict[str, GoldAnswer]:
    return {g.query: g for g in gold_answers()}

