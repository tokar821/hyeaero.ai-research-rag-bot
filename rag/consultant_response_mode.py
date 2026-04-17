"""
Consultant response modes (reasoning / structure router).

This module is intentionally deterministic: it routes queries into a small set of response
templates that improve clarity and decision usefulness without changing retrieval.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional


class ConsultantResponseMode(str, Enum):
    FACTUAL = "factual"
    VISUAL = "visual"
    TAIL_SPECIFIC = "tail_specific"
    COMPARISON = "comparison"
    MISSION_ADVISORY = "mission_advisory"
    STRATEGIC_OWNERSHIP = "strategic_ownership"
    INVALID_SANITY = "invalid_sanity"


_VISUAL_HINT = re.compile(
    r"\b(show\s+me|let\s+me\s+see|can\s+i\s+see|any\s+(photos|pics|pictures|images)|what\s+does\s+it\s+look\s+like|"
    r"walkaround|exterior|interior|cabin|cockpit|flight\s+deck)\b",
    re.I,
)

_STRATEGIC_HINT = re.compile(
    r"\b(own\s+vs|charter|fractional|lease|management\s+fee|fixed\s+cost|variable\s+cost|hourly\s+cost|"
    r"cost\s+of\s+ownership|operating\s+cost|maintenance|engine\s+program|hangar|crew|insurance|depreciation|"
    r"utilization|hours\s*/\s*year|roi|resale|liquidity|total\s+cost)\b",
    re.I,
)


def classify_consultant_response_mode(
    *,
    query: str,
    fine_intent: str,
    has_tail: bool,
    has_visual_intent: bool,
    suspicious_model_note: Optional[str],
) -> ConsultantResponseMode:
    """
    Deterministic router.

    Inputs are passed in from the main pipeline so this classifier does not need to know about
    retrieval systems or image engines.
    """
    q = (query or "").strip()
    ql = q.lower()

    if (suspicious_model_note or "").strip():
        return ConsultantResponseMode.INVALID_SANITY

    if has_tail:
        return ConsultantResponseMode.TAIL_SPECIFIC

    if has_visual_intent or _VISUAL_HINT.search(q):
        return ConsultantResponseMode.VISUAL

    if _STRATEGIC_HINT.search(q):
        return ConsultantResponseMode.STRATEGIC_OWNERSHIP

    if re.search(r"\bvs\.?\b|\bversus\b", ql):
        return ConsultantResponseMode.COMPARISON

    if fine_intent in ("aircraft_comparison",):
        return ConsultantResponseMode.COMPARISON
    if fine_intent in ("aircraft_recommendation", "aviation_mission"):
        return ConsultantResponseMode.MISSION_ADVISORY

    # Specs / factual by default for the remaining aviation intents.
    return ConsultantResponseMode.FACTUAL


def response_mode_prompt_suffix(mode: ConsultantResponseMode) -> str:
    """
    System-prompt suffix: enforce structured thinking templates + opinion + assumption awareness.

    Keep these templates plain-text and short; response depth control lives elsewhere.
    """
    common = (
        "\n\n**Reasoning quality (required):**\n"
        "- Always answer directly, then explain **why** (not just specs).\n"
        "- If key inputs are missing (route, pax, budget, constraints), state assumptions explicitly in one short line.\n"
        "- Never list aircraft without a 1-line fit rationale per model.\n"
    )

    if mode == ConsultantResponseMode.FACTUAL:
        return (
            "\n\n**Response mode: FACTUAL**\n"
            "- Direct answer first.\n"
            "- Optional context: **1 sentence max**.\n"
            + common
        )

    if mode == ConsultantResponseMode.VISUAL:
        return (
            "\n\n**Response mode: VISUAL**\n"
            "- Acknowledge visuals first (gallery/pictures) when present.\n"
            "- Then a short, helpful description of what to look for (exterior vs cabin vs cockpit cues).\n"
            + common
        )

    if mode == ConsultantResponseMode.TAIL_SPECIFIC:
        return (
            "\n\n**Response mode: TAIL-SPECIFIC**\n"
            "- Treat this as an aircraft-specific lookup. Do **not** drift into generic type marketing.\n"
            "- If evidence is incomplete, say **no verified data** rather than guessing.\n"
            + common
        )

    if mode == ConsultantResponseMode.COMPARISON:
        return (
            "\n\n**Response mode: COMPARISON (template required)**\n"
            "- Verdict first (1 sentence).\n"
            "- Key differences (bullets).\n"
            "- When to choose each.\n"
            "- **Bottom Line:** 1–2 sentences, no hedging.\n"
            "\n\n**Consultant Insight:** Add 1–2 sentences of real-world buyer/operator reasoning.\n"
            + common
        )

    if mode == ConsultantResponseMode.MISSION_ADVISORY:
        return (
            "\n\n**Response mode: MISSION ADVISORY (template required)**\n"
            "- Restate mission (distance/route, pax, constraints).\n"
            "- Define required capability (range + margin).\n"
            "- **Recommendations required:** recommend **2–4** aircraft.\n"
            "- For each recommendation: **why it fits** + one clear tradeoff.\n"
            "- Explicitly eliminate at least 1–2 unsuitable options (so the client sees the boundary).\n"
            "- **Bottom Line:** 1–2 sentences, no hedging.\n"
            "\n\n**Consultant Insight:** Add 1–2 sentences of real-world buyer reasoning.\n"
            + common
        )

    if mode == ConsultantResponseMode.STRATEGIC_OWNERSHIP:
        return (
            "\n\n**Response mode: STRATEGIC / OWNERSHIP (template required)**\n"
            "- Clear stance (own vs charter vs fractional).\n"
            "- Cost logic (fixed vs variable).\n"
            "- Thresholds (use rules of thumb when exact numbers aren't in context).\n"
            "- Risks / hidden costs.\n"
            "- **Bottom Line:** 1–2 sentences, no hedging.\n"
            "\n\n**Consultant Insight:** Add 1–2 sentences of real-world ownership reasoning.\n"
            + common
        )

    # INVALID / SANITY
    return (
        "\n\n**Response mode: INVALID / SANITY CHECK (template required)**\n"
        "- Reject clearly.\n"
        "- Suggest closest real models.\n"
        "- Do not invent specs, listings, or 'verified photos' for the fake model.\n"
        + common
    )

