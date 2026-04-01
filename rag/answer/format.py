"""LLM answer shaping (system-prompt suffixes) for consultant turns."""

from __future__ import annotations

from typing import Optional

from rag.intent.schemas import AviationIntent, ConsultantIntent

_AVIATION_STYLE = {
    AviationIntent.REGISTRATION_LOOKUP: (
        "\n\n**Answer shape (intent: registration):** Prioritize U.S. legal registrant lines when an FAA block is present; "
        "otherwise synthesize labeled web/vector evidence. Do not imply marketplace listings are PhlyData."
    ),
    AviationIntent.SERIAL_LOOKUP: (
        "\n\n**Answer shape (intent: serial):** Anchor on the cited manufacturer serial / MSN; separate identity facts from listing copy."
    ),
    AviationIntent.OPERATOR_LOOKUP: (
        "\n\n**Answer shape (intent: operator):** Distinguish **operating** carrier / manager from **U.S. FAA legal registrant** when both appear; label each source."
    ),
    AviationIntent.AIRCRAFT_FOR_SALE: (
        "\n\n**Answer shape (intent: for sale):** Lead on PhlyData internal snapshot when present; frame marketplace rows as **listing-ingest** with availability caveats."
    ),
    AviationIntent.MARKET_PRICE: (
        "\n\n**Answer shape (intent: market price):** Cite numbers from context; separate Phly internal ask from listing ask and web snippets."
    ),
    AviationIntent.AIRCRAFT_SPECS: (
        "\n\n**Answer shape (intent: specs):** Lead with operational relevance; prefer OEM-style performance facts from structured context, labeled by source — avoid spec lists with no mission tie-in."
    ),
    AviationIntent.MISSION_FEASIBILITY: (
        "\n\n**Answer shape (intent: mission):** Natural prose — range, reserves, winds, tech stops, and feasibility; "
        "tie route logic to evidence; say what is unknown or airport-specific. No rigid section headers."
    ),
    AviationIntent.AIRCRAFT_COMPARISON: (
        "\n\n**Answer shape (intent: comparison):** Compare range (state assumptions), cabin, passengers, mission fit, "
        "and operational differences in flowing prose — not a fixed rubric or titled blocks."
    ),
    AviationIntent.GENERAL_QUESTION: (
        "\n\n**Answer shape (intent: general):** Clear definitions; do not invent registry or internal DB facts."
    ),
}

_COARSE_FALLBACK = {
    ConsultantIntent.REGISTRATION_LOOKUP: _AVIATION_STYLE[AviationIntent.REGISTRATION_LOOKUP],
    ConsultantIntent.MARKET_PRICING: _AVIATION_STYLE[AviationIntent.MARKET_PRICE],
    ConsultantIntent.TECHNICAL_SPEC: _AVIATION_STYLE[AviationIntent.AIRCRAFT_SPECS],
    ConsultantIntent.AIRCRAFT_IDENTITY: (
        "\n\n**Answer shape (intent: identity):** Tie claims to the cited tail/serial; state conflicts between sources explicitly."
    ),
    ConsultantIntent.GENERAL_AVIATION: _AVIATION_STYLE[AviationIntent.GENERAL_QUESTION],
    ConsultantIntent.UNKNOWN: "",
}


def consultant_answer_style_suffix(
    primary: ConsultantIntent,
    aviation: Optional[AviationIntent] = None,
) -> str:
    """Prefer fine aviation intent copy when provided; else coarse consultant bucket."""
    if aviation is not None and aviation in _AVIATION_STYLE:
        return _AVIATION_STYLE[aviation]
    return _COARSE_FALLBACK.get(primary, "")
