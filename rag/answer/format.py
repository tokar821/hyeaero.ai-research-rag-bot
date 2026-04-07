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
        "\n\n**Answer shape (intent: specs):** Advisor tone—**short lead** (what it means for their mission), then **only** the specs that matter; avoid walls of numbers. Prefer facts from structured context, labeled by source. **~120–200 words** default unless they asked for full specs."
    ),
    AviationIntent.MISSION_FEASIBILITY: (
        "\n\n**Answer shape (intent: mission / recommendation):** **Concise first:** short mission read, a few aircraft names that fit, **one** clarifying question—expand specs only if asked. "
        "If **pax**, **route/distance**, **longest leg**, **budget**, or **private vs charter** are missing on an **open-ended** buy ask, ask **1–2** questions **before** a model shortlist. "
        "No philosophical filler; no serial/N-number dumps unless user wants detail. "
        "Budget bands: under ~$5M light; $5M–$10M light/entry midsize; $10M–$20M midsize/super-mid; $20M+ large cabin. "
        "Use *typical operational performance for this class* for general guidance; **do not** imply registry/market **data** unless context actually provides those facts—never database or internal-records wording."
    ),
    AviationIntent.AIRCRAFT_COMPARISON: (
        "\n\n**Answer shape (intent: comparison):** Start with a one-line comparison title (e.g. Falcon 2000 vs Challenger 604). "
        "Then use these plain-text section headings (no markdown #): **Range** — **Passengers** — **Cruise speed** — "
        "**Cabin characteristics** — **Mission strengths**. Under each, bullet both aircraft. "
        "No promotional or charter booking links. Consultant tone only."
    ),
    AviationIntent.GENERAL_QUESTION: (
        "\n\n**Answer shape (intent: general):** Clear, direct answers. Where facts are inferred, use "
        "**based on typical operational performance for this aircraft or class…** — never "
        "\"internal dataset,\" \"our database,\" \"records not found,\" or \"data not available.\""
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
