"""
Consultant response modes (intent router before answer generation).

Routes each turn into a small set of modes so the system prompt can enforce shape,
verbosity, and visual-vs-text priority. Also exposes a JSON-serializable router result
for logging / product (`data_used`).
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Literal, Optional, TypedDict


class ConsultantResponseMode(str, Enum):
    """Primary modes aligned with product routing (plus pipeline-specific helpers)."""

    VISUAL_MODE = "visual_mode"
    ADVISORY_MODE = "advisory_mode"
    COMPARISON_MODE = "comparison_mode"
    DEAL_ANALYSIS_MODE = "deal_analysis_mode"
    CONVERSATION_MODE = "conversation_mode"
    # Tail-in-query wins over pure visual framing (registry + optional gallery).
    TAIL_SPECIFIC = "tail_specific"
    INVALID_SANITY = "invalid_sanity"


Verbosity = Literal["minimal", "short", "detailed"]


class ConsultantResponseRouterResult(TypedDict):
    """JSON-shaped router output (e.g. persist under data_used)."""

    mode: str
    reason: str
    visual_priority: bool
    verbosity: Verbosity


_VISUAL_HINT = re.compile(
    r"\b("
    r"show\s+me|show\s+us|let\s+me\s+see|can\s+i\s+see|want\s+to\s+see|"
    r"any\s+(photos?|pics?|pictures?|images?)|what\s+does\s+it\s+look\s+like|"
    r"walkaround|exterior|interior|interiors|in\s+the\s+cabin|cabin|cockpit|flight\s+deck|"
    r"bedroom\s+setup|bedroom|berth|divan|ambient\s+light|ambient\s+lighting|"
    r"luxury\s+feel|luxury\s+vibe|premium\s+aesthetic|visual\s+vibe|design\s+inspiration|"
    r"hotel\s+vibe|private\s+airline|modern\s+jet\s+cabin|huge\s+windows|white\s+interior|"
    r"something\s+nicer|nicer\s+looking|more\s+modern|more\s+premium|"
    r"picture\s+of|images?\s+of|photos?\s+of"
    r")\b",
    re.I,
)

_DEAL_HINT = re.compile(
    r"\b("
    r"good\s+deal|bad\s+deal|great\s+deal|overpriced|underpriced|fair\s+price|"
    r"worth\s+it|worth\s+buying|would\s+you\s+buy|"
    r"suspicious\s+pric|too\s+good\s+to\s+be\s+true|"
    r"market\s+value|resale\s+value|resale\b|"
    r"is\s+this\s+priced|pricing\s+seem|deal\s+or\s+no|pass\s+or\s+buy|"
    r"operating\s+costs?\b.*\b(worth|deal|buy)|"
    r"total\s+cost.*\b(worth|deal)"
    r")\b",
    re.I,
)
# Tight: "Should I buy **this** …" / listing — not open-ended "what should I buy".
_DEAL_SHOULD_I_BUY_THIS = re.compile(
    r"\bshould\s+i\s+buy\b(?!\s+(?:for|when|if\s+i))",
    re.I,
)

_COMPARISON_HINT = re.compile(
    r"\b("
    r"\bversus\b|compare|compared\s+to|difference\s+between|"
    r"which\s+feels\s+better|which\s+is\s+more\s+modern|which\s+wins|"
    r"\b(?:\w+\s+){1,4}vs\.?\s+(?:\w+\s+){1,4}\w+"  # "Falcon 2000 vs Challenger" style
    r"|better\s+for\s+me\b.*\b(or|vs)\b"
    r")\b",
    re.I,
)
# Standalone "vs" / " v. " between tokens (not "own vs charter").
_VS_BETWEEN_MODELS = re.compile(
    r"(?:citation|gulfstream|falcon|challenger|global|lear|bombardier|embraer|pilatus|king\s*air|"
    r"jet|aircraft|series|lx|ex|lxr|\d{3,4})\b[^.?]{0,60}\bvs\.?\b",
    re.I,
)

_STRATEGIC_HINT = re.compile(
    r"\b("
    r"own\s+vs|charter|fractional|lease|management\s+fee|fixed\s+cost|variable\s+cost|hourly\s+cost|"
    r"cost\s+of\s+ownership|operating\s+cost|maintenance|engine\s+program|hangar|crew|insurance|depreciation|"
    r"utilization|hours\s*/\s*year|roi|liquidity|total\s+cost"
    r")\b",
    re.I,
)

_CONVERSATION_HINT = re.compile(
    r"^\s*("
    r"hi\b|hello\b|hey\b|thanks!?|thank\s+you!?|good\s+morning|good\s+afternoon|"
    r"how\s+are\s+you|what'?s\s+up|sup\b|yo\b|lol\b|haha|nice\s+one|love\s+it"
    r")[\s!.,?]*$",
    re.I,
)


def route_consultant_response_mode(
    *,
    query: str,
    fine_intent: str,
    has_tail: bool,
    has_visual_intent: bool,
    suspicious_model_note: Optional[str],
) -> ConsultantResponseRouterResult:
    """
    Classify user intent for answer generation.

    Returns JSON-serializable fields: mode, reason, visual_priority, verbosity.
    """
    q = (query or "").strip()
    ql = q.lower()
    fi = (fine_intent or "").strip().lower()

    def _out(mode: ConsultantResponseMode, reason: str, visual_priority: bool, verbosity: Verbosity) -> ConsultantResponseRouterResult:
        return {
            "mode": mode.value,
            "reason": reason,
            "visual_priority": visual_priority,
            "verbosity": verbosity,
        }

    if (suspicious_model_note or "").strip():
        return _out(ConsultantResponseMode.INVALID_SANITY, "suspicious_or_nonexistent_aircraft_model", False, "short")

    if has_tail:
        vp = bool(has_visual_intent or _VISUAL_HINT.search(q))
        return _out(
            ConsultantResponseMode.TAIL_SPECIFIC,
            "registration_or_tail_in_query",
            vp,
            "short",
        )

    if has_visual_intent or _VISUAL_HINT.search(q):
        return _out(ConsultantResponseMode.VISUAL_MODE, "visual_or_interior_intent", True, "minimal")

    if len(q) < 120 and _CONVERSATION_HINT.search(q) and not re.search(
        r"\b(jet|aircraft|tail|n\d|citation|gulfstream|falcon|challenger|global|lear|bombardier)\b",
        ql,
        re.I,
    ):
        return _out(ConsultantResponseMode.CONVERSATION_MODE, "short_social_or_thanks_without_aviation_topic", False, "short")

    if _STRATEGIC_HINT.search(q):
        return _out(ConsultantResponseMode.ADVISORY_MODE, "ownership_costs_or_operating_economics", False, "short")

    if _DEAL_HINT.search(q) or (
        _DEAL_SHOULD_I_BUY_THIS.search(q) and not re.search(r"\b(what|which)\s+should\s+i\s+buy\b", ql)
    ):
        return _out(ConsultantResponseMode.DEAL_ANALYSIS_MODE, "pricing_deal_worth_or_market_value_intent", False, "short")

    if fi in ("aircraft_comparison",) or _COMPARISON_HINT.search(q) or _VS_BETWEEN_MODELS.search(q):
        return _out(ConsultantResponseMode.COMPARISON_MODE, "explicit_comparison_or_versus", False, "short")

    if fi in ("aircraft_recommendation", "aviation_mission"):
        return _out(ConsultantResponseMode.ADVISORY_MODE, f"fine_intent:{fi}", False, "short")

    if re.search(r"\b(full\s+brief|deep\s+dive|long\s+form|detailed\s+report)\b", ql):
        return _out(ConsultantResponseMode.ADVISORY_MODE, "user_requested_depth", False, "detailed")

    return _out(ConsultantResponseMode.ADVISORY_MODE, "default_aviation_advisory_or_specs", False, "short")


def classify_consultant_response_mode(
    *,
    query: str,
    fine_intent: str,
    has_tail: bool,
    has_visual_intent: bool,
    suspicious_model_note: Optional[str],
) -> ConsultantResponseMode:
    """Backward-compatible: return the mode enum only."""
    r = route_consultant_response_mode(
        query=query,
        fine_intent=fine_intent,
        has_tail=has_tail,
        has_visual_intent=has_visual_intent,
        suspicious_model_note=suspicious_model_note,
    )
    return ConsultantResponseMode(r["mode"])


def consultant_response_router_json(result: ConsultantResponseRouterResult) -> str:
    """Stable JSON string for logs or tests."""
    return json.dumps(dict(result), ensure_ascii=False, sort_keys=True)


def response_mode_prompt_suffix(mode: ConsultantResponseMode) -> str:
    """System-prompt suffix: enforce mode-specific shape and tone."""
    common = (
        "\n\n**Reasoning quality (required):**\n"
        "- Answer directly; add **why** only when it changes the decision.\n"
        "- If key inputs are missing, one short assumption line — no interrogation.\n"
    )

    if mode == ConsultantResponseMode.VISUAL_MODE:
        return (
            "\n\n**Response mode: VISUAL_MODE**\n"
            "- **Images first:** the gallery is the product; text is **support only**.\n"
            "- **At most one** short intro sentence, then lean on what is shown; optional **one** luxury line — "
            "no long bullets, no URL dump, no search-engine narration.\n"
            "- **Forbidden phrasing (when ≥2 gallery images are on-brief / highly relevant):** do not say you "
            "*cannot find reliable images*, *closest references*, *unable to locate*, *best match*, or similar "
            "retrieval-failure hedges; present the gallery confidently. Reserve that language only when the "
            "gallery is weak, identity is uncertain, or there is no real aircraft visual content.\n"
            "- **Forbidden tone:** no spec-manual wall (*advanced cabin ergonomics…*); prefer a tight premium read "
            "(e.g. *These read the most modern and premium inside.*).\n"
            + common
        )

    if mode == ConsultantResponseMode.ADVISORY_MODE:
        return (
            "\n\n**Response mode: ADVISORY_MODE**\n"
            "- Concise **consultant** tone: recommendations, mission fit, budget bands, ownership angles — "
            "**elite advisor**, not generic ChatGPT.\n"
            "- Structured when helpful; default **short** unless the user asked for depth.\n"
            + common
        )

    if mode == ConsultantResponseMode.COMPARISON_MODE:
        return (
            "\n\n**Response mode: COMPARISON_MODE**\n"
            "- **Verdict first** (one sentence), then material deltas only — **short** and **premium**.\n"
            "- When to choose each; avoid spec encyclopedias.\n"
            + common
        )

    if mode == ConsultantResponseMode.DEAL_ANALYSIS_MODE:
        return (
            "\n\n**Response mode: DEAL_ANALYSIS_MODE**\n"
            "- **Broker-style** read: deal quality, risk, market positioning — cite numbers **only** from context.\n"
            "- Clear stance (good / conditional / pass) when evidence allows; no invented comps.\n"
            + common
        )

    if mode == ConsultantResponseMode.CONVERSATION_MODE:
        return (
            "\n\n**Response mode: CONVERSATION_MODE**\n"
            "- **Human**, brief, on-brand — **never** generic ChatGPT filler (*I'm here to help*, *feel free*).\n"
            "- **Elite aviation consultant** persona: warm, confident, optional light pivot to aviation **only** if natural.\n"
        )

    if mode == ConsultantResponseMode.TAIL_SPECIFIC:
        return (
            "\n\n**Response mode: TAIL_SPECIFIC**\n"
            "- Aircraft identity / registry turn: lead with verified facts; **no** generic type marketing drift.\n"
            "- If a gallery is attached and the user asked to see the jet, keep text **minimal** beside the images.\n"
            + common
        )

    # INVALID_SANITY
    return (
        "\n\n**Response mode: INVALID_SANITY**\n"
        "- Reject the bogus model name clearly; suggest closest **real** variants — **short**, no invented specs or photos.\n"
        + common
    )


__all__ = [
    "ConsultantResponseMode",
    "ConsultantResponseRouterResult",
    "Verbosity",
    "classify_consultant_response_mode",
    "consultant_response_router_json",
    "response_mode_prompt_suffix",
    "route_consultant_response_mode",
]
