"""
Consultant response modes (reasoning / structure router).

Deterministic routing into legacy :class:`ConsultantResponseMode` (pipeline + tests) plus a
structured **HyeAero response router** payload (mode, reason, visual_priority, verbosity)
for prompts and ``data_used``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Literal, Optional

VerbosityLevel = Literal["minimal", "short", "detailed"]


class HyeAeroRouterMode(str, Enum):
    """External / product router labels (JSON ``mode`` field)."""

    VISUAL_MODE = "VISUAL_MODE"
    ADVISORY_MODE = "ADVISORY_MODE"
    COMPARISON_MODE = "COMPARISON_MODE"
    DEAL_ANALYSIS_MODE = "DEAL_ANALYSIS_MODE"
    CONVERSATION_MODE = "CONVERSATION_MODE"


class ConsultantResponseMode(str, Enum):
    FACTUAL = "factual"
    VISUAL = "visual"
    TAIL_SPECIFIC = "tail_specific"
    COMPARISON = "comparison"
    MISSION_ADVISORY = "mission_advisory"
    STRATEGIC_OWNERSHIP = "strategic_ownership"
    DEAL_ANALYSIS = "deal_analysis"
    CONVERSATION = "conversation"
    INVALID_SANITY = "invalid_sanity"


@dataclass(frozen=True)
class ResponseModeRouterResult:
    mode: HyeAeroRouterMode
    reason: str
    visual_priority: bool
    verbosity: VerbosityLevel

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "reason": self.reason,
            "visual_priority": self.visual_priority,
            "verbosity": self.verbosity,
        }


# --- Signals (router layer, broader than legacy _VISUAL_HINT) ---

_VISUAL_ROUTER = re.compile(
    r"\b("
    r"show\s+me|let\s+me\s+see|can\s+i\s+see|\bsee\b(?!\s+(?:you|that\s+it))|showing|viewing|"
    r"any\s+(?:photos?|pics?|pictures?|images?)|what\s+does\s+it\s+look\s+like|walkaround|"
    r"exterior|interior|insides?|cabin|cockpit|flight\s+deck|bedroom\s+setup|bed\s+room|"
    r"ambient\s+lighting|luxury\s+feel|design\s+inspiration|visual\s+vibe|premium\s+aesthetic|"
    r"hotel\s+vibe|private\s+airline\s+experience|modern\s+jet\s+cabin|huge\s+windows|"
    r"white\s+interior|something\s+nicer|nicer\s+one|more\s+modern\s+inside"
    r")\b",
    re.I,
)

_DEAL_ANALYSIS_ROUTER = re.compile(
    r"\b("
    r"good\s+deal|bad\s+deal|worth\s+it|overpriced|underpriced|fair\s+price|"
    r"suspicious\s+pric|market\s+value|resale|residual|"
    r"asking\s+too\s+much|too\s+much\s+for|rip[\s-]?off|"
    r"operating\s+cost|cost\s+per\s+(?:hour|nm)|direct\s+operating|"
    r"ocr\b|cpi\b|should\s+i\s+(?:pay|buy)\s+at|is\s+this\s+(?:priced|fair)|"
    r"deal\s+or\s+pass|pass\s+on\s+this"
    r")\b",
    re.I,
)

_COMPARISON_ROUTER = re.compile(
    r"\b("
    r"vs\.?|versus|compare|comparison|difference\b|stack\s+up|side\s+by\s+side|"
    r"which\s+(?:feels|is)\s+better|which\s+is\s+more\s+modern|which\s+would\s+you|"
    r"better\s+than|or\s+the\s+other"
    r")\b",
    re.I,
)

_CONVERSATION_ROUTER = re.compile(
    r"^\s*("
    r"hi\b|hello\b|hey\b|yo\b|sup\b|good\s+(?:morning|afternoon|evening)|"
    r"thanks?\b|thank\s+you|ty\b|thx\b|cheers\b|"
    r"(?:how\s+are\s+you|what'?s\s+up|whats\s+up)"
    r")[!.\s?]*\s*$",
    re.I,
)

_ADVISORY_ROUTER = re.compile(
    r"\b("
    r"recommend|should\s+i\s+buy|what\s+should\s+i\s+buy|best\s+jet|best\s+aircraft|better\s+option|"
    r"budget|under\s+\$|mission\s+fit|luxury\s+rank|value\s+retention|"
    r"what\s+(?:should|would)\s+i\s+(?:get|pick|choose|buy)|"
    r"which\s+(?:jet|aircraft|plane)\s+(?:for|should|would)|"
    r"acquisition|shortlist"
    r")\b",
    re.I,
)

_STRATEGIC_HINT = re.compile(
    r"\b("
    r"own\s+vs|charter|fractional|lease|management\s+fee|fixed\s+cost|variable\s+cost|hourly\s+cost|"
    r"cost\s+of\s+ownership|maintenance|engine\s+program|hangar|crew|insurance|depreciation|"
    r"utilization|hours\s*/\s*year|roi|liquidity|total\s+cost"
    r")\b",
    re.I,
)

_LEGACY_VISUAL_HINT = re.compile(
    r"\b(show\s+me|let\s+me\s+see|can\s+i\s+see|any\s+(?:photos|pics|pictures|images)|what\s+does\s+it\s+look\s+like|"
    r"walkaround|exterior|interior|cabin|cockpit|flight\s+deck)\b",
    re.I,
)


def classify_hye_aero_response_router(
    *,
    query: str,
    fine_intent: str,
    has_tail: bool,
    has_visual_intent: bool,
    suspicious_model_note: Optional[str],
) -> ResponseModeRouterResult:
    """
    Classify user intent into one of five product modes + metadata.

    Invalid / fictional model strings short-circuit to ADVISORY with a sanity reason (legacy
    pipeline still uses :class:`ConsultantResponseMode.INVALID_SANITY` for the prompt tier).
    """
    q = (query or "").strip()
    ql = q.lower()
    fi = (fine_intent or "").strip().lower()

    if (suspicious_model_note or "").strip():
        return ResponseModeRouterResult(
            mode=HyeAeroRouterMode.ADVISORY_MODE,
            reason="invalid_or_unsupported_model",
            visual_priority=False,
            verbosity="short",
        )

    visual_hit = bool(
        has_visual_intent or _VISUAL_ROUTER.search(q) or _LEGACY_VISUAL_HINT.search(q)
    )
    if visual_hit:
        r = "gallery_or_visual_followup"
        if has_tail:
            r = "tail_specific_visual"
        return ResponseModeRouterResult(
            mode=HyeAeroRouterMode.VISUAL_MODE,
            reason=r,
            visual_priority=True,
            verbosity="minimal",
        )

    # Ownership economics / own-vs-charter — before DEAL (fine_intent can be market_question) and before `\bvs\b` comparisons.
    if _STRATEGIC_HINT.search(q) or re.search(
        r"\b(?:own|buy|lease|charter|fractional)\s+vs\s+(?:own|buy|lease|charter|fractional)\b",
        q,
        re.I,
    ):
        return ResponseModeRouterResult(
            mode=HyeAeroRouterMode.ADVISORY_MODE,
            reason="ownership_economics_strategic",
            visual_priority=False,
            verbosity="short",
        )

    if _DEAL_ANALYSIS_ROUTER.search(q) or fi == "aircraft_price_lookup":
        return ResponseModeRouterResult(
            mode=HyeAeroRouterMode.DEAL_ANALYSIS_MODE,
            reason="pricing_deal_or_economics",
            visual_priority=False,
            verbosity="short",
        )

    if _COMPARISON_ROUTER.search(q) or fi == "aircraft_comparison":
        return ResponseModeRouterResult(
            mode=HyeAeroRouterMode.COMPARISON_MODE,
            reason="explicit_compare_or_contrast",
            visual_priority=False,
            verbosity="short",
        )

    if len(q) < 96 and _CONVERSATION_ROUTER.match(q) and not re.search(
        r"\b(N\d{1,5}[A-Z]{0,2}|tail|citation|gulfstream|falcon|challenger|learjet|jet|aircraft)\b",
        ql,
        re.I,
    ):
        return ResponseModeRouterResult(
            mode=HyeAeroRouterMode.CONVERSATION_MODE,
            reason="brief_social_or_acknowledgment",
            visual_priority=False,
            verbosity="minimal",
        )

    if _ADVISORY_ROUTER.search(q) or fi in ("aircraft_recommendation", "aviation_mission"):
        vb: VerbosityLevel = "short"
        if len(q) > 180 or "report" in ql or "deep" in ql or "full brief" in ql:
            vb = "detailed"
        return ResponseModeRouterResult(
            mode=HyeAeroRouterMode.ADVISORY_MODE,
            reason="recommendation_mission_budget_or_acquisition",
            visual_priority=False,
            verbosity=vb,
        )

    # Default: consultant advisory tone for remaining aviation queries (specs, lookups phrased generally).
    return ResponseModeRouterResult(
        mode=HyeAeroRouterMode.ADVISORY_MODE,
        reason="general_consultant",
        visual_priority=False,
        verbosity="short",
    )


def _router_to_legacy_mode(
    r: ResponseModeRouterResult,
    *,
    fine_intent: str,
    query: str,
) -> ConsultantResponseMode:
    fi = (fine_intent or "").strip().lower()
    q = (query or "").strip()
    ql = q.lower()

    if r.mode == HyeAeroRouterMode.VISUAL_MODE:
        return ConsultantResponseMode.VISUAL
    if r.mode == HyeAeroRouterMode.COMPARISON_MODE:
        return ConsultantResponseMode.COMPARISON
    if r.mode == HyeAeroRouterMode.DEAL_ANALYSIS_MODE:
        return ConsultantResponseMode.DEAL_ANALYSIS
    if r.mode == HyeAeroRouterMode.CONVERSATION_MODE:
        return ConsultantResponseMode.CONVERSATION

    if r.reason == "invalid_or_unsupported_model":
        return ConsultantResponseMode.INVALID_SANITY
    if r.reason == "ownership_economics_strategic":
        return ConsultantResponseMode.STRATEGIC_OWNERSHIP

    # ADVISORY_MODE → legacy granularity
    if _STRATEGIC_HINT.search(q) and not _DEAL_ANALYSIS_ROUTER.search(q):
        return ConsultantResponseMode.STRATEGIC_OWNERSHIP
    if fi in ("aircraft_recommendation", "aviation_mission") or _ADVISORY_ROUTER.search(q):
        return ConsultantResponseMode.MISSION_ADVISORY
    if fi in (
        "aircraft_specs",
        "ownership_lookup",
        "tail_number_lookup",
        "serial_number_lookup",
    ) or re.search(r"\b(range|ceiling|mtow|seats|kts|mach)\b", ql):
        return ConsultantResponseMode.FACTUAL
    if " nm" in ql or re.search(r"\b\d{3,5}\s*nm\b", ql):
        return ConsultantResponseMode.FACTUAL
    return ConsultantResponseMode.MISSION_ADVISORY


def classify_consultant_response_mode(
    *,
    query: str,
    fine_intent: str,
    has_tail: bool,
    has_visual_intent: bool,
    suspicious_model_note: Optional[str],
) -> ConsultantResponseMode:
    """
    Deterministic router for the RAG pipeline (backward-compatible enum).

    ``has_tail`` forces **tail_specific** so registry-forward queries stay structured.
    """
    if (suspicious_model_note or "").strip():
        return ConsultantResponseMode.INVALID_SANITY

    if has_tail:
        return ConsultantResponseMode.TAIL_SPECIFIC

    r = classify_hye_aero_response_router(
        query=query,
        fine_intent=fine_intent,
        has_tail=False,
        has_visual_intent=has_visual_intent,
        suspicious_model_note=None,
    )
    return _router_to_legacy_mode(r, fine_intent=fine_intent, query=query)


def response_mode_prompt_suffix(
    mode: ConsultantResponseMode,
    *,
    router: Optional[ResponseModeRouterResult] = None,
) -> str:
    """
    System-prompt suffix: structured templates + HyeAero voice.

    When ``router`` is provided, **VISUAL_MODE** rules override the legacy VISUAL blurb.
    """
    common = (
        "\n\n**Reasoning quality (required):**\n"
        "- Always answer directly, then explain **why** (not just specs).\n"
        "- If key inputs are missing (route, pax, budget, constraints), state assumptions explicitly in one short line.\n"
        "- Never list aircraft without a 1-line fit rationale per model.\n"
    )

    if router and router.mode == HyeAeroRouterMode.VISUAL_MODE:
        return (
            "\n\n**Response mode: VISUAL_MODE (HyeAero router)**\n"
            "- **Prioritize** the in-app gallery over prose; images are the main deliverable.\n"
            "- **At most one** short intro sentence, then let the gallery carry the turn.\n"
            "- Optional: **one** line of luxury / design tone (e.g. what feels modern or premium) — no spec dumps.\n"
            "- **Never** say (or close variants): *I cannot find reliable images*, *closest references*, *best match*.\n"
            "- **No URLs**, no source dumps, no long bullets, no search-engine narration.\n"
            "- **Forbidden:** long *aircraft features advanced cabin ergonomics…* marketing filler.\n"
            "**Good:** *These read as the most modern, premium cabins in this set.*\n"
            "**Bad:** multi-paragraph ergonomic / OEM feature essay.\n"
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
            "- **Minimal text:** at most one lead-in sentence + optional one luxury/design line; do not lecture.\n"
            "- **No** long bullet lists for pure visual asks.\n"
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
            "\n\n**Response mode: COMPARISON (premium, concise)**\n"
            "- Verdict first (**1 sentence**).\n"
            "- **Material differences only** — short bullets or tight sentences.\n"
            "- When to choose each — **1 line each**.\n"
            "- **Bottom Line:** 1–2 sentences.\n"
            "\n\n**Consultant Insight:** 1 sentence of real-world buyer/operator reasoning.\n"
            + common
        )

    if mode == ConsultantResponseMode.MISSION_ADVISORY:
        return (
            "\n\n**Response mode: MISSION ADVISORY (template required)**\n"
            "- Restate mission when needed (distance/route, pax, constraints).\n"
            "- **Recommendations:** **2–4** aircraft with **why it fits** + one tradeoff each.\n"
            "- Eliminate at least **1** bad-fit example when useful.\n"
            "- **Bottom Line:** 1–2 sentences.\n"
            "\n\n**Consultant Insight:** 1–2 sentences of buyer reasoning.\n"
            + common
        )

    if mode == ConsultantResponseMode.STRATEGIC_OWNERSHIP:
        return (
            "\n\n**Response mode: STRATEGIC / OWNERSHIP**\n"
            "- Clear stance (own vs charter vs fractional) when relevant.\n"
            "- Cost logic (fixed vs variable) — rules of thumb if context lacks numbers.\n"
            "- Risks / hidden costs.\n"
            "- **Bottom Line:** 1–2 sentences.\n"
            "\n\n**Consultant Insight:** 1–2 sentences of ownership realism.\n"
            + common
        )

    if mode == ConsultantResponseMode.DEAL_ANALYSIS:
        return (
            "\n\n**Response mode: DEAL ANALYSIS**\n"
            "- Broker-style: **deal quality**, **pricing vs context**, **resale / liquidity** framing when relevant.\n"
            "- **No** fabricated dollars — cite only what the brief supports; otherwise ranges/unknowns are explicit.\n"
            "- Red flags when warranted; **Bottom line** verdict (\"good deal\" / \"rich\" / \"pass\") in plain words.\n"
            + common
        )

    if mode == ConsultantResponseMode.CONVERSATION:
        return (
            "\n\n**Response mode: CONVERSATION**\n"
            "- **Elite aviation consultant** — warm, human, **not** generic ChatGPT filler.\n"
            "- **Minimal** length; no capability brochure; no *I'm here to help* patterns.\n"
        )

    # INVALID / SANITY
    return (
        "\n\n**Response mode: INVALID / SANITY CHECK (template required)**\n"
        "- Reject clearly.\n"
        "- Suggest closest real models.\n"
        "- Do not invent specs, listings, or 'verified photos' for the fake model.\n"
        + common
    )
