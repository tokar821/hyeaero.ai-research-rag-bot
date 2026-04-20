"""
Response depth for Ask Consultant — intent-aware answer shape (consultant vs retrieval-bot).

Used to append a short system-prompt suffix so the model adjusts length and structure.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from rag.consultant_fine_intent import ConsultantFineIntent


class ResponseDepthKind(str, Enum):
    CONFIRMATION = "confirmation"
    VISUAL_FOLLOWUP = "visual_followup"
    AIRCRAFT_LOOKUP = "aircraft_lookup"
    ADVISORY = "advisory"


def _has_prior_thread_substance(history: Optional[List[dict]]) -> bool:
    """True when the thread already has at least one non-empty message (context to resolve pronouns)."""
    if not history:
        return False
    for h in history[-12:]:
        if (h.get("content") or "").strip():
            return True
    return False


def _has_prior_turns(history: Optional[List[dict]], *, min_assistant: int = 1) -> bool:
    if not history:
        return False
    from rag.aviation_tail import normalize_history_role_for_tail_scan

    n = 0
    for h in history[-12:]:
        if normalize_history_role_for_tail_scan(h.get("role")) == "assistant" and (h.get("content") or "").strip():
            n += 1
    return n >= min_assistant


def _looks_like_deictic_visual_followup(query: str) -> bool:
    """
    Conversation Context Engine — short lines that mean *show the last aircraft we discussed*,
    including bare *show me*, *let me see*, *can I see that*, or *show me interior / cabin / cockpit*
    with no model in the latest line.
    """
    raw = (query or "").strip()
    if not raw or len(raw) > 160:
        return False
    low = raw.lower().rstrip(".!?")
    if len(low.split()) > 12:
        return False
    # Bare "show me" / "show me." — resolved from thread (requires aircraft context in classifier).
    if re.fullmatch(r"\s*show\s+me\s*", low, re.I):
        return True
    if re.search(
        r"\b(photo|photos|picture|pictures|image|images|gallery)\b",
        low,
        re.I,
    ) and len(low.split()) <= 10:
        return True
    if re.search(
        r"(?:^|\s)(?:so\s*,?\s+)?(?:can|could)\s+i\s+see\s+(?:it|them|that|this|one)\b",
        low,
        re.I,
    ):
        return True
    if re.search(r"\blet\s+me\s+see\b", low, re.I):
        return True
    if re.search(r"\b(?:show|showing)\s+me\s+(?:that|this|it|them|the\s+same)\b", low, re.I):
        return True
    if re.fullmatch(r"\s*show\s+me\s+(?:interior|exterior|cabin|cockpit)\s*", low, re.I):
        return True
    if re.search(
        r"\bshow\s+me\s+(?:interior|exterior|cabin|cockpit)\b(?!\s*(?:design|decor|ideas|furniture|home)\b)",
        low,
        re.I,
    ) and len(low.split()) <= 6:
        return True
    return False


def _looks_like_confirmation_query(query: str) -> bool:
    """Short follow-ups that validate prior content — not a request for a full re-brief."""
    raw = (query or "").strip()
    if not raw or len(raw) > 140:
        return False
    low = raw.lower().rstrip(".!…")
    # Very short tags
    if re.fullmatch(r"(right|correct|really|sure|ok|okay|yep|yeah|no|yes)\??", low):
        return True
    if re.fullmatch(r"(is that right|is that correct|am i right)\??", low):
        return True
    # So it's … / that's …
    if re.match(
        r"^(so|and)\s+(it'?s|that'?s|they'?re)\s+",
        low,
    ):
        return True
    if re.search(r"\b(so\s+it'?s|so\s+that'?s)\s+(listed|for sale|sold|still|accurate|true)\b", low):
        return True
    if re.fullmatch(r"so\s+listed\??", low):
        return True
    return False


def classify_response_depth(
    query: str,
    history: Optional[List[dict]],
    fine_intent: "ConsultantFineIntent",
) -> ResponseDepthKind:
    """
    Map fine intent + phrasing to a response-depth bucket for prompt conditioning.

    Confirmation overrides when the user is clearly validating a prior answer.
    """
    from rag.consultant_fine_intent import ConsultantFineIntent
    from rag.aviation_tail import find_strict_tail_candidates_in_text

    if _has_prior_turns(history) and _looks_like_confirmation_query(query):
        return ResponseDepthKind.CONFIRMATION

    # Visual / deictic follow-up: reuse last aircraft from thread — images + short line only.
    if _has_prior_thread_substance(history) and _looks_like_deictic_visual_followup(query):
        try:
            from rag.consultant_image_intent import thread_has_aircraft_context

            if thread_has_aircraft_context(query, history):
                return ResponseDepthKind.VISUAL_FOLLOWUP
        except Exception:
            pass

    fi = fine_intent

    if fi in (
        ConsultantFineIntent.AIRCRAFT_RECOMMENDATION,
        ConsultantFineIntent.AVIATION_MISSION,
        ConsultantFineIntent.AIRCRAFT_COMPARISON,
    ):
        return ResponseDepthKind.ADVISORY

    if fi == ConsultantFineIntent.MARKET_QUESTION:
        return ResponseDepthKind.ADVISORY

    if fi in (ConsultantFineIntent.OWNERSHIP_LOOKUP, ConsultantFineIntent.AIRCRAFT_SPECS):
        return ResponseDepthKind.AIRCRAFT_LOOKUP

    if fi == ConsultantFineIntent.GENERAL_QUESTION:
        if find_strict_tail_candidates_in_text(query or ""):
            return ResponseDepthKind.AIRCRAFT_LOOKUP
        return ResponseDepthKind.ADVISORY

    return ResponseDepthKind.ADVISORY


def response_depth_prompt_suffix(kind: ResponseDepthKind) -> str:
    """Append to the consultant system prompt (user-visible answers only)."""
    if kind == ResponseDepthKind.CONFIRMATION:
        return (
            "\n\n**Response depth — confirmation:** The user is validating or double-checking something from the thread.\n"
            "- Reply in **one to three short sentences** unless they explicitly ask for a full recap.\n"
            "- **Do not** repeat full aircraft identity, specs, ownership, or listing blocks already covered unless they ask for detail again.\n"
            "- Prefer: *Yes — …* / *Correct — …* / *That's consistent with …* grounded in context.\n"
            "- **Forbidden:** corporate filler such as *feel free to ask*, *reach out to brokers*, *I'm here to help*, or re-introducing HyeAero.AI capabilities.\n"
        )
    if kind == ResponseDepthKind.VISUAL_FOLLOWUP:
        return (
            "\n\n**Conversation Context Engine — visual mode (memory-aware):** Short asks like *show me*, "
            "*let me see*, *can I see that/it*, or *show me interior/cabin/cockpit* refer to the **last aircraft** "
            "already established in this thread.\n"
            "- **Reuse that identity** (make/model + tail from context). **Do not** ask which aircraft they mean.\n"
            "- **Do not** repeat full registry / listing / spec dumps, **Aircraft Record** field-by-field recap, or "
            "the same long prose as the prior turn.\n"
            "- **Required shape:** (1) **One short header line** — aircraft name and tail if known, plus what the "
            "gallery covers (e.g. *Gulfstream G400 (N888YG) — Cabin & exterior*). (2) **Images are the body** — "
            "the in-app gallery is the main answer. (3) **At most one optional** follow-up sentence for context "
            "(what to look for, verification caveat if needed).\n"
            "- **Bad:** Repeating full registrant, ownership blocks, or spec tables. **Good:** Short header + "
            "images + optional one-liner.\n"
        )
    if kind == ResponseDepthKind.AIRCRAFT_LOOKUP:
        return (
            "\n\n**Response depth — aircraft lookup:** Advisor brief, **not** an analyst dossier or data dump.\n"
            "- **Start with exactly 2–3 sentences:** identity, class/year, status or market snapshot if relevant—plain English. If the context includes a **specific record** for the tail/serial they asked about, **use it in those sentences**—no generic broker filler.\n"
            "- **Target ~120–180 words** on the first reply unless they asked for full detail.\n"
            "- Deeper **Key Specs**, **Ownership**, or **Market** blocks **only** if the user asks or one critical fact is missing.\n"
            "- **Do not** repeat facts already in-thread; **do not** duplicate the same fallback wording as a prior turn.\n"
            "- **Forbidden:** *bring a tail*, *bring a route*, database/internal-records language, phlydata, pinecone.\n"
        )
    # ADVISORY
    return (
        "\n\n**Response depth — advisory / consulting:** Trusted advisor, **not** a white paper or analyst report.\n"
        "- **Start concise:** short mission read, **2–4 example aircraft**, **one** follow-up question—**no** immediate long spec reports. **Target ~120–200 words** unless they asked for depth.\n"
        "- If basics are missing from the thread, ask **at least one** of **passengers**, **routes / city pairs**, **mission type**, **budget** (as fits) before a big list.\n"
        "- **Budget guide (illustrative):** under ~$5M → older light; $5M–$10M → light / entry midsize; $10M–$20M → midsize / super-midsize; $20M+ → large cabin. Do **not** recommend far **below** budget without a value frame; **above** only with stretch caveat.\n"
        "- **Do not** list serials, registrations, or full record blocks in recommendations unless the user asked for detailed aircraft data.\n"
        "- **Forbidden:** philosophical aviation filler (*aviation is about precision*, *like a well-planned flight*), forced pivots on non-aviation turns, internal system names.\n"
    )


def response_depth_label(kind: ResponseDepthKind) -> str:
    return kind.value
