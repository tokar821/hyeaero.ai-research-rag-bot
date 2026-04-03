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
    AIRCRAFT_LOOKUP = "aircraft_lookup"
    ADVISORY = "advisory"


def _has_prior_turns(history: Optional[List[dict]], *, min_assistant: int = 1) -> bool:
    if not history:
        return False
    n = 0
    for h in history[-12:]:
        if (h.get("role") or "").strip().lower() == "assistant" and (h.get("content") or "").strip():
            n += 1
    return n >= min_assistant


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
