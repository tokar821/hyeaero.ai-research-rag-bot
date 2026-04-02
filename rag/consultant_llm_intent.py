"""
LLM-based tool routing: ``aviation_consultant`` (DB / FAA / listings / RAG) vs ``general_chat`` (no tools).

Runs **after** rule-based :mod:`rag.conversation_guard` (which fast-paths obvious chit-chat).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

INTENT_AVIATION_CONSULTANT = "aviation_consultant"
INTENT_GENERAL_CHAT = "general_chat"


def aviation_intent_min_confidence() -> float:
    try:
        v = float((os.getenv("CONSULTANT_AVIATION_INTENT_MIN_CONFIDENCE") or "0.6").strip())
    except ValueError:
        return 0.6
    return max(0.0, min(1.0, v))


def llm_tool_routing_disabled() -> bool:
    return (os.getenv("CONSULTANT_LLM_TOOL_ROUTING_DISABLED") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _history_blob(history: Optional[List[Dict[str, str]]], max_messages: int = 10) -> str:
    if not history:
        return ""
    lines: List[str] = []
    for h in history[-max_messages:]:
        role = (h.get("role") or "").strip().lower()
        if role not in ("user", "assistant"):
            continue
        c = (h.get("content") or "").strip()
        if c:
            lines.append(f"{role}: {c}")
    return "\n".join(lines)


_ROUTING_SYSTEM = """You route messages for Hye Aero's **aircraft research & market consultant**.

Return JSON only:
{"intent": "aviation_consultant" | "general_chat", "confidence": <number 0.0–1.0>}

**aviation_consultant** — Use when the user needs (or clearly expects) aircraft-specific help:
U.S. or foreign registrations / N-numbers / tail / MSN or serial, who owns / operating vs registrant,
make/model specs, range, mission feasibility, runway, fuel, comparisons between aircraft,
market value, listings, for-sale, broker-style research, or aviation operations tied to a vehicle.

**general_chat** — Greetings already handled by rules may still appear; pure thanks, pleasantries,
simple math, unrelated trivia, generic life advice, coding homework, non-aviation topics with no
aircraft angle, or messages that only **vaguely** mention flying with no actionable aircraft question.

**confidence**: How sure you are this turn should trigger **full consultant tools** (database,
FAA, listings, RAG). Be **high** (0.85+) for clear ownership or tail lookups (e.g. "Who owns N123AB?"),
specs, range, market, or comparisons. Use **low** (<0.6) for borderline or off-topic.

If unsure between the two, prefer **general_chat** with confidence below 0.6."""


def classify_tool_routing_intent_llm(
    query: str,
    history: Optional[List[Dict[str, str]]],
    *,
    api_key: str,
    model: str,
    timeout: float = 14.0,
) -> Tuple[str, float]:
    """
    Returns ``(intent, confidence)`` with ``intent`` in ``{aviation_consultant, general_chat}``.

    On any failure, returns ``(aviation_consultant, 1.0)`` so outages do not strip tools from
    clear consultant questions.
    """
    if not (api_key or "").strip() or not (query or "").strip():
        return INTENT_AVIATION_CONSULTANT, 1.0
    try:
        import openai

        blob = _history_blob(history)
        user_msg = f"Conversation (oldest last):\n{blob}\n\n---\n\nLatest user message:\n{(query or '').strip()}"
        client = openai.OpenAI(api_key=api_key, timeout=max(6.0, min(30.0, timeout)))
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _ROUTING_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=96,
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        intent = (data.get("intent") or "").strip().lower()
        conf_raw = data.get("confidence")
        try:
            confidence = float(conf_raw) if conf_raw is not None else 0.75
        except (TypeError, ValueError):
            confidence = 0.75
        confidence = max(0.0, min(1.0, confidence))

        if intent in ("aviation", "aircraft", "consultant", "tools", "rag"):
            intent = INTENT_AVIATION_CONSULTANT
        if intent in ("general", "chat", "chitchat", "smalltalk"):
            intent = INTENT_GENERAL_CHAT

        if intent == INTENT_GENERAL_CHAT:
            return INTENT_GENERAL_CHAT, confidence
        if intent == INTENT_AVIATION_CONSULTANT:
            return INTENT_AVIATION_CONSULTANT, confidence
    except Exception as e:
        logger.warning("consultant LLM tool-routing classify failed: %s", e)
    return INTENT_AVIATION_CONSULTANT, 1.0


_GENERAL_CHAT_SYSTEM = """You are **HyeAero.AI** — the aviation intelligence assistant for **Hye Aero** (this chat turn uses no aircraft DB / FAA / RAG; keep the reply short).

Respond briefly and naturally:
- Introduce or sign as HyeAero.AI when greeting.
- Short math: give the answer.
- Thanks / small talk: warm, professional; offer aviation help in one clause.
- If they ask what Hye Aero is: explain confidently — aviation intelligence platform, brokerage support, data-driven market research; specs, ownership, missions, listings, comparison, buyer advisory.
Never say you don't know what Hye Aero is. Avoid "internal dataset" / "our database" / "records not found". Plain text, no markdown # headers."""


def generate_general_chat_reply_llm(
    query: str,
    history: Optional[List[Dict[str, str]]],
    *,
    api_key: str,
    model: str,
    timeout: float = 45.0,
    max_tokens: int = 400,
) -> str:
    """Single LLM turn for tool-free general conversation."""
    if not (api_key or "").strip():
        return (
            "Hi—I'm here when you want to dig into aircraft specs, ownership, or the market. "
            "What can I help with?"
        )
    try:
        import openai

        messages: List[Dict[str, str]] = [{"role": "system", "content": _GENERAL_CHAT_SYSTEM}]
        if history:
            for h in history[-10:]:
                role = (h.get("role") or "user").strip().lower()
                if role not in ("user", "assistant"):
                    role = "user"
                c = (h.get("content") or "").strip()
                if c:
                    messages.append({"role": role, "content": c})
        messages.append(
            {"role": "user", "content": f"Latest message:\n{(query or '').strip()}"}
        )
        client = openai.OpenAI(api_key=api_key, timeout=max(15.0, min(90.0, timeout)))
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.35,
        )
        out = (resp.choices[0].message.content or "").strip()
        return out or "How can I help you today?"
    except Exception as e:
        logger.warning("consultant general-chat LLM failed: %s", e)
        return (
            "Thanks for the message — I'm HyeAero.AI for Hye Aero. When you're ready, ask about aircraft, "
            "missions, or the market."
        )
