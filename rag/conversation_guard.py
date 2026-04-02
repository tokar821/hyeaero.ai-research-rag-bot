"""
Conversation guard — runs **before** the heavy consultant tool pipeline in
:func:`rag.consultant_retrieval.run_consultant_retrieval_bundle`.

- **Greetings / small talk / identity** — short template replies.
- **Non-aviation general** (math, trivia, geography, jokes) — brief correct answer, then an
  aviation-specialization reminder (templates or LLM). Does not claim dataset or system limitations.
- **Aviation** — returns ``aviation_query`` so SQL / FAA / listings / Pinecone can run.

Optional: ``CONSULTANT_CONVERSATION_GUARD_LLM=1`` refines classification via LLM.
``CONSULTANT_NON_AVIATION_LLM=0`` disables the extra LLM step for general-knowledge answers (defaults on).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)

# Second paragraph after answering non-aviation questions — professional, no "database" / limitation language.
_AVIATION_FOCUS_REMINDERS = [
    (
        "I'm HyeAero.AI — focused on aircraft intelligence and aviation consulting for Hye Aero.\n"
        "If you need help with aircraft missions, specifications, ownership research, or market insights, I'm here to help."
    ),
    (
        "I'm HyeAero.AI — your aviation intelligence assistant for Hye Aero.\n"
        "Feel free to ask about aircraft missions, specifications, or market insights anytime."
    ),
    (
        "I'm HyeAero.AI for Hye Aero — broker-grade support on missions, specs, registry intelligence, and the jet market.\n"
        "Whenever you're ready to talk aircraft, I'm here."
    ),
]


def _pick_aviation_reminder(query: str) -> str:
    h = int(hashlib.md5((query or "").strip().lower().encode(), usedforsecurity=False).hexdigest(), 16)
    return _AVIATION_FOCUS_REMINDERS[h % len(_AVIATION_FOCUS_REMINDERS)]


_JOKE_REQUEST_RE = re.compile(
    r"\b("
    r"tell\s+me\s+(a\s+)?joke|"
    r"make\s+me\s+laugh|"
    r"got\s+(a\s+)?joke|"
    r"any\s+jokes?|"
    r"dad\s+joke|"
    r"funny\s+joke"
    r")\b",
    re.I,
)

_AVIATION_JOKES = [
    (
        "Why don't pilots ever get lost?\n\n"
        "Because they always follow their heading."
    ),
    (
        "Why did the student pilot bring a pencil to the flight deck?\n\n"
        "In case they needed to draw a holding pattern."
    ),
    (
        "What's the difference between a jet engine and a pilot?\n\n"
        "The engine stops whining when you shut it down."
    ),
    (
        "How do you know an aircraft mechanic had a good weekend?\n\n"
        "They torque about it all week."
    ),
]


def _try_joke_reply(raw: str) -> Optional[str]:
    if not _JOKE_REQUEST_RE.search(raw or ""):
        return None
    if query_has_aviation_signals(raw):
        return None
    h = int(hashlib.md5((raw or "").strip().lower().encode(), usedforsecurity=False).hexdigest(), 16)
    joke = _AVIATION_JOKES[h % len(_AVIATION_JOKES)]
    return f"{joke}\n\n{_pick_aviation_reminder(raw)}"


class ConversationMessageType(str, Enum):
    GREETING = "greeting"
    SMALL_TALK = "small_talk"
    IDENTITY_QUESTION = "identity_question"
    # Math, geography, trivia, jokes — brief answer, then aviation focus (no tool pipeline).
    NON_AVIATION_GENERAL = "non_aviation_general"
    AVIATION_QUERY = "aviation_query"


@dataclass
class ConversationGuardResult:
    message_type: ConversationMessageType
    reply: Optional[str]
    """Set when message_type != AVIATION_QUERY (caller returns without tools)."""


def _env_truthy(key: str) -> bool:
    return (os.getenv(key) or "").strip().lower() in ("1", "true", "yes")


# --- Aircraft / consultant signals (if present, not "conversational only") ---

_AIRCRAFT_RE = re.compile(
    r"(\b"
    r"aircraft|\bjet\b|\bplanes?\b|\baviation|\bFAA\b|registry|\btail\b|"
    r"\bmsn\b|serial|n-?number|\bn\d[\w\-]{1,6}\b|"
    r"citation|challenger|gulfstream|falcon|learjet|global\s*\d|phenom|praetor|"
    r"pilatus|king\s*air|hawker|bombardier|embraer|beechcraft|"
    r"nautical|\bnm\b|\b(range|payload|mtow)\b|"
    r"\bbroker\b|\bcharter\b|\bleasing\b|\bicao\b|\beasa\b|\bstall\b|\blift\b|"
    r"\d{2,5}-\d{2,6}"
    r")",
    re.I,
)

_TRIVIA_NON_AVIATION_HINT = re.compile(
    r"\b("
    r"capital of|largest city in|population of|continent|"
    r"president of|prime minister of|currency of|time zone"
    r")\b",
    re.I,
)

_WH_GENERAL = re.compile(
    r"^(what|who|where|when|why|how|which|name|define|explain|list|is|are)\b",
    re.I,
)


def query_has_aviation_signals(text: str) -> bool:
    """True if message likely needs aircraft DB / RAG (not pure chit-chat)."""
    if not (text or "").strip():
        return False
    return bool(_AIRCRAFT_RE.search(text))


# --- Greeting (incl. yo, sup, casual) ---

_GREETING_EXACT = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "yo",
        "sup",
        "hey there",
        "hi there",
        "hello there",
        "good morning",
        "good afternoon",
        "good evening",
    }
)
_GREETING_CASUAL = re.compile(
    r"^(hi|hello|hey|yo|sup)([\s,]+(there|bro|buddy|dude|mate|man|friend|pal))?(\s*[!?.…]*)?$",
    re.IGNORECASE,
)
_GREETING_TIMEOFDAY = re.compile(
    r"^(good\s+(morning|afternoon|evening))"
    r"([\s,]+(everyone|all|there|bro|buddy|dude|mate))?(\s*[!?.…]*)?$",
    re.IGNORECASE,
)
_GREETING_MAX_LEN = 48

_GREETING_REPLIES = [
    (
        "Hello — I'm HyeAero.AI, the aviation intelligence assistant for Hye Aero.\n"
        "I can help with aircraft missions, specifications, ownership research, and market insights. What are you working on?"
    ),
    (
        "HyeAero.AI — the aviation intelligence assistant for Hye Aero.\n"
        "Ask me about missions, specs, registry, comparisons, or buyer advisory anytime."
    ),
    (
        "Hi — I'm HyeAero.AI for Hye Aero. Whether it's range planning, market context, or ownership questions, I can steer you in the right direction."
    ),
    (
        "Good to connect — HyeAero.AI here, your aviation intelligence assistant for Hye Aero. What would you like to explore?"
    ),
    (
        "Hey — HyeAero.AI, the aviation intelligence assistant for Hye Aero. Tell me about the mission or aircraft you're thinking about."
    ),
]

# --- Small talk ---

_SMALL_TALK_EXACT = frozenset(
    {
        "nice",
        "cool",
        "thanks",
        "thank you",
        "thx",
        "ok",
        "okay",
        "k",
        "gg",
    }
)
_SMALL_TALK_PHRASES = [
    (re.compile(r"^hi\s+good\b", re.I), "Hi there! What can I help you with today?"),
    (re.compile(r"^how\s+are\s+you", re.I), "I'm doing well, thanks—and ready to help with aircraft research or market questions whenever you are."),
    (re.compile(r"^what'?s\s+up\b", re.I), "Hey! What can I help you with today?"),
]

_SMALL_TALK_GENERIC = [
    "Happy to help — HyeAero.AI here for Hye Aero. What are you working on?",
    "Glad we're chatting — I can dive into specs, ownership, missions, or market context whenever you're ready.",
    "Sounds good. If you have an aviation question, I'm here.",
]

# --- Identity ---

_IDENTITY = re.compile(
    r"^("
    r"who are you|what are you\b|what do you do|introduce yourself|"
    r"tell me about yourself|"
    r"what can you help(\s+me)?\s+with|what can you do for me|what do you help with"
    r")\??\s*$",
    re.I,
)
_HYEAERO_COMPANY = re.compile(
    r"^("
    r"what\s+is\s+hye\s*aero|what'?s\s+hye\s*aero|who\s+is\s+hye\s*aero|"
    r"tell\s+me\s+about\s+hye\s*aero|what\s+does\s+hye\s*aero\s+do|"
    r"what\s+is\s+hyeaero|what\s+does\s+hyeaero\s+do|tell\s+me\s+about\s+hyeaero|"
    r"describe\s+hye\s*aero|about\s+hye\s*aero"
    r")\b.*$",
    re.I,
)
_IDENTITY_MAX_LEN = 400

IDENTITY_REPLY = (
    "I'm HyeAero.AI — the aviation intelligence assistant for Hye Aero.\n\n"
    "I help you with aircraft missions, specifications, ownership research, market insights, comparisons, and buyer advisory — "
    "the same topics Hye Aero supports as an aviation intelligence and brokerage-support platform."
)

HYEAERO_COMPANY_REPLY = (
    "Hye Aero is an aviation intelligence platform: it combines aircraft brokerage support, "
    "data-driven market research, and tooling for specifications, ownership intelligence, "
    "mission analysis, market listings, aircraft comparison, and buyer advisory.\n\n"
    "I'm HyeAero.AI — I bring that expertise into the conversation so you get broker-grade, mission-aware answers in one place."
)

# Exported for tests / stable branding line
GREETING_REPLY = _GREETING_REPLIES[0]
ALL_GREETING_REPLIES = tuple(_GREETING_REPLIES)


def _pick_greeting_reply(query: str) -> str:
    h = int(hashlib.md5((query or "").strip().lower().encode(), usedforsecurity=False).hexdigest(), 16)
    return _GREETING_REPLIES[h % len(_GREETING_REPLIES)]


def _is_primary_greeting(raw: str) -> bool:
    low = raw.strip().lower().rstrip("!?.… ")
    if low in _GREETING_EXACT:
        return True
    if len(raw) > _GREETING_MAX_LEN:
        return False
    if _GREETING_CASUAL.match(low):
        return True
    if _GREETING_TIMEOFDAY.match(low):
        return True
    return False


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w\'\-]+\b", (text or "").strip()))


def _op_symbol_word(op: str) -> str:
    return {"+": "+", "-": "-", "*": "×", "/": "÷"}.get(op, op)


def _try_arithmetic_reply(raw: str) -> Optional[str]:
    """Simple integer arithmetic → formatted answer + HyeAero.AI aviation reminder (no tools)."""
    s0 = (raw or "").strip()
    if query_has_aviation_signals(s0):
        return None
    s = re.sub(r"^(what\s+is|what's|calc|compute|calculate)\s+", "", s0, flags=re.I)
    s = s.rstrip().rstrip("!?.…")
    s_compact = re.sub(r"\s+", "", s)
    if not re.match(r"^\d+[\+\-\*\/]\d+$", s_compact):
        return None
    try:
        op = next(c for c in s_compact if c in "+-*/")
        a_str, b_str = s_compact.split(op, 1)
        a, b = int(a_str), int(b_str)
        sym = _op_symbol_word(op)
        if op == "+":
            val = a + b
        elif op == "-":
            val = a - b
        elif op == "*":
            val = a * b
        else:
            if b == 0:
                return None
            val = a / b
            if val == int(val):
                val = int(val)
    except (ValueError, StopIteration):
        return None
    line1 = f"{a} {sym} {b} = {val}."
    return f"{line1}\n\n{_pick_aviation_reminder(s0)}"


def _non_aviation_llm_gate(query: str) -> bool:
    """True when the message looks like general knowledge (not aviation) worth a short LLM answer."""
    q = (query or "").strip()
    if not q or query_has_aviation_signals(q):
        return False
    if _TRIVIA_NON_AVIATION_HINT.search(q):
        return True
    wc = _word_count(q)
    if wc < 4:
        return False
    if "?" in q:
        return True
    if _WH_GENERAL.match(q.strip()):
        return True
    return False


def _non_aviation_llm_enabled() -> bool:
    """Non-aviation LLM answers are on by default; set CONSULTANT_NON_AVIATION_LLM=0 to skip."""
    v = (os.getenv("CONSULTANT_NON_AVIATION_LLM") or "").strip().lower()
    return v not in ("0", "false", "no", "off")


def _non_aviation_llm_reply(
    query: str,
    history: Optional[List[dict]],
    *,
    api_key: str,
    model: str,
    timeout: float = 22.0,
) -> Optional[str]:
    """
    Brief, accurate non-aviation answer + aviation-specialization reminder.
    Never mentions internal datasets, databases, or system limitations.
    """
    if not (api_key or "").strip() or not (query or "").strip():
        return None
    try:
        import openai

        hist = ""
        if history:
            for h in history[-4:]:
                role = (h.get("role") or "user").strip().lower()
                c = (h.get("content") or "").strip()
                if c and role in ("user", "assistant"):
                    hist += f"{role}: {c}\n"

        sys = """You are HyeAero.AI — the user-facing assistant for Hye Aero, an aviation intelligence and brokerage-support platform.

The user asked something that is NOT about aviation (e.g. math, geography, general knowledge, science trivia).

Output rules:
1) Answer their question first — briefly, correctly, and directly (plain text). No markdown # headers or ** bold.
2) Then one blank line, then a short friendly second paragraph reminding them you specialize in aviation intelligence for Hye Aero: missions, aircraft specifications, ownership research, and market insights. Sound like a knowledgeable aviation consultant, not a restricted chatbot.
3) If they asked for a joke, tell ONE short friendly joke, preferably aviation-themed, before the reminder paragraph.
4) NEVER say or imply: "internal dataset", "database records", "our database", "system limitations", "I can't access", or similar. Never refuse a normal general-knowledge question — answer it, then pivot to aviation.
5) Keep the total response concise (under ~180 words). Friendly, professional, broker-adjacent tone."""

        user = f"Recent conversation:\n{hist}\n\nUser message:\n{query.strip()}"
        client = openai.OpenAI(api_key=api_key, timeout=timeout)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            max_tokens=400,
            temperature=0.35,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return None
        low = text.lower()
        for bad in (
            "internal dataset",
            "our database",
            "database records",
            "system limitation",
            "i don't have access",
            "records not found",
        ):
            if bad in low:
                logger.warning("non_aviation LLM reply contained disallowed phrase; falling back")
                return None
        return text
    except Exception as e:
        logger.debug("non_aviation LLM reply failed: %s", e)
        return None


def _rules_classify_and_reply(query: str) -> ConversationGuardResult:
    raw = (query or "").strip()
    if not raw:
        return ConversationGuardResult(
            ConversationMessageType.SMALL_TALK,
            "HyeAero.AI — the aviation intelligence assistant for Hye Aero. What mission or market question can I help with?",
        )

    jr = _try_joke_reply(raw)
    if jr:
        return ConversationGuardResult(ConversationMessageType.NON_AVIATION_GENERAL, jr)

    ar = _try_arithmetic_reply(raw)
    if ar:
        return ConversationGuardResult(ConversationMessageType.NON_AVIATION_GENERAL, ar)

    if len(raw) <= _IDENTITY_MAX_LEN and _HYEAERO_COMPANY.match(raw.strip()):
        return ConversationGuardResult(ConversationMessageType.IDENTITY_QUESTION, HYEAERO_COMPANY_REPLY)

    if len(raw) <= _IDENTITY_MAX_LEN and _IDENTITY.match(raw):
        return ConversationGuardResult(ConversationMessageType.IDENTITY_QUESTION, IDENTITY_REPLY)

    if _is_primary_greeting(raw):
        return ConversationGuardResult(ConversationMessageType.GREETING, _pick_greeting_reply(raw))

    low = raw.lower().strip()
    if low in _SMALL_TALK_EXACT:
        return ConversationGuardResult(
            ConversationMessageType.SMALL_TALK,
            random.choice(_SMALL_TALK_GENERIC),
        )
    for cre, rep in _SMALL_TALK_PHRASES:
        if cre.match(low) or cre.match(raw.strip()):
            return ConversationGuardResult(ConversationMessageType.SMALL_TALK, rep)

    wc = _word_count(raw)
    if 1 <= wc <= 3 and not query_has_aviation_signals(raw):
        return ConversationGuardResult(
            ConversationMessageType.SMALL_TALK,
            _pick_greeting_reply(raw),
        )

    return ConversationGuardResult(ConversationMessageType.AVIATION_QUERY, None)


def _llm_classify_message_type(
    query: str,
    history: Optional[List[dict]],
    *,
    api_key: str,
    model: str,
    timeout: float = 12.0,
) -> Optional[ConversationMessageType]:
    if not (api_key or "").strip() or not (query or "").strip():
        return None
    try:
        import openai

        hist = ""
        if history:
            for h in history[-6:]:
                role = (h.get("role") or "user").strip().lower()
                c = (h.get("content") or "").strip()
                if c and role in ("user", "assistant"):
                    hist += f"{role}: {c}\n"

        sys = """You classify the user's latest message for HyeAero.AI (aviation intelligence assistant for Hye Aero).
Return JSON only: {"type": "<one word>"}

Allowed values:
- greeting — hi, hello, hey, yo, sup, good morning, short casual hellos (no substantive question).
- small_talk — thanks, ok, nice, cool, how are you, hi good, tiny non-aviation fragments under ~4 words that are not trivia questions.
- identity_question — who are you, what do you do, what can you help with, what is Hye Aero.
- non_aviation_general — general knowledge unrelated to aviation: math (beyond tiny fragments), geography, capitals, science trivia, history, jokes, "what is X" when X is not aircraft/missions/market/registry.
- aviation_question — aircraft, tails, N-numbers, registrations, listings, range, payload, MTOW, specs, ownership lookup, broker market, comparisons, flight planning, charter operations on aircraft, etc.

Choose aviation_question if the message is about aircraft or business aviation even partly.
Choose non_aviation_general when the user clearly wants a general (non-aviation) fact or joke."""

        user = f"Conversation:\n{hist}\nLatest message:\n{query.strip()}"
        client = openai.OpenAI(api_key=api_key, timeout=timeout)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            max_tokens=64,
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        t = (data.get("type") or data.get("intent") or "").strip().lower()
        if t == "identity":
            t = "identity_question"
        if t in ("aviation_query", "aviation_question", "aircraft", "consultant"):
            return ConversationMessageType.AVIATION_QUERY
        if t in ("non_aviation_general", "general_knowledge", "trivia", "non_aviation"):
            return ConversationMessageType.NON_AVIATION_GENERAL
        if t in ("greeting", "small_talk", "identity_question"):
            try:
                return ConversationMessageType(t)
            except ValueError:
                return None
    except Exception as e:
        logger.debug("conversation guard LLM classify failed: %s", e)
    return None


def evaluate_conversation_guard(
    query: str,
    history: Optional[List[dict]] = None,
    *,
    openai_api_key: str = "",
    chat_model: str = "",
) -> ConversationGuardResult:
    """
    Classify message and produce a conversational reply when tools must not run.

    General non-aviation questions get a direct answer first, then a gentle aviation-focus reminder.
    """
    rules = _rules_classify_and_reply(query)
    key = (openai_api_key or "").strip()
    model = (chat_model or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip()

    if _env_truthy("CONSULTANT_CONVERSATION_GUARD_LLM") and key:
        llm_t = _llm_classify_message_type(query, history, api_key=key, model=model)
        if llm_t is not None:
            if llm_t == ConversationMessageType.AVIATION_QUERY:
                return ConversationGuardResult(ConversationMessageType.AVIATION_QUERY, None)
            if llm_t == ConversationMessageType.NON_AVIATION_GENERAL:
                reply = _non_aviation_llm_reply(query, history, api_key=key, model=model)
                if reply:
                    return ConversationGuardResult(ConversationMessageType.NON_AVIATION_GENERAL, reply)
                jr = _try_joke_reply((query or "").strip())
                if jr:
                    return ConversationGuardResult(ConversationMessageType.NON_AVIATION_GENERAL, jr)
                ar = _try_arithmetic_reply((query or "").strip())
                if ar:
                    return ConversationGuardResult(ConversationMessageType.NON_AVIATION_GENERAL, ar)
                return rules
            if llm_t == ConversationMessageType.IDENTITY_QUESTION:
                qstrip = (query or "").strip()
                if len(qstrip) <= _IDENTITY_MAX_LEN and _HYEAERO_COMPANY.match(qstrip):
                    return ConversationGuardResult(ConversationMessageType.IDENTITY_QUESTION, HYEAERO_COMPANY_REPLY)
                return ConversationGuardResult(ConversationMessageType.IDENTITY_QUESTION, IDENTITY_REPLY)
            if llm_t == ConversationMessageType.GREETING:
                return ConversationGuardResult(ConversationMessageType.GREETING, _pick_greeting_reply(query))
            if llm_t == ConversationMessageType.SMALL_TALK:
                ar = _try_arithmetic_reply((query or "").strip())
                if ar:
                    return ConversationGuardResult(ConversationMessageType.NON_AVIATION_GENERAL, ar)
                jr = _try_joke_reply((query or "").strip())
                if jr:
                    return ConversationGuardResult(ConversationMessageType.NON_AVIATION_GENERAL, jr)
                raw = (query or "").strip()
                low = raw.lower()
                if low in _SMALL_TALK_EXACT:
                    return ConversationGuardResult(
                        ConversationMessageType.SMALL_TALK,
                        random.choice(_SMALL_TALK_GENERIC),
                    )
                for cre, rep in _SMALL_TALK_PHRASES:
                    if cre.match(low):
                        return ConversationGuardResult(ConversationMessageType.SMALL_TALK, rep)
                if _non_aviation_llm_enabled() and _non_aviation_llm_gate(raw):
                    reply = _non_aviation_llm_reply(query, history, api_key=key, model=model)
                    if reply:
                        return ConversationGuardResult(ConversationMessageType.NON_AVIATION_GENERAL, reply)
                return ConversationGuardResult(
                    ConversationMessageType.SMALL_TALK,
                    random.choice(_SMALL_TALK_GENERIC),
                )

    if (
        rules.message_type == ConversationMessageType.AVIATION_QUERY
        and key
        and _non_aviation_llm_enabled()
        and _non_aviation_llm_gate(query)
    ):
        reply = _non_aviation_llm_reply(query, history, api_key=key, model=model)
        if reply:
            return ConversationGuardResult(ConversationMessageType.NON_AVIATION_GENERAL, reply)

    return rules


def consultant_small_talk_reply(
    query: str,
    history: Optional[List[dict]] = None,
    *,
    openai_api_key: str = "",
    chat_model: str = "",
) -> Optional[str]:
    """Backward-compatible: return reply text if tools should be skipped, else None."""
    r = evaluate_conversation_guard(
        query,
        history,
        openai_api_key=openai_api_key,
        chat_model=chat_model,
    )
    if r.message_type == ConversationMessageType.AVIATION_QUERY:
        return None
    return r.reply
