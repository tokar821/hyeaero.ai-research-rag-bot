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

_JOKE_REQUEST_RE = re.compile(
    r"\b("
    r"tell\s+me\s+(a\s+)?joke|"
    r"make\s+me\s+laugh|"
    r"got\s+(a\s+)?joke|"
    r"any\s+jokes?|"
    r"dad\s+joke|"
    r"funny\s+joke|"
    r"joke\s+me|"
    r"pls\s+joke|"
    r"please\s+joke"
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
    # Simple joke request: deliver the joke only—no forced aviation pivot (consultant, not script).
    return joke


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
    "Morning — what are you working on today?",
    "Hey — good to connect. What's on your mind?",
    "Hi there. If you have a route, aircraft, or market question, I can help you think it through.",
    "Good to connect. Where would you like to start?",
    "Hello — I'm here when you're ready with an aviation question.",
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
    (re.compile(r"^hi\s+good\b", re.I), "Hi — good to hear it. What's on your mind?"),
    (re.compile(r"^how\s+are\s+you", re.I), "Doing well — thanks. What are you working on in the jet space?"),
    (re.compile(r"^what'?s\s+up\b", re.I), "Hey — what's the question?"),
]

# Softer than product-pitching every time — casual chat should feel human first.
_SMALL_TALK_GENERIC = [
    "Got it — I'm here when you want to pick up an aircraft or market topic.",
    "Sounds good. Whenever you're ready to go deeper on something, just say the word.",
    "Understood — talk soon.",
]

_SMALL_TALK_WARM_SHORT = [
    "Thanks — I appreciate it.",
    "Noted. Same here.",
    "Appreciate it.",
]

# Two-word well-wishes / cheer — not aviation greetings.
_HAPPY_OR_CHEER_RE = re.compile(
    r"^(happy\s+(today|birthday|monday|tuesday|wednesday|thursday|friday|weekend|holidays?)|"
    r"have\s+a\s+(great|good|nice)\s+(day|week|weekend)|"
    r"cheers\b|"
    r"congrats|congratulations)\b",
    re.I,
)

# Laughter / compliments — answer like a person, not a CRM form.
_COMPLIMENT_OR_FUN_RE = re.compile(
    r"\b("
    r"you\s*(?:'|’)?re\s+funny|"
    r"you\s+are\s+funny|"
    r"you'?re\s+hilarious|"
    r"that\s*(?:was\s*)?funny|"
    r"\b(?:haha|hahaha|lol|lmao|rofl)\b|"
    r"good\s+one|nice\s+one|"
    r"love\s+it|"
    r"you\s+crack\s+me\s+up"
    r")\b",
    re.I,
)

_WHAT_DO_YOU_MEAN_RE = re.compile(
    r"^\s*(?:"
    r"what'?s\s+mean|"
    r"what\s+do\s+you\s+mean|"
    r"what\s+does\s+that\s+mean|"
    r"wdym|"
    r"meaning"
    r")\s*(?:bro|dude|man|mate|fam|lol)?\s*[?.!…]*\s*$",
    re.I,
)

# Only these short lines get a **greeting-style** reply — not every 2–3 word message.
_SHORT_GREETING_LIKE = re.compile(
    r"^(hi|hello|hey|yo|sup|hiya|howdy)([\s,]+(there|bro|buddy|dude|mate|man|friend|pal|all))?(\s*[!?.…]*)?$",
    re.I,
)

# --- Identity ---

_LEADING_DISCOURSE_PREFIX_RE = re.compile(
    r"^(?:(?:good|great|nice|awesome|cool|so|ok|okay|well|hey|hi|yeah|yep|thanks|thank\s+you|oh)(?:[!?.…,]\s*|\s+))+",
    re.I,
)


def _normalize_for_identity_match(raw: str) -> str:
    """Strip leading fillers like 'good, ' / 'so, ' so company/identity regexes still match."""
    s = (raw or "").strip()
    s = _LEADING_DISCOURSE_PREFIX_RE.sub("", s)
    return s.strip()


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
    r"describe\s+hye\s*aero|about\s+hye\s*aero|"
    r"(?:how\s+about\s+)?(?:discuss(?:ing)?|talk(?:ing)?)\s+(?:about\s+){0,2}(?:hye\s*aero|hyeaero)\b|"
    r"(?:discuss|talk)\s+about\s+(?:hye\s*aero|hyeaero)\b"
    r")\s*[?.!…]*\s*$",
    re.I,
)
# Meta: what to ask the assistant — answer with examples, not the full identity brochure.
_WHAT_CAN_I_ASK_RE = re.compile(
    r"^("
    r"what\s+can\s+i\s+ask|"
    r"what\s+should\s+i\s+ask|"
    r"what\s+do\s+i\s+ask"
    r")\s*\??\s*$",
    re.I,
)

# Follow-ups like "great! can I ask more?" — consultant, not customer-service bot.
_CAN_I_ASK_MORE_RE = re.compile(
    r"^("
    r"can\s+i\s+ask\s+more|"
    r"may\s+i\s+ask\s+more|"
    r"can\s+i\s+keep\s+asking|"
    r"can\s+i\s+ask\s+another|"
    r"can\s+i\s+ask\s+something\s+else"
    r")\s*[?.!…]*\s*$",
    re.I,
)

_THANKS_AND_FAREWELL_RE = re.compile(
    r"(?is)"
    r"\b(?:thanks|thank\s+you|thx|ty)\b.{0,120}?\b"
    r"(?:have\s+a\s+(?:great|good|nice)\s+(?:day|week|weekend)|"
    r"(?:great|good|nice)\s+day)\b",
)
_IDENTITY_MAX_LEN = 400

_IDENTITY_REPLIES = [
    (
        "I'm HyeAero.AI — I work with Hye Aero as an aviation intelligence assistant. "
        "I help people think through missions, specs, ownership, and market questions in plain language."
    ),
    (
        "I'm HyeAero.AI for Hye Aero — basically a consultant-style assistant for aircraft and market topics. "
        "What are you trying to figure out?"
    ),
    (
        "HyeAero.AI here — I focus on business aviation: aircraft, trips, registry context, and market readouts for Hye Aero. "
        "How can I help?"
    ),
]

HYEAERO_COMPANY_REPLY = (
    "Hye Aero is an aviation intelligence and brokerage support platform focused on aircraft research, "
    "ownership intelligence, and market insights.\n\n"
    "HyeAero.AI is the conversational side — concise, consultant-style answers, not a generic chatbot."
)

_BOT_OR_AI_QUESTION_RE = re.compile(
    r"^\s*(are\s+you\s+(a\s+)?bot\b|are\s+you\s+an?\s+ai\b|you\s+a\s+chatgpt\b|"
    r"is\s+this\s+a\s+bot)\??\s*$",
    re.I,
)

_CONFUSED_USER_RE = re.compile(
    r"\b(i'?m\s+confused|i\s+don'?t\s+get\s+it|this\s+is\s+confusing|"
    r"that\s+confuses\s+me|i'?m\s+lost)\b",
    re.I,
)

# Backward compatibility — first variant; actual replies rotate via _pick_identity_reply.
IDENTITY_REPLY = _IDENTITY_REPLIES[0]


def _pick_identity_reply(query: str) -> str:
    h = int(hashlib.md5((query or "").strip().lower().encode(), usedforsecurity=False).hexdigest(), 16)
    return _IDENTITY_REPLIES[h % len(_IDENTITY_REPLIES)]


_WHAT_CAN_I_ASK_REPLIES = [
    (
        "Anything in business aviation works: a registration, a trip you're planning, a model you're comparing, "
        "or a buy/sell question — whatever's most useful to you right now."
    ),
    (
        "You could start with something specific — a city pair, cabin size, budget band, or a tail you're curious about — "
        "and I'll help you think it through."
    ),
    (
        "Common threads are mission fit, specs, ownership context, or market color — pick what matters and we'll go from there."
    ),
]


def _pick_what_can_i_ask_reply(query: str) -> str:
    h = int(hashlib.md5((query or "").strip().lower().encode(), usedforsecurity=False).hexdigest(), 16)
    return _WHAT_CAN_I_ASK_REPLIES[h % len(_WHAT_CAN_I_ASK_REPLIES)]


_CAN_I_ASK_MORE_REPLIES = [
    "Of course — what's next on your list?",
    "Sure — go ahead. What are you trying to pin down?",
    "Anytime. What do you want to look at next?",
]

_FAREWELL_REPLIES = [
    "You too — fly safe if you're heading out.",
    "Likewise. Reach out when something aircraft-related is on deck.",
    "Appreciate it — have a good one.",
]


def _pick_can_i_ask_more_reply(query: str) -> str:
    h = int(hashlib.md5((query or "").strip().lower().encode(), usedforsecurity=False).hexdigest(), 16)
    return _CAN_I_ASK_MORE_REPLIES[h % len(_CAN_I_ASK_MORE_REPLIES)]


def _pick_farewell_reply(query: str) -> str:
    h = int(hashlib.md5((query or "").strip().lower().encode(), usedforsecurity=False).hexdigest(), 16)
    return _FAREWELL_REPLIES[h % len(_FAREWELL_REPLIES)]


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
    """Simple arithmetic → numeric answer only (no aviation commentary)."""
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
    # Minimal answer only, e.g. "13." for 5+8 — no extra sentences.
    return f"{val}."


def _non_aviation_llm_gate(query: str) -> bool:
    """True when the message looks like general knowledge (not aviation) worth a short LLM answer."""
    q = (query or "").strip()
    if not q or query_has_aviation_signals(q):
        return False
    if _TRIVIA_NON_AVIATION_HINT.search(q):
        return True
    # Clarification / "what did you mean" — allow LLM even when short (rules may still catch common typos).
    if re.search(r"\b(wdym|what\s*(do\s+you|’?s)\s+mean|what\s+does\s+that\s+mean)\b", q, re.I):
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


def _is_hye_aero_company_query_exact(query: str) -> bool:
    """Stable branding copy is handled by templates — skip LLM override."""
    raw = (query or "").strip()
    if len(raw) > _IDENTITY_MAX_LEN:
        return False
    return bool(_HYEAERO_COMPANY.match(_normalize_for_identity_match(raw)))


def _non_aviation_llm_system_prompt(hint: Optional[ConversationMessageType]) -> str:
    """Strong identity + role; hint steers tone for this turn (hybrid with keyword routing)."""
    base = """You are **HyeAero.AI** — you represent **Hye Aero** as a professional **business-aviation** consultant assistant.
**Who you are:** HyeAero.AI — the conversational front for Hye Aero.
**What Hye Aero is:** aviation intelligence and brokerage support (aircraft research, ownership context, market insight, mission thinking).
**Your main role:** help users with aircraft, missions, specs, registry/market questions, and buyer-style guidance — but **this specific message is not an aviation task**, so you respond like a sharp human, not a brochure.

**Output rules (all modes):**
1) Plain text only. No markdown # headers or ** bold.
2) Do **not** add forced aviation commentary, metaphors, or tie-ins unless the user brought up flying. No philosophical filler ("Aviation is about precision", "like a well-planned flight").
3) For **simple arithmetic** when they only want the result: reply with **just the number and a period** (e.g. "13.") — no extra sentences.
4) NEVER mention: datasets, databases, internal records, scraping, Tavily, Pinecone, RAG, vector, SQL, tools, or "training data".
5) Forbidden: "I'm here to help", "feel free", "just let me know", "if you're curious", "what would you like to know", "absolutely", "don't hesitate".
6) Keep it concise unless the question truly needs more (cap ~180 words)."""

    if hint == ConversationMessageType.GREETING:
        return (
            base
            + "\n\n**This turn (greeting):** Respond warmly and briefly — like a colleague. **Do not** pitch aviation or list product capabilities unless they asked."
        )
    if hint == ConversationMessageType.SMALL_TALK:
        return (
            base
            + "\n\n**This turn (small talk / thanks / cheer):** Match the user's tone in **one short reply**. No aviation pivot unless natural."
        )
    if hint == ConversationMessageType.IDENTITY_QUESTION:
        return (
            base
            + "\n\n**This turn (who are you / what do you do):** Explain HyeAero.AI and Hye Aero in **2–4 short sentences** — consultant-like, not a feature list. Mention missions, specs, ownership, market naturally."
        )
    if hint == ConversationMessageType.NON_AVIATION_GENERAL:
        return (
            base
            + "\n\n**This turn (general knowledge / joke / trivia):** Answer correctly and briefly. If a joke was requested, ONE short joke (aviation-themed is OK), then stop. For math, numeric-only when appropriate."
        )
    return (
        base
        + "\n\n**This turn:** Answer what they asked — briefly, naturally, consultant peer tone."
    )


def _non_aviation_llm_reply(
    query: str,
    history: Optional[List[dict]],
    *,
    api_key: str,
    model: str,
    timeout: float = 22.0,
    hint: Optional[ConversationMessageType] = None,
) -> Optional[str]:
    """
    Non-aviation reply via LLM — hybrid path with keyword routing (hint).
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

        sys = _non_aviation_llm_system_prompt(hint)

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
            "just let me know",
            "if you're curious",
            "if you are curious",
            "feel free to",
            "what would you like to know",
            "don't hesitate",
            "aviation is about precision",
            "well-planned flight",
            "while we're on the topic of jets",
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
            "I'm here when you're ready.",
        )

    jr = _try_joke_reply(raw)
    if jr:
        return ConversationGuardResult(ConversationMessageType.NON_AVIATION_GENERAL, jr)

    ar = _try_arithmetic_reply(raw)
    if ar:
        return ConversationGuardResult(ConversationMessageType.NON_AVIATION_GENERAL, ar)

    if len(raw) <= 200 and not query_has_aviation_signals(raw) and _BOT_OR_AI_QUESTION_RE.match(raw.strip()):
        return ConversationGuardResult(
            ConversationMessageType.IDENTITY_QUESTION,
            "I'm HyeAero.AI — an aviation intelligence assistant for Hye Aero, not a generic chatbot. "
            "I help with missions, specs, ownership, and market questions in a consultant-style conversation.",
        )

    if len(raw) <= 320 and not query_has_aviation_signals(raw) and _CONFUSED_USER_RE.search(raw):
        return ConversationGuardResult(
            ConversationMessageType.SMALL_TALK,
            "No problem — what part would you like me to clarify?",
        )

    _id_raw = raw.strip()
    _id_norm = _normalize_for_identity_match(_id_raw)
    if len(_id_raw) <= _IDENTITY_MAX_LEN and _HYEAERO_COMPANY.match(_id_norm):
        return ConversationGuardResult(ConversationMessageType.IDENTITY_QUESTION, HYEAERO_COMPANY_REPLY)

    if len(_id_raw) <= _IDENTITY_MAX_LEN and _WHAT_CAN_I_ASK_RE.match(_id_norm):
        return ConversationGuardResult(
            ConversationMessageType.IDENTITY_QUESTION,
            _pick_what_can_i_ask_reply(_id_raw),
        )

    if len(_id_raw) <= _IDENTITY_MAX_LEN and _IDENTITY.match(_id_norm):
        return ConversationGuardResult(ConversationMessageType.IDENTITY_QUESTION, _pick_identity_reply(_id_raw))

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

    stripped = raw.strip()
    if _HAPPY_OR_CHEER_RE.match(stripped):
        return ConversationGuardResult(
            ConversationMessageType.SMALL_TALK,
            "Same to you — safe travels if you're flying.",
        )
    if _COMPLIMENT_OR_FUN_RE.search(raw):
        return ConversationGuardResult(
            ConversationMessageType.SMALL_TALK,
            "Glad that landed. Happy to go deeper on aircraft, missions, or market when you want.",
        )
    if _WHAT_DO_YOU_MEAN_RE.match(stripped):
        return ConversationGuardResult(
            ConversationMessageType.SMALL_TALK,
            "If you're asking what something meant — point me at which line or term and I'll clarify. "
            "If it's aviation jargon (NM, reserves, MTOW, etc.), say the word and I'll explain in plain English.",
        )

    wc = _word_count(raw)
    if 1 <= wc <= 3 and not query_has_aviation_signals(raw):
        low = stripped.lower().rstrip("!?.… ")
        # Do **not** treat arbitrary 2–3 word phrases as "hello" (e.g. "Happy today") — that reads robotic.
        if _SHORT_GREETING_LIKE.match(stripped) or low in ("hi", "hey", "hello", "yo", "sup", "hiya", "howdy"):
            return ConversationGuardResult(ConversationMessageType.GREETING, _pick_greeting_reply(raw))
        if low in ("thanks", "thx", "ty", "ok", "okay", "k", "cool", "nice", "cheers", "gg"):
            return ConversationGuardResult(
                ConversationMessageType.SMALL_TALK,
                random.choice(_SMALL_TALK_WARM_SHORT),
            )
        return ConversationGuardResult(
            ConversationMessageType.SMALL_TALK,
            random.choice(_SMALL_TALK_GENERIC),
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

If the message is only a hello/thanks/small fragment with no aircraft or mission content, it is NOT aviation_question.
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
    raw = (query or "").strip()
    # Clarification follow-ups need the full consultant pipeline + history (e.g. "what's mean?" = explain prior reply).
    if history and _WHAT_DO_YOU_MEAN_RE.match(raw):
        return ConversationGuardResult(ConversationMessageType.AVIATION_QUERY, None)

    # Consultant-style closings / follow-ups — run before guard LLM so misclassification cannot
    # send "thanks, have a great day" into the aircraft tool pipeline.
    if not query_has_aviation_signals(raw):
        _prio_norm = _normalize_for_identity_match(raw)
        if _CAN_I_ASK_MORE_RE.match(_prio_norm):
            return ConversationGuardResult(
                ConversationMessageType.SMALL_TALK,
                _pick_can_i_ask_more_reply(raw),
            )
        if _THANKS_AND_FAREWELL_RE.search(raw):
            return ConversationGuardResult(
                ConversationMessageType.SMALL_TALK,
                _pick_farewell_reply(raw),
            )
        if len(raw) <= 200 and _BOT_OR_AI_QUESTION_RE.match(raw.strip()):
            return ConversationGuardResult(
                ConversationMessageType.IDENTITY_QUESTION,
                "I'm HyeAero.AI — an aviation intelligence assistant for Hye Aero, not a generic chatbot. "
                "I help with missions, specs, ownership, and market questions in a consultant-style conversation.",
            )

    # Deterministic tiny math — no LLM (avoids extra sentences).
    _arith = _try_arithmetic_reply(raw)
    if _arith:
        return ConversationGuardResult(ConversationMessageType.NON_AVIATION_GENERAL, _arith)

    rules = _rules_classify_and_reply(query)
    key = (openai_api_key or "").strip()
    model = (chat_model or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip()

    # --- Hybrid (keywords + LLM): keyword router picks message type; LLM generates the reply when possible. ---
    if (
        rules.message_type != ConversationMessageType.AVIATION_QUERY
        and key
        and _non_aviation_llm_enabled()
        and not _is_hye_aero_company_query_exact(raw)
    ):
        _hybrid = _non_aviation_llm_reply(
            query, history, api_key=key, model=model, hint=rules.message_type
        )
        if _hybrid:
            return ConversationGuardResult(rules.message_type, _hybrid)

    # --- Classifier LLM when keyword router said aviation (fix intent misfires). ---
    if (
        rules.message_type == ConversationMessageType.AVIATION_QUERY
        and _env_truthy("CONSULTANT_CONVERSATION_GUARD_LLM")
        and key
    ):
        llm_t = _llm_classify_message_type(query, history, api_key=key, model=model)
        if llm_t is not None and llm_t != ConversationMessageType.AVIATION_QUERY:
            if llm_t == ConversationMessageType.NON_AVIATION_GENERAL:
                if _non_aviation_llm_enabled():
                    reply = _non_aviation_llm_reply(
                        query,
                        history,
                        api_key=key,
                        model=model,
                        hint=ConversationMessageType.NON_AVIATION_GENERAL,
                    )
                    if reply:
                        return ConversationGuardResult(ConversationMessageType.NON_AVIATION_GENERAL, reply)
                jr = _try_joke_reply(raw)
                if jr:
                    return ConversationGuardResult(ConversationMessageType.NON_AVIATION_GENERAL, jr)
                ar = _try_arithmetic_reply(raw)
                if ar:
                    return ConversationGuardResult(ConversationMessageType.NON_AVIATION_GENERAL, ar)
                return rules
            if llm_t == ConversationMessageType.IDENTITY_QUESTION:
                qstrip = raw.strip()
                qnorm = _normalize_for_identity_match(qstrip)
                if len(qstrip) <= _IDENTITY_MAX_LEN and _HYEAERO_COMPANY.match(qnorm):
                    return ConversationGuardResult(ConversationMessageType.IDENTITY_QUESTION, HYEAERO_COMPANY_REPLY)
                if len(qstrip) <= _IDENTITY_MAX_LEN and _WHAT_CAN_I_ASK_RE.match(qnorm):
                    return ConversationGuardResult(
                        ConversationMessageType.IDENTITY_QUESTION,
                        _pick_what_can_i_ask_reply(qstrip),
                    )
                if _non_aviation_llm_enabled():
                    reply = _non_aviation_llm_reply(
                        query,
                        history,
                        api_key=key,
                        model=model,
                        hint=ConversationMessageType.IDENTITY_QUESTION,
                    )
                    if reply:
                        return ConversationGuardResult(ConversationMessageType.IDENTITY_QUESTION, reply)
                return ConversationGuardResult(
                    ConversationMessageType.IDENTITY_QUESTION,
                    _pick_identity_reply(qstrip),
                )
            if llm_t == ConversationMessageType.GREETING:
                if _non_aviation_llm_enabled():
                    reply = _non_aviation_llm_reply(
                        query, history, api_key=key, model=model, hint=ConversationMessageType.GREETING
                    )
                    if reply:
                        return ConversationGuardResult(ConversationMessageType.GREETING, reply)
                return ConversationGuardResult(ConversationMessageType.GREETING, _pick_greeting_reply(query))
            if llm_t == ConversationMessageType.SMALL_TALK:
                jr = _try_joke_reply(raw)
                if jr:
                    return ConversationGuardResult(ConversationMessageType.NON_AVIATION_GENERAL, jr)
                if _non_aviation_llm_enabled():
                    reply = _non_aviation_llm_reply(
                        query, history, api_key=key, model=model, hint=ConversationMessageType.SMALL_TALK
                    )
                    if reply:
                        return ConversationGuardResult(ConversationMessageType.SMALL_TALK, reply)
                low = raw.lower()
                if low in _SMALL_TALK_EXACT:
                    return ConversationGuardResult(
                        ConversationMessageType.SMALL_TALK,
                        random.choice(_SMALL_TALK_GENERIC),
                    )
                for cre, rep in _SMALL_TALK_PHRASES:
                    if cre.match(low):
                        return ConversationGuardResult(ConversationMessageType.SMALL_TALK, rep)
                if _non_aviation_llm_gate(raw):
                    reply = _non_aviation_llm_reply(
                        query,
                        history,
                        api_key=key,
                        model=model,
                        hint=ConversationMessageType.NON_AVIATION_GENERAL,
                    )
                    if reply:
                        return ConversationGuardResult(ConversationMessageType.NON_AVIATION_GENERAL, reply)
                return ConversationGuardResult(
                    ConversationMessageType.SMALL_TALK,
                    random.choice(_SMALL_TALK_GENERIC),
                )

    # Keyword router said aviation but message is general knowledge / chat (classifier missed).
    if (
        rules.message_type == ConversationMessageType.AVIATION_QUERY
        and key
        and _non_aviation_llm_enabled()
        and _non_aviation_llm_gate(query)
    ):
        reply = _non_aviation_llm_reply(
            query,
            history,
            api_key=key,
            model=model,
            hint=ConversationMessageType.NON_AVIATION_GENERAL,
        )
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
