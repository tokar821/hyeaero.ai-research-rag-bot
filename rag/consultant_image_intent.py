"""
Hybrid image-intent detection for Ask Consultant.

Architecture (high level)::

    User message
         → keyword image detector (strict + broad phrases)
         → visual follow-up phrases + thread aircraft entity (pronouns)
         → LLM multi-class intent (when enabled and keywords miss)
         → (downstream) aircraft entity + :func:`rag.consultant_market_lookup.build_aircraft_photo_focus_tavily_query`
         → Tavily with include_images
         → curated gallery URLs (see :func:`services.consultant_aircraft_images.build_consultant_aircraft_images`)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import List, Optional, Tuple

from rag.aviation_tail import is_invalid_placeholder_us_n_tail
from rag.consultant_market_lookup import wants_consultant_aircraft_images_in_answer
from rag.phlydata_consultant_lookup import consultant_phly_lookup_token_list

logger = logging.getLogger(__name__)

CONSULTANT_INTENTS = (
    "aircraft_image_request",
    "aircraft_information",
    "aircraft_recommendation",
    "aircraft_registry_lookup",
    "general_question",
)

_HYBRID_INTENT_SYSTEM = """You classify the latest user message for Hye Aero Ask Consultant (business aviation brokers).

Use the **conversation excerpt** only to resolve pronouns ("it", "that one", "them") and **recent aircraft context**. The classification applies to what the user wants **in this turn**.

**Semantic rule:** If the user is asking to **see** the **aircraft** in a **visual** sense (photos, pictures, what it looks like), choose **aircraft_image_request** even when they do not use the words "photo" or "image". Do **not** require exact keyword matches. Paraphrases and intent matter.

Choose exactly one label:

- **aircraft_image_request** — They want **photos / pictures / images** of the aircraft (not maps, not live flight tracking). Treat as image intent when they say or mean things like: "show me", "can I see", "let me see", "photo", "picture", "image", "what does it look like", "do you have photos", "I've never seen one before", "I'm curious what it looks like", "wanna see", "trying to see", "any pics?", visual/gallery language, or **see/show + tail or model** from context.
- **aircraft_information** — Specs, range, performance, cabin **dimensions as numbers**, programs, history, "tell me about", **without** asking to **see** or **picture** the aircraft in this message.
- **aircraft_recommendation** — Which jet to buy, shortlist, compare models for acquisition, mission fit **without** a visual request in this message.
- **aircraft_registry_lookup** — Who owns, tail lookup, registration, serial, FAA/registrant **without** asking for photos now.
- **general_question** — Greetings, math, non-aviation, or aviation small talk with **no** visual / "see the jet" ask.

Output only valid JSON: {"intent": "<one of the five labels above>"}"""

# Short lines that imply "show me the aircraft we were discussing" (require thread aircraft context).
# "See" / "wanna see" / "try to see" mean **photos of the aircraft**, not maps or flight tracking.
_VISUAL_FOLLOWUP_INNER = re.compile(
    r"(?is)(?:"
    r"\bcan\s+i\s+see\s+(?:it|them|that|one|this)\b"
    r"|\bcould\s+i\s+see\s+(?:it|them|that|one|this)\b"
    r"|\blet\s+me\s+see\s+(?:it|them|that|one|this)\b"
    r"|\bi\s+want\s+to\s+see\s+(?:it|them|that|one|this)\b"
    r"|\b(?:i\s+)?wanna\s+see\s+(?:it|them|that|one|this)\b"
    r"|\btry(?:ing)?\s+to\s+see\s+(?:it|them|that|one|this)\b"
    r"|\b(?:i\s+)?(?:would\s+)?love\s+to\s+see\s+(?:it|them|that|one|this)\b"
    r"|\bi(?:'|’|\s)?d\s+like\s+to\s+see\s+(?:it|them|that|one|this)\b"
    r"|\bhope\s+to\s+see\s+(?:it|them|that|one|this)\b"
    r"|\bjust\s+wanna\s+see\s+(?:it|them|that|one|this)\b"
    r"|\b(?:show|showing)\s+me\s+(?:that\s+)?(?:one|it|this)\b"
    r"|\bshow\s+that\s+one\b"
    r"|\bany\s+photos?\s*\??\s*$"
    r"|\bgot\s+(?:any\s+)?(?:pics?|photos?)\s*\??\s*$"
    r"|\bpictures?\s*\?\s*$"
    r"|\bimages?\s*\?\s*$"
    r"|\bdo\s+you\s+have\s+(?:any\s+)?(?:photos?|pictures?|images?|pics?)\b"
    r"|\bi'?ve\s+never\s+(?:actually\s+)?seen\s+(?:one|it|this|that)\b"
    r"|\bi'?m\s+curious\s+(?:what\s+)?(?:it|this|that)\s+looks\s+like\b"
    r"|\bcurious\s+what\s+(?:it|this|that)\s+looks\s+like\b"
    r")",
)


def _thread_text_for_entity_resolution(
    query: str,
    history: Optional[List[dict]],
    *,
    max_messages: int = 14,
    max_chars_per_msg: int = 1400,
) -> str:
    parts: List[str] = []
    if history:
        for h in history[-max_messages:]:
            role = (h.get("role") or "").strip().lower()
            if role not in ("user", "assistant"):
                continue
            c = (h.get("content") or "").strip()
            if c:
                parts.append(c[:max_chars_per_msg])
    q = (query or "").strip()
    if q:
        parts.append(q)
    return "\n".join(parts)


def thread_has_aircraft_context(
    query: str,
    history: Optional[List[dict]],
) -> bool:
    """True when recent thread + current message mention a tail, serial, or recognized make/model."""
    blob = _thread_text_for_entity_resolution(query, history)
    if not blob.strip():
        return False
    from rag.aviation_tail import find_strict_tail_candidates_in_text

    if find_strict_tail_candidates_in_text(blob):
        return True
    if consultant_phly_lookup_token_list(query, history):
        return True
    from rag.consultant_query_expand import _detect_manufacturers, _detect_models

    lc = blob.lower()
    return bool(_detect_manufacturers(lc)) or bool(_detect_models(blob))


def broad_keyword_suggests_image_request(query: str) -> bool:
    """
    Loose keyword layer (current message only): photo / image / gallery / look like / show me a … / etc.

    Avoids bare \"can I see\" without an object (e.g. \"can I see the range\").
    """
    q = (query or "").strip()
    if not q:
        return False
    low = q.lower()
    if re.search(
        r"\b(?:photo|photograph|photographs|photos|picture|pictures|pic|pics|image|images|imags|gallery)\b",
        low,
    ):
        return True
    # "search on google for … images" style (no bare "google" — avoids unrelated product questions).
    if "google" in low and re.search(
        r"\b(?:image|images|imags|photo|photos|photograph|pictures|picture|pics)\b",
        low,
    ):
        return True
    if "look like" in low or "looks like" in low:
        return True
    if re.search(r"\bany\s+photos?\b", low):
        return True
    if re.search(r"\bwhat\s+does\s+it\s+look\s+like\b", low):
        return True
    if re.search(r"\bwhat\s+do\s+they\s+look\s+like\b", low):
        return True
    if re.search(r"\b(?:show|showing)\s+me\s+(?:a|an|the)\s+\S", low):
        return True
    # Pronoun "see it / that one" needs thread context — handled by :func:`visual_followup_suggests_image_request`.
    if re.search(r"\b(?:can|could)\s+you\s+show\s+me\b", low):
        return True
    if re.search(r"\blet\s+me\s+see\s+the\s+(?:aircraft|plane|jet)\b", low):
        return True
    if re.search(r"\bi\s+want\s+to\s+see\s+the\s+(?:aircraft|plane|jet)\b", low):
        return True
    if re.search(r"\b(?:i\s+)?wanna\s+see\s+the\s+(?:aircraft|plane|jet)\b", low):
        return True
    if re.search(r"\btry(?:ing)?\s+to\s+see\s+the\s+(?:aircraft|plane|jet)\b", low):
        return True
    if re.search(r"\bdo\s+you\s+have\s+(?:any\s+)?(?:photos?|pictures?|images?|pics?)\b", low):
        return True
    if re.search(r"\bi'?ve\s+never\s+(?:actually\s+)?seen\s+(?:one|it|this|that)\b", low):
        return True
    if re.search(r"\bi'?m\s+curious\s+(?:what\s+)?(?:it|this|that)\s+looks\s+like\b", low):
        return True
    if re.search(r"\bcurious\s+what\s+(?:it|this|that)\s+looks\s+like\b", low):
        return True
    if re.search(r"\b(?:show|showing)\s+me\s+(?:the\s+)?(?:jet|plane|aircraft)\b", low):
        return True
    return False


def visual_followup_suggests_image_request(
    query: str,
    history: Optional[List[dict]],
) -> bool:
    """Pronoun-style visual follow-ups: only when the thread clearly has aircraft context."""
    q = (query or "").strip()
    if not q or len(q) > 160:
        return False
    if not _VISUAL_FOLLOWUP_INNER.search(q):
        return False
    return thread_has_aircraft_context(query, history)


def classify_consultant_hybrid_intent_llm(
    query: str,
    history: Optional[List[dict]],
    *,
    api_key: str,
    model: str,
    timeout: float,
) -> Optional[str]:
    """
    Returns one of CONSULTANT_INTENTS, or None on failure.
    """
    if not (api_key or "").strip():
        return None
    try:
        import openai

        excerpt = _thread_text_for_entity_resolution(query, history, max_messages=10, max_chars_per_msg=900)
        user_block = (
            "Conversation excerpt (most recent first in thread order as listed):\n"
            f"{excerpt}\n\nLatest user message:\n{(query or '').strip()}"
        )
        client = openai.OpenAI(api_key=api_key, timeout=timeout)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _HYBRID_INTENT_SYSTEM},
                {"role": "user", "content": user_block},
            ],
            max_tokens=80,
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        intent = str(data.get("intent") or "").strip()
        if intent in CONSULTANT_INTENTS:
            return intent
        for k in CONSULTANT_INTENTS:
            if intent.lower() == k.lower():
                return k
    except Exception as e:
        logger.debug("classify_consultant_hybrid_intent_llm failed: %s", e)
    return None


def resolve_hybrid_image_gallery_intent(
    query: str,
    history: Optional[List[dict]],
    *,
    api_key: str,
    model: str,
) -> Tuple[bool, str, Optional[str]]:
    """
    Decide whether to run Tavily image search + UI gallery.

    Returns:
        (show_gallery, source_tag, llm_intent_or_none)

    ``source_tag`` is one of:
        keywords_strict, keywords_broad, keywords_followup, llm_aircraft_image_request,
        llm_not_image, llm_failed, off
    """
    q = (query or "").strip()
    if not q:
        return False, "off", None

    for _m in re.finditer(r"\bN(?=[A-Z0-9]*\d)[A-Z0-9]{2,6}\b", q, re.I):
        if is_invalid_placeholder_us_n_tail(_m.group(0)):
            return False, "invalid_placeholder_tail", None

    if wants_consultant_aircraft_images_in_answer(q):
        return True, "keywords_strict", None

    if broad_keyword_suggests_image_request(q):
        return True, "keywords_broad", None

    if visual_followup_suggests_image_request(q, history):
        return True, "keywords_followup", None

    if (os.getenv("CONSULTANT_IMAGE_INTENT_LLM") or "1").strip().lower() in ("0", "false", "no"):
        return False, "llm_disabled", None

    if not (api_key or "").strip():
        return False, "llm_no_api_key", None

    try:
        to = float((os.getenv("CONSULTANT_IMAGE_INTENT_TIMEOUT_SEC") or "14").strip())
    except ValueError:
        to = 14.0
    to = max(6.0, min(30.0, to))

    intent = classify_consultant_hybrid_intent_llm(
        q, history, api_key=api_key, model=model, timeout=to
    )
    if intent is None:
        return False, "llm_failed", None
    if intent == "aircraft_image_request":
        return True, "llm_aircraft_image_request", intent
    return False, "llm_not_image", intent
