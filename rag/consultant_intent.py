"""
Semantic intent for Ask Consultant gates (LLM + safe fallbacks).

Primary use: decide whether to fetch and return aircraft **photo URLs** from Tavily/listings
based on what the user **means**, not fixed keyword lists.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_IMAGE_INTENT_SYSTEM = """You classify user intent for Hye Aero Ask Consultant (aircraft research for brokers).

Should this turn include a **gallery of real aircraft photo URLs** (web + marketplace scrape), shown next to the text answer?

Classify using **only the latest user message** — do not infer image intent from earlier conversation turns.

**true** — The latest message **explicitly** asks for photos, images, pictures, a gallery, exterior/cabin shots, or clear visual "show me" language.

**false** — Specs, mission, price, ownership, comparison, follow-ups that do not ask for pictures, "describe" or "tell me about" without a visual request, registry lookups, or general aviation knowledge without a visual request in this message.

Output only valid JSON: {"show_aircraft_images": true or false}"""


def consultant_wants_aircraft_images_semantic(
    query: str,
    _history: Optional[List[Dict[str, str]]],
    *,
    api_key: str,
    model: str,
    timeout: float = 14.0,
) -> Optional[bool]:
    """
    LLM meaning-based image-gallery intent. Returns ``None`` on failure — use keyword fallback.
    """
    if not (api_key or "").strip() or not (query or "").strip():
        return None
    try:
        import openai

        user_msg = (query or "").strip()
        client = openai.OpenAI(api_key=api_key, timeout=timeout)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _IMAGE_INTENT_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=64,
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        v = data.get("show_aircraft_images")
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
    except Exception as e:
        logger.debug("consultant_wants_aircraft_images_semantic failed: %s", e)
    return None


def resolve_aircraft_image_gallery_intent(
    query: str,
    history: Optional[List[Dict[str, str]]],
    *,
    api_key: str,
    model: str,
    keyword_fallback: Callable[[], bool],
    keywords_only: bool,
) -> Tuple[bool, str]:
    """
    Legacy 2-tuple resolver. Production uses ``rag.consultant_image_intent.resolve_hybrid_image_gallery_intent``.

    Returns (show_gallery, source) where source is ``llm``, ``keywords``, or ``keywords_fallback``.
    """
    if keywords_only or not (api_key or "").strip():
        return keyword_fallback(), "keywords"
    try:
        to = float((os.getenv("CONSULTANT_IMAGE_INTENT_TIMEOUT_SEC") or "14").strip())
    except ValueError:
        to = 14.0
    to = max(6.0, min(30.0, to))
    sem = consultant_wants_aircraft_images_semantic(
        query, history, api_key=api_key, model=model, timeout=to
    )
    if sem is not None:
        return sem, "llm"
    return keyword_fallback(), "keywords_fallback"
