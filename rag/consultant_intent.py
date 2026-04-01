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

**true** — User **explicitly** wants photos, images, pictures, a gallery, exterior/cabin shots, "what does it look like" in a visual sense, or short follow-ups that clearly mean "show me pics" in the same thread.

**false** — Specs, mission, price, ownership, comparison, "describe" or "tell me about" the aircraft **without** asking for photos, most operational questions, registry lookups, or general aviation knowledge without a visual request.

Output only valid JSON: {"show_aircraft_images": true or false}"""


def _history_blob(history: Optional[List[Dict[str, str]]], *, max_messages: int = 10) -> str:
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


def consultant_wants_aircraft_images_semantic(
    query: str,
    history: Optional[List[Dict[str, str]]],
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

        blob = _history_blob(history)
        user_msg = (
            f"Conversation (oldest last lines first):\n{blob}\n\n---\n\nLatest user message:\n{(query or '').strip()}"
        )
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
