"""
Premium aircraft image retrieval pipeline (standalone).

Design goals:
- Accuracy over recall (strict scoring + hard thresholds)
- Trust over completeness (prefer returning nothing over wrong aircraft)
- No coupling to RAG / vector DB / chat history

This module intentionally keeps payloads small:
- SearchAPI responses are parsed into minimal dicts only
- No large blobs are logged by default
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import urlparse

import requests

from services.searchapi_aircraft_images import (
    SEARCHAPI_SEARCH_URL,
    build_searchapi_image_request_params,
    normalize_aircraft_name,
)

logger = logging.getLogger(__name__)

ImageView = Literal["exterior", "cabin", "cockpit", "interior", "bedroom"]


@dataclass(frozen=True)
class ImageIntent:
    tail: Optional[str]
    model: Optional[str]
    view: Optional[ImageView]


@dataclass(frozen=True)
class ImageHit:
    title: str
    imageUrl: str
    source: str


@dataclass(frozen=True)
class ScoredImage:
    imageUrl: str
    title: str
    source: str
    score: float


_VIEW_SYNONYMS: Tuple[Tuple[ImageView, Tuple[str, ...]], ...] = (
    ("cockpit", ("cockpit", "flight deck", "flightdeck")),
    ("bedroom", ("bedroom", "stateroom", "master suite", "suite")),
    ("interior", ("interior", "inside")),
    ("cabin", ("cabin", "cabin layout", "seat", "seating", "salon")),
    ("exterior", ("exterior", "outside", "ramp", "walkaround", "night")),
)

# Aviation-ish hosts get a small trust bump (not sufficient alone).
_AV_HOST_FRAGS: Tuple[str, ...] = (
    "jetphotos.",
    "planespotters.",
    "controller.com",
    "globalair.",
    "airliners.",
)

# Lightweight “other model” detector for penalty heuristics (not exhaustive).
_KNOWN_MODEL_MARKERS: Tuple[str, ...] = (
    "gulfstream",
    "global ",
    "challenger",
    "falcon",
    "citation",
    "phenom",
    "praetor",
    "learjet",
    "king air",
    "pc-12",
    "hondajet",
    "vision jet",
)


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def normalize_tail_token(raw: str) -> str:
    return (raw or "").strip().upper().replace(" ", "")


def infer_view_from_text(user_input: str) -> Optional[ImageView]:
    low = (user_input or "").lower()
    for view, syns in _VIEW_SYNONYMS:
        if any(w in low for w in syns):
            return view
    return None


def _safe_json_obj(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def _openai_extract_intent_json(user_input: str) -> Optional[Dict[str, Any]]:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return None

    model = (os.getenv("OPENAI_IMAGE_INTENT_MODEL") or "gpt-4o-mini").strip()
    prompt = (
        "You extract structured aviation image intent.\n\n"
        f'User input:\n"{user_input.replace(chr(34), chr(39))}"\n\n'
        "Return JSON:\n"
        "{\n"
        '  "tail": string | null,\n'
        '  "model": string | null,\n'
        '  "view": one of [\"exterior\",\"cabin\",\"cockpit\",\"interior\",\"bedroom\"] | null\n'
        "}\n\n"
        "Rules:\n"
        "* Normalize tail numbers to uppercase\n"
        "* Infer model if possible (e.g. g650 -> Gulfstream G650)\n"
        "* Map synonyms: inside -> cabin, flight deck -> cockpit\n"
        "* Tail number overrides model if both exist\n"
        "* No explanation, only JSON\n"
    )

    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "temperature": 0,
                "max_tokens": 120,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
            timeout=20.0,
        )
        r.raise_for_status()
        data = r.json()
        text = str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        return _safe_json_obj(text)
    except Exception as e:
        logger.warning("premium_image_pipeline: OpenAI intent extraction failed: %s", e)
        return None


def _heuristic_intent(user_input: str) -> ImageIntent:
    """
    Deterministic fallback when OpenAI is unavailable.
    Conservative: only extracts obvious US N-tails; model inference is minimal.
    """
    blob = user_input or ""
    m = re.search(r"\bN[1-9A-Z][A-Z0-9]{1,5}\b", blob, flags=re.I)
    tail = normalize_tail_token(m.group(0)) if m else None

    # Minimal shorthand model inference (normalize_aircraft_name handles common tokens).
    model_guess = None
    upper = blob.upper()
    if any(k in upper for k in ("G650", "G700", "G550", "G500", "G600", "G280", "CL350", "GLOBAL", "FALCON", "PHENOM")):
        model_guess = normalize_aircraft_name(blob)

    view = infer_view_from_text(blob)
    if tail:
        return ImageIntent(tail=tail, model=None, view=view)
    if model_guess:
        return ImageIntent(tail=None, model=model_guess, view=view)
    return ImageIntent(tail=None, model=None, view=view)


async def extract_image_intent(user_input: str) -> ImageIntent:
    """
    LLM-first intent extraction with a deterministic fallback.

    Note: async API matches the requested contract; the implementation uses sync HTTP
    under the hood (fine for typical server runtimes).
    """
    obj = _openai_extract_intent_json(user_input or "")
    if not obj:
        return _heuristic_intent(user_input)

    tail_raw = obj.get("tail")
    model_raw = obj.get("model")
    view_raw = obj.get("view")

    tail = normalize_tail_token(str(tail_raw)) if tail_raw else None
    model = normalize_aircraft_name(str(model_raw).strip()) if model_raw else None
    view_s = str(view_raw).strip().lower() if view_raw else ""
    view: Optional[ImageView] = view_s if view_s in ("exterior", "cabin", "cockpit", "interior", "bedroom") else None
    view = view or infer_view_from_text(user_input)

    # Tail overrides model.
    if tail:
        return ImageIntent(tail=tail, model=None, view=view)
    if model:
        return ImageIntent(tail=None, model=model, view=view)
    return ImageIntent(tail=None, model=None, view=view)


def build_image_query(intent: ImageIntent) -> str:
    """
    Max 5 words, no punctuation, no filler.
    """
    view = intent.view
    if intent.tail:
        tail = normalize_tail_token(intent.tail)
        second = view if view else "aircraft"
        q = f"{tail} {second}"
    elif intent.model:
        mdl = _norm_ws(intent.model)
        second = view if view else "cabin"
        q = f"{mdl} {second}"
    else:
        q = f"{view if view else 'private jet cabin'}"

    q = re.sub(r"[^A-Za-z0-9\s]", " ", q)
    q = _norm_ws(q)
    words = q.split()
    return " ".join(words[:5])


def search_images(query: str) -> List[ImageHit]:
    """
    SearchAPI image search (Google Images / Light or Bing). Returns at most 8 minimal hits.
    """
    q = (query or "").strip()
    if not q:
        return []

    params = build_searchapi_image_request_params(q=q, num_results=8)
    if not params:
        return []
    try:
        r = requests.get(SEARCHAPI_SEARCH_URL, params=params, timeout=28.0)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning("premium_image_pipeline: SearchAPI request failed: %s", e)
        return []

    images = data.get("images") if isinstance(data, dict) else None
    if not isinstance(images, list):
        return []

    try:
        limit = int(params.get("num") or 8)
    except (TypeError, ValueError):
        limit = 8
    limit = max(1, min(20, limit))

    out: List[ImageHit] = []
    for item in images[:limit]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        orig = item.get("original") if isinstance(item.get("original"), dict) else {}
        url = str((orig or {}).get("link") or "").strip()
        if not title or not url.startswith("https://"):
            continue
        src = item.get("source") if isinstance(item.get("source"), dict) else {}
        src_name = str((src or {}).get("name") or "").strip()
        src_link = str((src or {}).get("link") or "").strip()
        host = ""
        try:
            host = (urlparse(url).netloc or "").lower()
        except Exception:
            host = ""
        source_label = src_name or host or "web"
        # Intentionally ignore large fields (snippet/html/etc.).
        _ = src_link  # keep variable to avoid unused warnings in some linters
        out.append(ImageHit(title=title, imageUrl=url, source=source_label))
    return out


def _aviation_source_bonus(url: str, source: str) -> float:
    blob = f"{url} {source}".lower()
    return 0.2 if any(x in blob for x in _AV_HOST_FRAGS) else 0.0


def _view_match_bonus(title: str, view: Optional[ImageView]) -> float:
    if not view:
        return 0.0
    low = (title or "").lower()
    for v, syns in _VIEW_SYNONYMS:
        if v != view:
            continue
        return 0.1 if any(s in low for s in syns) else 0.0
    return 0.0


def _title_has_other_tail(title: str, expected: str) -> bool:
    exp = normalize_tail_token(expected)
    for m in re.finditer(r"\bN[1-9A-Z][A-Z0-9]{1,5}\b", title or "", flags=re.I):
        t = normalize_tail_token(m.group(0))
        if t and t != exp:
            return True
    return False


def _title_variant_bonus(title: str, model: str) -> float:
    """
    Small bonus if title contains a common variant token beyond the canonical model string.
    """
    low = (title or "").lower()
    ml = (model or "").lower()
    if "g650er" in low and "650er" not in ml:
        return 0.2
    if "g650" in low and "g650er" in ml:
        return 0.1
    return 0.0


def _title_other_model_penalty(title: str, intent_model: str) -> float:
    """
    Penalize obvious mentions of a different aircraft family than the requested model.
    Conservative: only applies when intent_model is present.
    """
    low = (title or "").lower()
    im = (intent_model or "").lower()
    if not im:
        return 0.0

    # If title contains another strong marker not substring-overlapping with intent model.
    for marker in _KNOWN_MODEL_MARKERS:
        if marker in low and marker not in im:
            return 0.5
    return 0.0


def score_image(result: ImageHit, intent: ImageIntent) -> float:
    """
    Returns 0..1 score.

    This is intentionally heuristic but strict on tail mismatch.
    """
    title = result.title or ""
    url = result.imageUrl or ""
    src = result.source or ""

    score = 0.0
    if intent.tail:
        t = normalize_tail_token(intent.tail)
        if t and t in title.upper():
            score += 0.7
        score += _aviation_source_bonus(url, src)
        score += _view_match_bonus(title, intent.view)
        if _title_has_other_tail(title, t):
            score -= 0.7
        score -= _title_other_model_penalty(title, intent.model or "")
    elif intent.model:
        m = (intent.model or "").strip()
        if not m:
            return 0.0
        ml = m.lower()
        if ml in (title or "").lower():
            score += 0.5
        score += _title_variant_bonus(title, m)
        score += _aviation_source_bonus(url, src)
        score += _view_match_bonus(title, intent.view)
        score -= _title_other_model_penalty(title, m)
    else:
        # No anchor: do not guess aircraft identity from generic queries.
        return 0.0

    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (title or "").lower())


def filter_and_rank_images(results: List[ImageHit], intent: ImageIntent) -> List[ScoredImage]:
    scored: List[ScoredImage] = []
    for r in results or []:
        s = score_image(r, intent)
        if s < 0.6:
            continue
        scored.append(ScoredImage(imageUrl=r.imageUrl, title=r.title, source=r.source, score=s))

    scored.sort(key=lambda x: x.score, reverse=True)

    deduped: List[ScoredImage] = []
    seen: set[str] = set()
    for it in scored:
        k = _title_key(it.title)
        if not k or k in seen:
            continue
        seen.add(k)
        deduped.append(it)
        if len(deduped) >= 3:
            break
    return deduped


async def get_premium_aircraft_images(user_input: str) -> Dict[str, Any]:
    intent = await extract_image_intent(user_input)
    query = build_image_query(intent)
    raw = search_images(query)

    ranked = filter_and_rank_images(raw, intent)
    best = ranked[0].score if ranked else 0.0
    if not ranked or best < 0.7:
        return {
            "success": False,
            "message": "I couldn’t find verified images for this specific aircraft.",
        }

    return {
        "success": True,
        "images": [
            {"imageUrl": x.imageUrl, "title": x.title, "source": x.source, "score": x.score} for x in ranked
        ],
    }
