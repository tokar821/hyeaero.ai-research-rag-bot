"""
Aviation Intent Normalizer — standalone intent JSON from the **current** user message.

Uses a small LLM pass when enabled; otherwise a deterministic heuristic so callers always
get the same schema. History is **only** for resolving the latest line (pronouns / recent
aircraft), not for inheriting unrelated topics.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from rag.consultant_market_lookup import wants_consultant_aircraft_images_in_answer
from rag.intent.aviation_classifier import classify_aviation_intent_detailed
from rag.intent.schemas import AviationIntent
from services.searchapi_aircraft_images import detect_query_image_intent

logger = logging.getLogger(__name__)

INTENT_TYPES = frozenset(
    {
        "aircraft_lookup",
        "cabin_search",
        "interior_visual",
        "comparison",
        "cockpit",
        "generic_visual",
    }
)
CATEGORIES = frozenset({"light jet", "midsize", "heavy", "ultra long range"})
VISUAL_FOCUS = frozenset({"interior", "exterior", "cockpit", "cabin", "bedroom", "galley"})

_AVIATION_INTENT_NORMALIZER_SYSTEM = """You are the Aviation Intent Normalizer.

Your job is to IGNORE chat history noise and extract a clean, standalone aviation intent for the **current user query only**. Use conversation excerpt **only** to resolve pronouns (it, that one, them) or the **most recent aircraft** the user clearly refers to in the **latest** message.

OUTPUT (JSON object only, no markdown):
{
  "intent_type": "aircraft_lookup | cabin_search | interior_visual | comparison | cockpit | generic_visual",
  "aircraft": "exact model name string OR null",
  "category": "light jet | midsize | heavy | ultra long range | null",
  "visual_focus": "interior | exterior | cockpit | cabin | bedroom | galley | null",
  "constraints": {
    "budget": <number or null>,
    "style": "<short string or null>",
    "comparison_target": "<model string or null>"
  }
}

RULES:
- Prefer the **latest user message** semantics; do not let old unrelated turns change intent_type.
- When the user says **cabin** in a jet context, interpret as **aircraft cabin interior** — set **visual_focus** to **interior** (not generic "room"). If they ask cabin **dimensions/layout** without wanting photos, prefer **cabin_search**; if they want to **see** / **show** / **photos**, prefer **interior_visual**.
- Do **not** output ambiguous lone words like "cabin" as **intent_type**; always use one of the six intent_type literals.
- **"something like G650 but cheaper"** → **intent_type** aircraft_lookup or comparison as fits; **category** heavy or ultra long range; **comparison_target** "Gulfstream G650" (or G650); suggest **same class** alternatives mentally — do not downgrade to light jet.
- **cockpit** photos or layout → **intent_type** cockpit when cockpit is the main ask; else map **visual_focus** to cockpit.
- **generic_visual**: aviation images requested but facet unclear.
- **budget**: parse millions (e.g. under $12M → 12000000) as a number, or null if none.
- **style**: e.g. modern, luxury, minimal, ambient lighting — or null.
- Use **null** (JSON null) for unknown string fields, not the word "null".
"""


def default_normalized_aviation_intent() -> Dict[str, Any]:
    return {
        "intent_type": "aircraft_lookup",
        "aircraft": None,
        "category": None,
        "visual_focus": None,
        "constraints": {"budget": None, "style": None, "comparison_target": None},
    }


def _coerce_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("null", "none", "n/a", "unknown"):
        return None
    return s


def _coerce_budget(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        x = float(v)
        return x if x > 0 else None
    s = str(v).strip().replace(",", "")
    if not s:
        return None
    try:
        x = float(s)
        return x if x > 0 else None
    except ValueError:
        return None


def coerce_normalized_aviation_intent(raw: Any) -> Dict[str, Any]:
    """Validate and fix a parsed JSON object to the public schema."""
    out = default_normalized_aviation_intent()
    if not isinstance(raw, dict):
        return out
    it = _coerce_str(raw.get("intent_type"))
    if it and it in INTENT_TYPES:
        out["intent_type"] = it
    elif it:
        low = it.lower().replace(" ", "_")
        alias = {
            "aircraft": "aircraft_lookup",
            "lookup": "aircraft_lookup",
            "compare": "comparison",
            "vs": "comparison",
            "visual": "generic_visual",
            "image": "interior_visual",
            "photos": "interior_visual",
        }
        if low in alias:
            out["intent_type"] = alias[low]
    out["aircraft"] = _coerce_str(raw.get("aircraft"))
    cat = _coerce_str(raw.get("category"))
    if cat and cat.lower() in {c.lower() for c in CATEGORIES}:
        for c in CATEGORIES:
            if c.lower() == cat.lower():
                out["category"] = c
                break
    vf = _coerce_str(raw.get("visual_focus"))
    if vf and vf.lower() in {v.lower() for v in VISUAL_FOCUS}:
        for v in VISUAL_FOCUS:
            if v.lower() == vf.lower():
                out["visual_focus"] = v
                break
    cons = raw.get("constraints")
    if isinstance(cons, dict):
        out["constraints"]["budget"] = _coerce_budget(cons.get("budget"))
        out["constraints"]["style"] = _coerce_str(cons.get("style"))
        out["constraints"]["comparison_target"] = _coerce_str(cons.get("comparison_target"))
    return out


def _history_excerpt(history: Optional[List[Dict[str, str]]], *, max_messages: int = 10) -> str:
    lines: List[str] = []
    if not history:
        return ""
    for h in history[-max_messages:]:
        if not isinstance(h, dict):
            continue
        role = (h.get("role") or "user").strip().lower()
        c = (h.get("content") or "").strip()
        if not c or role not in ("user", "assistant"):
            continue
        lines.append(f"{role}: {c[:1200]}")
    return "\n".join(lines).strip()


def _budget_from_text(text: str) -> Optional[float]:
    low = (text or "").lower()
    m = re.search(
        r"(?:under|below|less\s+than|up\s+to|around|about|~)\s*\$?\s*([\d,.]+)\s*(m|mm|million)?",
        low,
        re.I,
    )
    if not m:
        m = re.search(r"\$\s*([\d,.]+)\s*(m|mm|million)?", low, re.I)
    if not m:
        return None
    try:
        num = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    suf = (m.group(2) or "").lower()
    if suf in ("m", "mm", "million"):
        return num * 1_000_000
    if num < 300 and re.search(r"\b(m|mm|million)\b", low):
        return num * 1_000_000
    if num < 300 and "$" in text:
        return num * 1_000_000
    return num


def normalize_aviation_intent_heuristic(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Fast path: current query + light history blob for compare/image cues."""
    q = (query or "").strip()
    out = default_normalized_aviation_intent()
    if not q:
        return out

    blob = (q + " " + _history_excerpt(history, max_messages=4)).lower()
    av = classify_aviation_intent_detailed(q, None)

    facet = detect_query_image_intent(q)
    wants_gallery = wants_consultant_aircraft_images_in_answer(q)

    if re.search(r"\b(compare|vs\.?|versus|or\s+better|which\s+is\s+better)\b", q, re.I):
        out["intent_type"] = "comparison"
    elif av.intent == AviationIntent.AIRCRAFT_COMPARISON:
        out["intent_type"] = "comparison"
    elif re.search(r"\bcockpit\b", q, re.I) and wants_gallery:
        out["intent_type"] = "cockpit"
        out["visual_focus"] = "cockpit"
    elif re.search(r"\bcockpit\b", q, re.I):
        out["intent_type"] = "cockpit"
        out["visual_focus"] = "cockpit"
    elif wants_gallery or (
        re.search(r"\b(show|see|photo|picture|image|gallery|look\s+like)\b", q, re.I) and re.search(
            r"\b(jet|aircraft|cabin|interior|g\d|falcon|global|challenger|citation|lear)\b",
            blob,
            re.I,
        )
    ):
        out["intent_type"] = "interior_visual"
        if facet == "cockpit":
            out["intent_type"] = "cockpit"
            out["visual_focus"] = "cockpit"
        elif facet == "exterior":
            out["visual_focus"] = "exterior"
        elif facet == "cabin" or re.search(r"\bcabin|interior|bedroom|galley\b", q, re.I):
            out["visual_focus"] = "interior"
        else:
            out["visual_focus"] = "interior"
    elif re.search(
        r"\bcabin\s+(height|width|length|volume|dimensions?|layout|size)\b|\bcabin\s+spec",
        q,
        re.I,
    ):
        out["intent_type"] = "cabin_search"
        out["visual_focus"] = "interior"
    elif facet == "cabin" and not wants_gallery:
        out["intent_type"] = "cabin_search"
        out["visual_focus"] = "interior"
    elif av.intent in (AviationIntent.AIRCRAFT_SPECS, AviationIntent.MISSION_FEASIBILITY):
        out["intent_type"] = "aircraft_lookup"

    b = _budget_from_text(q)
    if b is not None:
        out["constraints"]["budget"] = b

    if re.search(r"gulfstream\s*g?\s*650|g650\b", blob, re.I):
        out["constraints"]["comparison_target"] = "Gulfstream G650"
        out["category"] = "ultra long range"
    if re.search(r"global\s*7500|7500\b", blob, re.I) and out["constraints"]["comparison_target"] is None:
        out["constraints"]["comparison_target"] = "Bombardier Global 7500"

    return out


def normalize_aviation_intent_llm(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    *,
    api_key: str,
    model: str,
    timeout: float = 14.0,
) -> Optional[Dict[str, Any]]:
    if not (api_key or "").strip() or not (query or "").strip():
        return None
    try:
        import openai

        excerpt = _history_excerpt(history, max_messages=12)
        user_block = (
            "Conversation excerpt (for pronoun / recent-aircraft resolution only; "
            "the latest user message defines the task):\n"
            f"{excerpt or '(no prior messages)'}\n\n"
            "Current user query:\n"
            f"{query.strip()}"
        )
        client = openai.OpenAI(api_key=api_key.strip(), timeout=float(timeout))
        resp = client.chat.completions.create(
            model=(model or "gpt-4o-mini").strip(),
            messages=[
                {"role": "system", "content": _AVIATION_INTENT_NORMALIZER_SYSTEM},
                {"role": "user", "content": user_block},
            ],
            max_tokens=220,
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        return coerce_normalized_aviation_intent(data)
    except Exception as e:
        logger.debug("normalize_aviation_intent_llm failed: %s", e)
        return None


def normalize_aviation_intent(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    *,
    api_key: str = "",
    model: str = "",
    timeout: float = 14.0,
) -> Dict[str, Any]:
    """
    Return normalized intent JSON. Tries LLM when ``AVIATION_INTENT_NORMALIZER_LLM`` is on
    and ``api_key`` (or ``OPENAI_API_KEY``) is set; otherwise heuristic only.
    """
    key = (api_key or "").strip() or (os.getenv("OPENAI_API_KEY") or "").strip()
    mdl = (model or "").strip() or (os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip()
    env_on = (os.getenv("AVIATION_INTENT_NORMALIZER_LLM") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if env_on and key:
        llm_out = normalize_aviation_intent_llm(
            query, history, api_key=key, model=mdl, timeout=timeout
        )
        if llm_out is not None:
            return llm_out
    return normalize_aviation_intent_heuristic(query, history)
