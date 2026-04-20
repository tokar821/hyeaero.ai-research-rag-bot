"""
HyeAero.AI **image query engine** (optional LLM): high-precision Google Image ``q`` strings
from user intent — not chat, JSON-only contract.

Deterministic :mod:`image_query_decision_engine` remains; enable with ``SEARCHAPI_IMAGE_QUERY_LLM``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from services.aviation_image_query_engine_prompt import IMAGE_QUERY_ENGINE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AviationImageQueryEngineResult:
    """LLM image-query engine output (see ``IMAGE_QUERY_ENGINE_SYSTEM_PROMPT``)."""

    queries: List[str]
    confidence: float
    reasoning: str


def searchapi_image_query_llm_mode() -> str:
    """
    ``SEARCHAPI_IMAGE_QUERY_LLM``:

    - ``0`` / off: disabled
    - ``1`` / ``on`` / ``smart``: call when deterministic queries are empty; when the question is
      **vague** without an aircraft anchor; or when it is **complex** (budget caps, “similar but
      cheaper”, alternatives vs a model, long multi-constraint image asks).
    - ``empty`` / ``fallback``: call only when the deterministic engine returned no queries.
    - ``always`` / ``all``: always call and merge LLM queries ahead of deterministic ones.
    """
    return (os.getenv("SEARCHAPI_IMAGE_QUERY_LLM") or "0").strip().lower()


def searchapi_image_query_llm_enabled() -> bool:
    return searchapi_image_query_llm_mode() not in ("0", "false", "no", "off", "")


def should_run_image_query_llm(
    *,
    user_query: str,
    intent: Dict[str, Any],
    deterministic_queries: List[str],
    required_tail: Optional[str],
    required_marketing_type: Optional[str],
    mm_for_scoring: Optional[str],
) -> bool:
    if not searchapi_image_query_llm_enabled():
        return False
    mode = searchapi_image_query_llm_mode()
    qs = list(deterministic_queries or [])
    if mode in ("always", "all", "2"):
        return True
    if mode in ("empty", "fallback"):
        return not qs
    if not qs:
        return True
    # Deterministic decision engine already produced a full high-precision set (3–5). Do not
    # invoke the LLM just because the user used budget/superlatives — this avoids drift into
    # out-of-band classes (e.g. Falcon 8X for "under $12M").
    if len(qs) >= 3:
        joined = " ".join(qs).lower()
        if any(k in joined for k in ("high resolution", "cabin interior", "cockpit")):
            return False
    if _user_image_query_needs_llm_for_complexity(user_query, intent=intent):
        return True
    return _user_query_lacks_aviation_anchor_for_images(
        user_query,
        intent=intent,
        required_tail=required_tail,
        required_marketing_type=required_marketing_type,
        mm_for_scoring=mm_for_scoring,
    )


def _user_text_suggests_image_seek(low: str) -> bool:
    return bool(
        re.search(
            r"\b(cabin|interior|cockpit|exterior|inside|photo|photos|picture|pictures|images?|"
            r"show\s+me|just\s+show|let\s+me\s+see|gallery)\b",
            low,
            re.I,
        )
    )


def _user_image_query_needs_llm_for_complexity(user_query: str, *, intent: Dict[str, Any]) -> bool:
    raw = (user_query or "").strip()
    low = raw.lower()
    if len(low) < 8:
        return False

    itype = str(intent.get("image_type") or "").strip()
    wants_images = _user_text_suggests_image_seek(low) or bool(itype)
    if not wants_images:
        return False

    has_budget = bool(
        re.search(
            r"[\$£€]|"
            r"\b\d[\d,.\s]{0,12}\s*[kKmM]\b|"
            r"\b\d+\s*(million|mil)\b|"
            r"\b(under|below|less\s+than|up\s+to|around|about|between)\s+[\$£€]?\s*\d|"
            r"\b(budget|price|cost|asking|list\s+price)\b",
            low,
            re.I,
        )
    )
    has_comparison = bool(
        re.search(
            r"\b("
            r"cheaper|less\s+expensive|more\s+affordable|"
            r"similar\s+to|something\s+similar|"
            r"alternative\s+to|instead\s+of|"
            r"compared\s+to|comparison|compare|versus|\bvs\.?\b"
            r")\b",
            low,
            re.I,
        )
    )
    word_n = len(low.split())
    long_compound = word_n >= 14

    if has_budget or has_comparison:
        return True
    if long_compound:
        return True
    return False


def _user_query_lacks_aviation_anchor_for_images(
    user_query: str,
    *,
    intent: Dict[str, Any],
    required_tail: Optional[str],
    required_marketing_type: Optional[str],
    mm_for_scoring: Optional[str],
) -> bool:
    raw = (user_query or "").strip()
    low = raw.lower()
    if len(low) < 3:
        return False
    if (required_tail or "").strip() or str(intent.get("tail_number") or "").strip():
        return False
    if (required_marketing_type or "").strip() or str(intent.get("aircraft") or "").strip():
        return False
    if (mm_for_scoring or "").strip():
        return False
    if re.search(r"\bn[1-9]\w{0,5}\b", raw, re.I):
        return False
    if re.search(
        r"(?<![a-z0-9])(?:g|f|cl|cj|pc)[-\s]?\d{2,4}\b|\b(?:gulfstream|falcon|challenger|citation|"
        r"phenom|learjet|global\s*\d|king\s*air|pilatus|embraer|bombardier|cessna|airbus|boeing)\b",
        low,
        re.I,
    ):
        return False
    visual = bool(
        re.search(
            r"\b(cabin|interior|cockpit|exterior|inside|seats?|layout|galley|divan|berth)\b",
            low,
            re.I,
        )
    )
    if not visual:
        return False
    vague = bool(re.search(r"\b(best|nicest|top|luxur(y|ious)|beautiful|stunning|compare)\b", low, re.I))
    short = len(low.split()) <= 8
    return vague or short


def _llm_image_query_model() -> str:
    return (os.getenv("SEARCHAPI_IMAGE_QUERY_LLM_MODEL") or "").strip() or (
        os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini"
    ).strip()


def _llm_image_query_timeout_s() -> float:
    try:
        return max(4.0, min(45.0, float((os.getenv("SEARCHAPI_IMAGE_QUERY_LLM_TIMEOUT") or "14").strip())))
    except ValueError:
        return 14.0


_LLM_INJECTION_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _llm_image_query_is_safe_and_aviation(q: str) -> bool:
    """
    Validate engine output: aviation anchor, visual facet, allow Google minus-terms,
    cap length for SSRF-safe ``q``.
    """
    qn = " ".join((q or "").strip().split())
    if len(qn) < 6 or len(qn) > 280:
        return False
    if _LLM_INJECTION_RE.search(qn):
        return False
    low = qn.lower()
    if any(x in low for x in ("javascript:", "data:", "<script")):
        return False
    parts = qn.split()
    if len(parts) > 18:
        return False
    # Prefer Google minus-terms (engine contract); allow pure tail+facet without minus if tail present
    has_minus = bool(re.search(r"-(?:house|home|hotel|airbnb|wood)", low))

    facet_pat = (
        r"\b(cabin|interior|cockpit|exterior|bedroom|lavatory|"
        r"real\s+photo|actual\s+interior|private\s+jet)\b"
    )
    has_facet = bool(re.search(facet_pat, low, re.I))
    has_tail = bool(re.search(r"\bn[1-9]\w{0,5}\b", qn, re.I))
    has_model_shorthand = bool(
        re.search(
            r"(?<![a-z0-9])(?:g|f|cl|cj|pc)[-\s]?\d|\b(?:gulfstream|falcon|challenger|citation|"
            r"phenom|learjet|eclipse|global\s*\d|king\s*air|pilatus|embraer|bombardier|cessna|dassault|"
            r"hawker|beechcraft|hondaJet|honda\s*jet|daher|sovereign|longitude|latitude|excel|xls|"
            r"ultra|encore|premier)\b",
            low,
            re.I,
        )
    )
    has_aviation = bool(
        re.search(
            r"\b(?:aviation|aircraft|airplane|aeroplane|bizjet|business\s*jet|private\s*jet|"
            r"corporate\s+jet|charter\s+jet|jet\s+cabin|airliner)\b",
            low,
            re.I,
        )
    )
    intent_qual = bool(
        re.search(
            r"\b(luxury|premium|modern|budget|midsize|super\s*midsize|light\s*jet|heavy|ultra|best|"
            r"nicest|cheaper|affordable|comparison|versus|under\s+\$|below\s+\$|"
            r"high\s+res(?:olution)?|\bhd\b)\b",
            low,
            re.I,
        )
    )
    if has_tail or has_model_shorthand:
        ok = has_facet or has_aviation
        if has_tail and ok:
            return True
        # Named model + visual section: product spec allows clean q strings without minus-terms.
        if has_model_shorthand and ok:
            return True
        return ok and has_minus
    # Vague browse (e.g. "best midsize jet cabin"): require aviation cue + facet + intent + length.
    if has_facet and has_aviation and intent_qual and len(parts) >= 5:
        return True
    return (has_facet and has_aviation) and has_minus


def _clamp_confidence(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, x))


def run_aviation_image_query_engine_llm(
    *,
    user_query: str,
    intent: Dict[str, Any],
    required_tail: Optional[str],
    required_marketing_type: Optional[str],
    mm_for_scoring: Optional[str],
) -> AviationImageQueryEngineResult:
    """
    Single OpenAI JSON call — :class:`AviationImageQueryEngineResult`.
    """
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        logger.debug("SEARCHAPI_IMAGE_QUERY_LLM: no OPENAI_API_KEY; skipping")
        return AviationImageQueryEngineResult([], 0.0, "no_openai_api_key")

    raw_u = (user_query or "").strip()
    if len(raw_u) < 2:
        return AviationImageQueryEngineResult([], 0.0, "empty_user_query")

    tail = (required_tail or "").strip() or str(intent.get("tail_number") or "").strip()
    model = (required_marketing_type or "").strip() or str(intent.get("aircraft") or "").strip()
    if not model:
        model = (mm_for_scoring or "").strip()
    itype = str(intent.get("image_type") or "").strip().lower()
    facets = intent.get("image_facets")
    facet_list: List[str] = []
    if isinstance(facets, list):
        facet_list = [str(x).strip() for x in facets[:6] if str(x).strip()]

    payload_ctx = {
        "user_query": raw_u[:4000],
        "known_tail": tail or None,
        "known_aircraft_model": model or None,
        "image_type_hint": itype or None,
        "image_facets": facet_list or None,
        "product": "HyeAero.AI — image query engine (Google Images q strings only)",
    }

    try:
        import openai

        client = openai.OpenAI(api_key=api_key, timeout=_llm_image_query_timeout_s())
        resp = client.chat.completions.create(
            model=_llm_image_query_model(),
            temperature=0.15,
            max_tokens=650,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": IMAGE_QUERY_ENGINE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Context JSON (follow it exactly):\n" + json.dumps(payload_ctx, ensure_ascii=False),
                },
            ],
        )
        txt = (resp.choices[0].message.content or "").strip()
        data = json.loads(txt)
    except Exception as e:
        logger.warning("SEARCHAPI_IMAGE_QUERY_LLM: call failed: %s", e)
        return AviationImageQueryEngineResult([], 0.0, "llm_parse_or_api_error")

    if not isinstance(data, dict):
        return AviationImageQueryEngineResult([], 0.0, "invalid_json_shape")

    raw_list = data.get("queries")
    if not isinstance(raw_list, list):
        return AviationImageQueryEngineResult([], _clamp_confidence(data.get("confidence")), "no_queries_array")

    conf = _clamp_confidence(data.get("confidence"))
    reasoning = str(data.get("reasoning") or "").strip()
    if len(reasoning) > 400:
        reasoning = reasoning[:397] + "..."

    out: List[str] = []
    seen: set[str] = set()
    for x in raw_list:
        s = " ".join(str(x or "").strip().split())
        if not s or s.lower() in seen:
            continue
        if not _llm_image_query_is_safe_and_aviation(s):
            continue
        seen.add(s.lower())
        out.append(s)
        if len(out) >= 5:
            break

    if out:
        logger.info(
            "SEARCHAPI_IMAGE_QUERY_LLM engine: %s queries confidence=%s",
            len(out),
            conf,
        )
    return AviationImageQueryEngineResult(out, conf, reasoning or "engine_ok")


def aviation_google_image_queries_from_llm(
    *,
    user_query: str,
    intent: Dict[str, Any],
    required_tail: Optional[str],
    required_marketing_type: Optional[str],
    mm_for_scoring: Optional[str],
) -> List[str]:
    """Backward-compatible: queries list only."""
    return run_aviation_image_query_engine_llm(
        user_query=user_query,
        intent=intent,
        required_tail=required_tail,
        required_marketing_type=required_marketing_type,
        mm_for_scoring=mm_for_scoring,
    ).queries
