"""
Premium aviation image search orchestration (intent → short queries → validation).

Spec: deterministic queries stay short (≤5 words). Optional **image query engine** (LLM) emits
longer high-precision Google ``q`` strings (OEM + facet + booster + ``-house`` negatives) with
confidence; SearchAPI follows Google rank + filters.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from rag.aviation_tail import is_invalid_placeholder_us_n_tail, normalize_tail_token

from services.searchapi_aircraft_images import (
    compose_manufacturer_model_phrase,
    normalize_aircraft_name,
    strip_domains,
)

PREMIUM_VERIFIED_IMAGE_FAILURE = (
    "I cannot find verified images for this specific request."
)


def searchapi_precision_queries_enabled() -> bool:
    """
    When true (default), literal SearchAPI mode uses 3–5 orchestrated short queries
    instead of one long user-string ``q``.

    Set ``SEARCHAPI_PRECISION_QUERIES=0`` to restore single-string literal behavior.
    """
    return (os.getenv("SEARCHAPI_PRECISION_QUERIES") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _word_cap(s: str, max_words: int = 5) -> str:
    parts = [p for p in (s or "").strip().split() if p]
    return " ".join(parts[:max_words]).strip()


def _uniq_nonempty(qs: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for q in qs:
        qn = _word_cap(q, 5)
        if len(qn) < 2 or qn.lower() in seen:
            continue
        seen.add(qn.lower())
        out.append(qn)
    return out[:5]


def _uniq_image_queries(qs: List[str], *, max_words: int, max_out: int = 5) -> List[str]:
    """Dedupe short Google ``q`` strings; allow longer strings for LLM engine (negatives, OEM names)."""
    seen: Set[str] = set()
    out: List[str] = []
    for q in qs:
        qn = _word_cap(q, max_words)
        if len(qn) < 2 or qn.lower() in seen:
            continue
        seen.add(qn.lower())
        out.append(qn)
    return out[:max_out]


_INVALID_MODEL_MARKERS = (
    r"\bfalcon\s*9000\b",
    r"\bg650\s*999\b",
    r"\bchallenger\s*9999\b",
)

_FACET_WORD_RE = re.compile(r"\b(exterior|cabin|cockpit|interior)\b", re.I)


def detect_ordered_image_facets(user_query: str) -> List[str]:
    """
    Distinct visual facet tokens in **left-to-right** order as they appear in the user text.

    Covers explicit words plus common synonyms not spelled as ``exterior`` / ``cockpit``.
    """
    low = (user_query or "").lower()
    seen: Set[str] = set()
    out: List[str] = []
    for m in _FACET_WORD_RE.finditer(low):
        w = m.group(1).lower()
        if w not in seen:
            seen.add(w)
            out.append(w)
    if re.search(r"\b(flight deck|flightdeck)\b", low) and "cockpit" not in seen:
        seen.add("cockpit")
        out.append("cockpit")
    if re.search(r"\b(ramp|walkaround|outside)\b", low) and "exterior" not in seen:
        seen.add("exterior")
        out.append("exterior")
    return out


def classify_premium_aviation_intent(
    user_query: str,
    *,
    required_tail: Optional[str],
    required_marketing_type: Optional[str],
    phly_rows: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Structured intent (deterministic). Fields align with product spec; extra keys are internal.
    """
    raw = (user_query or "").strip()
    low = raw.lower()

    out: Dict[str, Any] = {
        "type": "GENERAL",
        "aircraft": "",
        "tail_number": "",
        "image_type": "",
        "priority": "accuracy",
        "suppress_image_search": False,
        "validate_images": False,
    }

    if any(re.search(p, low) for p in _INVALID_MODEL_MARKERS):
        out["type"] = "INVALID"
        out["suppress_image_search"] = True
        return out

    comparison = any(
        tok in low
        for tok in (
            " vs ",
            " versus ",
            " compared to ",
            " compare ",
            "comparison",
        )
    )
    buy_hits = any(
        x in low
        for x in (
            "first jet",
            "first aircraft",
            "what jet",
            "what aircraft",
            "which jet",
            "which aircraft",
            "should i buy",
            "budget",
            "entry level",
            "pre-owned",
            "preowned",
        )
    )
    if comparison:
        out["type"] = "COMPARISON"
    elif buy_hits:
        out["type"] = "BUYING_ADVISORY"
        out["priority"] = "advisory"

    tail = normalize_tail_token(required_tail or "")
    if not tail:
        for m in re.finditer(r"\b(N\d{1,5}[A-Z]{0,2})\b", raw, re.I):
            tail = normalize_tail_token(m.group(1))
            if tail:
                break
    if tail:
        out["tail_number"] = tail
        out["aircraft"] = tail

    facets = detect_ordered_image_facets(raw)
    out["image_facets"] = facets
    if len(facets) >= 2:
        out["image_type"] = ""
    elif len(facets) == 1:
        out["image_type"] = facets[0]
    else:
        out["image_type"] = ""
        if any(w in low for w in ("cockpit", "flight deck", "flightdeck")):
            out["image_type"] = "cockpit"
        elif any(w in low for w in ("cabin", "salon", "seating", "layout", "interior", "inside")):
            out["image_type"] = "cabin"
        elif any(w in low for w in ("exterior", "outside", "ramp", "walkaround")):
            out["image_type"] = "exterior"

    if tail and is_invalid_placeholder_us_n_tail(tail):
        out["type"] = "INVALID"
        out["suppress_image_search"] = True
        out["invalid_registration"] = True
        out["validate_images"] = False
        return out

    visual = any(
        w in low
        for w in (
            "photo",
            "photos",
            "image",
            "images",
            "picture",
            "pictures",
            "show me",
            "show ",
            "see ",
            "gallery",
            "look at",
        )
    )

    if out["type"] not in ("INVALID", "COMPARISON", "BUYING_ADVISORY"):
        if tail and visual:
            out["type"] = "IMAGE_REQUEST"
        elif tail:
            out["type"] = "AIRCRAFT_LOOKUP"
        elif visual or out["image_type"]:
            out["type"] = "IMAGE_REQUEST"
        else:
            mm_hint = (required_marketing_type or "").strip()
            if not mm_hint and phly_rows:
                for r in (phly_rows or [])[:4]:
                    man = (r.get("manufacturer") or "").strip()
                    mdl = (r.get("model") or "").strip()
                    if man or mdl:
                        mm_hint = compose_manufacturer_model_phrase(man, mdl)
                        break
            if not mm_hint:
                try:
                    from rag.consultant_query_expand import _detect_manufacturers, _detect_models

                    mans = _detect_manufacturers(low)
                    mdls = _detect_models(raw)
                    mm_hint = compose_manufacturer_model_phrase(
                        mans[0] if mans else "",
                        mdls[0] if mdls else "",
                    )
                except Exception:
                    mm_hint = ""
            mm_hint = normalize_aircraft_name(mm_hint.strip()) if mm_hint else ""
            if mm_hint and len(mm_hint) >= 3:
                out["type"] = "AIRCRAFT_MODEL"
                out["aircraft"] = mm_hint

    if out["type"] == "BUYING_ADVISORY" and not (visual or out["image_type"] or tail):
        out["suppress_image_search"] = True

    out["validate_images"] = bool(
        out["type"] not in ("INVALID", "BUYING_ADVISORY", "COMPARISON", "GENERAL")
        or out["image_type"]
        or (tail and visual)
    )
    return out


def build_precision_image_search_queries(
    intent: Dict[str, Any],
    *,
    user_query: str,
    strict_tail_mode: bool,
    required_tail: Optional[str],
    required_marketing_type: Optional[str],
    phly_rows: Optional[List[Dict[str, Any]]],
    mm_for_scoring: Optional[str],
) -> Tuple[List[str], Dict[str, Any]]:
    """
    3–5 SearchAPI ``q`` strings from the decision engine, optionally merged with the
    **HyeAero image query engine** (LLM). Second return value may include
    ``image_query_engine`` (confidence, reasoning, suppress_gallery).
    """
    from services.consultant_aviation_image_query_llm import (
        run_aviation_image_query_engine_llm,
        should_run_image_query_llm,
    )
    from services.image_query_decision_engine import generate_ultra_precise_google_image_queries_json

    meta_out: Dict[str, Any] = {}

    payload = generate_ultra_precise_google_image_queries_json(
        user_query,
        required_tail=required_tail,
        required_marketing_type=required_marketing_type,
        phly_rows=phly_rows,
        strict_tail_mode=strict_tail_mode,
        mm_for_scoring=mm_for_scoring,
        intent=intent,
    )
    qs = list(payload.get("queries") or [])

    if should_run_image_query_llm(
        user_query=user_query,
        intent=intent,
        deterministic_queries=qs,
        required_tail=required_tail,
        required_marketing_type=required_marketing_type,
        mm_for_scoring=mm_for_scoring,
    ):
        try:
            eng = run_aviation_image_query_engine_llm(
                user_query=user_query,
                intent=intent,
                required_tail=required_tail,
                required_marketing_type=required_marketing_type,
                mm_for_scoring=mm_for_scoring,
            )
        except Exception:
            eng = None
        if eng is not None:
            meta_out["image_query_engine"] = {
                "confidence": float(eng.confidence),
                "reasoning": eng.reasoning,
            }
            llm_qs = list(eng.queries or [])
            if float(eng.confidence) < 0.7:
                merged = _uniq_image_queries(qs, max_words=5, max_out=5)
                meta_out["image_query_engine"]["llm_queries_suppressed_low_confidence"] = True
                if merged:
                    return merged, meta_out
                meta_out["image_query_engine"]["suppress_gallery"] = True
                return [], meta_out
            if llm_qs:
                merged = _uniq_image_queries(llm_qs + qs, max_words=14, max_out=5)
                if merged:
                    meta_out["image_query_engine"]["suppress_gallery"] = False
                    return merged, meta_out

    if qs:
        return qs, meta_out
    # Last resort: compressed user text only if it passes the same banned-token gate.
    if intent.get("suppress_image_search") or intent.get("type") == "INVALID":
        return [], meta_out
    from services.image_query_decision_engine import query_violates_banned_terms

    fallback = _word_cap(" ".join((user_query or "").strip().split()), 5)
    if len(fallback) >= 3 and not query_violates_banned_terms(fallback):
        return _uniq_nonempty([fallback]), meta_out
    return [], meta_out


def _premium_facet_matches_blob(blob: str, blob_u: str, tail: str, facet: str) -> bool:
    """Whether ``blob`` supports a single requested visual facet (used alone or in multi-facet OR)."""
    f = (facet or "").strip().lower()
    if f == "cockpit":
        if not any(
            x in blob
            for x in (
                "cockpit",
                "flight deck",
                "flightdeck",
                "flight-deck",
                "instrument panel",
                "glass cockpit",
                "avionics",
                "fms",
                "pfd",
                "mfd",
                "throttle",
                "pedestal",
                "overhead panel",
            )
        ):
            return False
        if any(
            x in blob
            for x in (
                "cabin layout",
                "cabin interior",
                "seating layout",
                "divan",
                "galley",
                "club seating",
                "berthing",
            )
        ) and "cockpit" not in blob and "flight deck" not in blob and "flightdeck" not in blob:
            return False
        return True
    if f in ("cabin", "interior"):
        cabinish = any(
            x in blob
            for x in (
                "cabin",
                "interior",
                "salon",
                "seating",
                "seats",
                "layout",
                "inside",
            )
        )
        if cabinish:
            return True
        if not tail:
            return False
        if any(
            x in blob
            for x in (
                "exterior",
                "ramp",
                "takeoff",
                "landing",
                "approach",
                "taxiing",
            )
        ):
            return False
        return True
    if f == "exterior":
        return any(
            x in blob
            for x in (
                "exterior",
                "ramp",
                "taxi",
                "takeoff",
                "landing",
                "approach",
                "parked",
                "airborne",
                "in flight",
            )
        )
    return False


def premium_image_row_passes_validation(row: Dict[str, Any], intent: Dict[str, Any]) -> bool:
    """Title/page/url relevance vs tail, model facet, and requested image type."""
    if intent.get("type") == "INVALID":
        return False

    title = str(row.get("title") or "")
    snippet = str(row.get("snippet") or "")
    page = str(row.get("_source_page") or "")
    url = str(row.get("url") or "")
    blob = f"{title} {snippet} {page} {url}".lower()
    blob_u = strip_domains(f"{title} {snippet} {page} {url}").upper()

    tail = normalize_tail_token(str(intent.get("tail_number") or "").strip())
    if tail:
        if tail.upper() not in blob_u and tail.lower() not in blob:
            return False

    facets = intent.get("image_facets")
    if isinstance(facets, list) and len(facets) > 1:
        return any(_premium_facet_matches_blob(blob, blob_u, tail, str(x)) for x in facets)

    itype = str(intent.get("image_type") or "").strip().lower()
    if itype and not _premium_facet_matches_blob(blob, blob_u, tail, itype):
        return False

    ac = str(intent.get("aircraft") or "").strip()
    if not tail and ac and len(ac) >= 3:
        low = blob
        parts = [p for p in re.split(r"\s+", ac) if len(p) >= 3]
        if parts:
            hits = sum(1 for p in parts if p.lower() in low)
            if hits < max(1, min(2, len(parts) // 2 or 1)):
                return False

    return True


def apply_premium_image_validation(
    rows: List[Dict[str, Any]], intent: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], bool]:
    """Returns (filtered_rows, had_any_input)."""
    if not intent.get("validate_images"):
        return rows, False
    kept = [r for r in rows if premium_image_row_passes_validation(r, intent)]
    return kept, True
