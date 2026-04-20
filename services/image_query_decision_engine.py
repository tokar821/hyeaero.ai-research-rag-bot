"""
Strict retrieval decision engine: ultra-precise Google Image ``q`` strings (no LLM).

Output shape: ``{"queries": ["...", ...]}`` — each query ≤6 words, identity + facet + quality cues,
no banned generic tokens (jet, plane, aircraft, …).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from rag.aviation_tail import normalize_tail_token

from services.searchapi_aircraft_images import (
    compose_manufacturer_model_phrase,
    detect_query_image_intent,
    normalize_aircraft_name,
)

# Whole-word ban (case-insensitive). Do not include tokens that appear inside legitimate model names.
_BANNED_WORDS = frozenset(
    {
        "jet",
        "jets",
        "plane",
        "planes",
        "aircraft",
        "airplane",
        "airplanes",
        "airliners",
        "airliner",
        "photo",
        "photos",
        "picture",
        "pictures",
        "image",
        "images",
        "pic",
        "pics",
        "gallery",
        "private",
        "flying",
        "show",
        "showing",
        "look",
        "see",
        "some",
        "any",
        "the",
        "for",
        "with",
        "from",
        "about",
        "into",
        "your",
        "this",
        "that",
        "have",
        "need",
        "want",
    }
)

_ALLOWED_FACETS = frozenset({"cockpit", "cabin", "interior", "exterior"})

# Google Image ``q`` length cap (deterministic path + dedupe); allows e.g. ``… high resolution``.
_MAX_WORDS_PER_GOOGLE_IMAGE_Q = 6


def _ban_pattern() -> re.Pattern[str]:
    inner = "|".join(sorted(re.escape(w) for w in _BANNED_WORDS))
    return re.compile(rf"\b(?:{inner})\b", re.I)


_BAN_RE = _ban_pattern()

_ULTRA_CABIN_BROWSE_RE = re.compile(
    r"(?is)"
    r"(?:\b(?:best|top|nicest|finest|ultimate|greatest)\b.+\b(?:cabin|interior)\b)"
    r"|\b(?:best|top)\s+(?:private\s+)?jets?\s+cabin\b"
    r"|\bultra[\s-]*long[\s-]*range\s+(?:cabin|interior)\b"
    r"|\b(?:hotel|resort)\s+feel\b"
    r"|^\s*(?:premium|luxury)\s*$"
)


def query_violates_banned_terms(q: str) -> bool:
    return bool(_BAN_RE.search((q or "").strip()))


def _word_cap(s: str, max_words: int = _MAX_WORDS_PER_GOOGLE_IMAGE_Q) -> str:
    parts = [p for p in (s or "").strip().split() if p]
    return " ".join(parts[:max_words]).strip()


def _is_ultra_long_range_cabin_discovery(
    user_low: str, intent: Dict[str, Any], mm: str, *, user_text: str
) -> bool:
    """Generic luxury / cabin / 'hotel feel' browse with no tail/model — fan out to **large-cabin** examples."""
    if str(intent.get("type") or "").upper() == "INVALID":
        return False
    if intent.get("suppress_image_search"):
        return False
    if normalize_tail_token(str(intent.get("tail_number") or "").strip()):
        return False
    # Do not let incidental model hints from retrieval context (mm_for_scoring / phly rows) disable
    # browse fan-out. Only treat this as "model specified" when the **user text** names a model.
    try:
        from rag.consultant_query_expand import _detect_models

        if _detect_models((user_text or "").strip()):
            return False
    except Exception:
        if len((mm or "").strip()) >= 3:
            return False
    if not _ULTRA_CABIN_BROWSE_RE.search(user_low):
        return False
    if str(intent.get("image_type") or "").lower() in ("exterior", "cockpit"):
        return False
    facets = intent.get("image_facets")
    if isinstance(facets, list) and any(str(x).lower() == "cockpit" for x in facets):
        return False
    return True


def _uniq_cap(qs: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for q in qs:
        qn = _word_cap(q, _MAX_WORDS_PER_GOOGLE_IMAGE_Q)
        if len(qn) < 2 or query_violates_banned_terms(qn):
            continue
        # Every query must contain at least one allowed facet token as a whole word.
        low = f" {qn.lower()} "
        if not any(f" {f} " in low for f in _ALLOWED_FACETS):
            continue
        k = qn.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(qn)
    return out[:5]


def _resolve_model_marketing(
    intent: Dict[str, Any],
    *,
    user_text: str,
    required_marketing_type: Optional[str],
    phly_rows: Optional[List[Dict[str, Any]]],
    mm_for_scoring: Optional[str],
) -> str:
    # For "best cabin / luxury / hotel feel" browse queries, do not let unrelated context rows
    # (Phly/SQL/MM-for-scoring) force a specific model. We want a multi-aircraft fan-out.
    raw = (user_text or "").strip()
    low = raw.lower()
    if (
        _ULTRA_CABIN_BROWSE_RE.search(low)
        and not normalize_tail_token(str(intent.get("tail_number") or "").strip())
        and not (required_marketing_type or "").strip()
        and not str(intent.get("aircraft") or "").strip()
    ):
        try:
            from rag.consultant_query_expand import _detect_models

            mdls = _detect_models(raw)
        except Exception:
            mdls = []
        if not mdls:
            mm_for_scoring = None
            phly_rows = None

    mm = (
        (required_marketing_type or "").strip()
        or str(intent.get("aircraft") or "").strip()
        or (mm_for_scoring or "").strip()
    )
    if not mm and phly_rows:
        for r in (phly_rows or [])[:4]:
            man = (r.get("manufacturer") or "").strip()
            mdl = (r.get("model") or "").strip()
            if man or mdl:
                mm = compose_manufacturer_model_phrase(man, mdl)
                break
    if not mm:
        try:
            from rag.consultant_query_expand import _detect_manufacturers, _detect_models

            mans = _detect_manufacturers(low)
            mdls = _detect_models(raw)
            mm = compose_manufacturer_model_phrase(
                mans[0] if mans else "",
                mdls[0] if mdls else "",
            )
        except Exception:
            mm = ""
    return normalize_aircraft_name(mm.strip()) if mm else ""


def _prepend_searchapi_high_recall_queries(identity: str, facet: str) -> List[str]:
    """
    Short ``{token} {facet}`` strings that mirror high-performing manual Google Image searches
    (e.g. ``G650 interior``) before longer marketing-style variants.
    """
    i = (identity or "").strip()
    if not i or facet not in _ALLOWED_FACETS:
        return []
    low = i.lower()
    q: List[str] = []

    m = re.search(r"\bg\s*[-.]?\s*(\d{3,4})(?:\s*er)?\b", i, re.I)
    if m:
        g = f"G{m.group(1)}"
        if facet in ("cabin", "interior"):
            q.extend([f"{g} interior", f"{g} cabin"])
        elif facet == "cockpit":
            q.extend([f"{g} cockpit", f"{g} cockpit view"])
        elif facet == "exterior":
            q.extend([f"{g} exterior"])

    m = re.search(r"\bglobal\s*(\d{4})\b", low)
    if m:
        gl = f"Global {m.group(1)}"
        if facet in ("cabin", "interior"):
            q.extend([f"{gl} interior", f"{gl} cabin"])
        elif facet == "cockpit":
            q.extend([f"{gl} cockpit"])
        elif facet == "exterior":
            q.extend([f"{gl} exterior"])

    m = re.search(r"\bfalcon\s*(\d{3,4})\b", low)
    if m:
        falc = f"Falcon {m.group(1)}"
        if facet in ("cabin", "interior"):
            q.extend([f"{falc} interior", f"{falc} cabin"])
        elif facet == "cockpit":
            q.extend([f"{falc} cockpit"])
        elif facet == "exterior":
            q.extend([f"{falc} exterior"])

    m = re.search(r"\bchallenger\s*(\d{3})\b", low)
    if m:
        ch = f"Challenger {m.group(1)}"
        if facet in ("cabin", "interior"):
            q.extend([f"{ch} interior", f"{ch} cabin"])
        elif facet == "cockpit":
            q.extend([f"{ch} cockpit"])
        elif facet == "exterior":
            q.extend([f"{ch} exterior"])

    m = re.search(r"\bcj\s*(\d+[a-z]?)\b", i, re.I)
    if m:
        cj = f"CJ{m.group(1).upper()}"
        if facet in ("cabin", "interior"):
            q.extend([f"{cj} interior", f"{cj} cabin"])
        elif facet == "cockpit":
            q.extend([f"{cj} cockpit"])
        elif facet == "exterior":
            q.extend([f"{cj} exterior"])

    m = re.search(r"\bphenom\s*(100|300|500)\b", low)
    if m:
        ph = f"Phenom {m.group(1)}"
        if facet in ("cabin", "interior"):
            q.extend([f"{ph} interior", f"{ph} cabin"])
        elif facet == "cockpit":
            q.extend([f"{ph} cockpit"])
        elif facet == "exterior":
            q.extend([f"{ph} exterior"])

    m = re.search(r"\blearjet\s*(\d{2,3})\b", low)
    if m:
        lj = f"Learjet {m.group(1)}"
        if facet in ("cabin", "interior"):
            q.extend([f"{lj} interior", f"{lj} cabin"])
        elif facet == "cockpit":
            q.extend([f"{lj} cockpit"])
        elif facet == "exterior":
            q.extend([f"{lj} exterior"])

    m = re.search(r"\bking\s*air\s*([a-z]?\d{2,4})\b", low)
    if m:
        ka = f"King Air {m.group(1).upper()}"
        if facet in ("cabin", "interior"):
            q.extend([f"{ka} interior", f"{ka} cabin"])
        elif facet == "cockpit":
            q.extend([f"{ka} cockpit"])
        elif facet == "exterior":
            q.extend([f"{ka} exterior"])

    m = re.search(r"\bpc[\s-]?(12|24)\b", low)
    if m:
        pc = f"PC-{m.group(1)}"
        if facet in ("cabin", "interior"):
            q.extend([f"{pc} interior", f"PC{m.group(1)} interior"])
        elif facet == "cockpit":
            q.extend([f"{pc} cockpit"])
        elif facet == "exterior":
            q.extend([f"{pc} exterior"])

    if "citation" in low and "latitude" in low:
        if facet in ("cabin", "interior"):
            q.extend(["Citation Latitude interior", "Latitude interior"])
        elif facet == "cockpit":
            q.extend(["Citation Latitude cockpit"])
        elif facet == "exterior":
            q.extend(["Citation Latitude exterior"])
    if "citation" in low and "longitude" in low:
        if facet in ("cabin", "interior"):
            q.extend(["Citation Longitude interior", "Longitude interior"])
        elif facet == "cockpit":
            q.extend(["Citation Longitude cockpit"])
        elif facet == "exterior":
            q.extend(["Citation Longitude exterior"])
    if re.search(r"\bcitation\s+x\b", low):
        if facet in ("cabin", "interior"):
            q.extend(["Citation X interior", "C750 interior"])
        elif facet == "cockpit":
            q.extend(["Citation X cockpit"])
        elif facet == "exterior":
            q.extend(["Citation X exterior"])

    return q


def _pin_compact_google_image_queries_first(
    qs: List[str],
    *,
    user_low: str,
    model_identity: str,
    intent_image_type: str,
    intent_facets: Optional[List[Any]] = None,
    canonical_tail: str = "",
) -> List[str]:
    """
    Move compact OEM-style queries (e.g. ``G650 interior``) to the front when the user (or intent)
    clearly cares about that facet, so SearchAPI's first ``q`` matches high-recall Google Image
    searches instead of a longer ``Gulfstream G650 exterior`` discovery default.
    """
    if not qs or not (model_identity or "").strip() or len((model_identity or "").strip()) < 2:
        return qs
    ct = normalize_tail_token((canonical_tail or "").strip())
    if ct and len(ct) >= 3:
        head = qs[: min(6, len(qs))]
        if head and all(re.search(rf"\b{re.escape(ct)}\b", (q or ""), re.I) for q in head):
            return qs
    mm = (model_identity or "").strip()
    facets_l = [
        str(x).strip().lower()
        for x in (intent_facets or [])
        if isinstance(x, (str, int)) and str(x).strip()
    ]
    want: Optional[str] = None
    it = (intent_image_type or "").strip().lower()
    if it in ("interior", "cabin", "cockpit", "exterior"):
        want = "interior" if it == "interior" else it
    if not want and facets_l:
        for f in facets_l:
            if f in ("interior", "cabin", "cockpit", "exterior"):
                want = "interior" if f == "interior" else f
                break
    if not want:
        di = detect_query_image_intent(user_low)
        if di:
            want = di
    if not want and any(
        w in user_low for w in ("interior", "cabin", "inside", "salon", "seating", "layout")
    ):
        want = "interior" if re.search(r"\binterior\b", user_low) else "cabin"
    if not want and any(
        w in user_low for w in ("exterior", "outside", "ramp", "walkaround", "livery")
    ):
        want = "exterior"
    if not want:
        return qs
    pre = _prepend_searchapi_high_recall_queries(mm, want)
    if not pre:
        return qs
    pre_l = {p.lower() for p in pre}
    if len(qs) >= len(pre) and all((qs[i] or "").lower() == pre[i].lower() for i in range(len(pre))):
        return qs
    rest = [x for x in qs if (x or "").strip().lower() not in pre_l]
    return _uniq_cap(pre + rest)


def _build_query_variants(
    *,
    identity: str,
    facet: str,
    user_low: str,
) -> List[str]:
    """3–5 strings: identity + allowed facet(s), ≤6 words, no banned terms."""
    prelude = _prepend_searchapi_high_recall_queries(identity, facet)
    out: List[str] = []
    if not identity or facet not in _ALLOWED_FACETS:
        return _uniq_cap(prelude)

    if facet == "cockpit":
        out.extend(
            [
                f"{identity} cockpit",
                f"{identity} cockpit layout",
                f"{identity} cockpit seating",
            ]
        )
    elif facet == "exterior":
        out.extend(
            [
                f"{identity} exterior",
                f"{identity} exterior livery",
                f"{identity} exterior front",
            ]
        )
    elif facet == "interior":
        out.extend(
            [
                f"{identity} interior",
                f"{identity} interior layout",
                f"{identity} interior seating",
            ]
        )
    else:  # cabin
        want_interior = "interior" in user_low and "cabin" not in user_low
        if want_interior:
            out.extend(
                [
                    f"{identity} interior",
                    f"{identity} interior layout",
                    f"{identity} interior seating",
                ]
            )
        else:
            out.extend(
                [
                    f"{identity} cabin",
                    f"{identity} interior",
                    f"{identity} cabin interior",
                ]
            )

    return _uniq_cap(prelude + out)


def _default_discovery_facets(*, user_low: str) -> List[str]:
    """When the user did not name a facet, issue disciplined discovery queries (one per facet)."""
    return ["exterior", "cabin", "interior", "cockpit"]


def generate_ultra_precise_google_image_queries_json(
    user_input: str,
    *,
    required_tail: Optional[str] = None,
    required_marketing_type: Optional[str] = None,
    phly_rows: Optional[List[Dict[str, Any]]] = None,
    strict_tail_mode: bool = False,
    mm_for_scoring: Optional[str] = None,
    intent: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[str]]:
    """
    Decision engine: deterministic intent → 3–5 Google Image search strings.

    Returns ``{"queries": [...]}`` only (strict JSON-friendly dict).
    """
    raw = (user_input or "").strip()
    user_low = raw.lower()

    from services.consultant_image_search_orchestrator import classify_premium_aviation_intent

    intent = intent or classify_premium_aviation_intent(
        raw,
        required_tail=required_tail,
        required_marketing_type=required_marketing_type,
        phly_rows=phly_rows,
    )

    tail = normalize_tail_token(str(intent.get("tail_number") or required_tail or "").strip())
    if strict_tail_mode and not tail:
        tail = normalize_tail_token(str(required_tail or "").strip())

    mm = _resolve_model_marketing(
        intent,
        user_text=raw,
        required_marketing_type=required_marketing_type,
        phly_rows=phly_rows,
        mm_for_scoring=mm_for_scoring,
    )

    # Premium interior icons (no budget given): ensure flagship examples for "best jet interior".
    # Must run BEFORE generic "best cabin / luxury" browse fan-out.
    if re.search(r"(?i)\bbest\s+jet\s+interior\b|\bbest\s+jet\s+interiors\b", raw):
        return {
            "queries": _uniq_cap(
                [
                    "Global 7500 cabin interior high resolution",
                    "Gulfstream G700 cabin interior high resolution",
                    "Falcon 10X cabin interior luxury",
                    "Global 8000 cabin interior high resolution",
                    "Gulfstream G800 cabin interior high resolution",
                ]
            )
        }

    if _is_ultra_long_range_cabin_discovery(user_low, intent, mm, user_text=raw):
        # Expert browse: named models + cabin/interior + quality; no bare “cabin”; avoid non-aviation drift.
        # Budget-sensitive: under ~$15M should not default to ULR flagships.
        if re.search(r"(?i)\bunder\s*\$?\s*1[0-5]\s*m\b|\bunder\s*\$?\s*1[0-5]\s*million\b|\bunder\s*\$?\s*15,?000,?000\b", raw):
            return {
                "queries": _uniq_cap(
                    [
                        "Challenger 300 cabin interior high resolution",
                        "Citation Latitude interior modern cabin",
                        "Falcon 2000EX cabin interior luxury",
                        "Challenger 350 cabin interior high resolution",
                        "Legacy 450 cabin interior high resolution",
                    ]
                )
            }
        return {
            "queries": _uniq_cap(
                [
                    "Challenger 300 cabin interior high resolution",
                    "Citation Latitude interior modern cabin",
                    "Falcon 2000LXS luxury cabin interior",
                    "Challenger 650 cabin interior high resolution",
                    "Global 6000 cabin interior high resolution",
                ]
            )
        }

    if intent.get("suppress_image_search") or intent.get("type") == "INVALID":
        return {"queries": []}

    itype = str(intent.get("image_type") or "").strip().lower()
    if itype not in ("cockpit", "cabin", "exterior", "interior", ""):
        itype = ""

    facets_intent = intent.get("image_facets")
    if not isinstance(facets_intent, list):
        facets_intent = []

    qs: List[str] = []
    if tail and len(facets_intent) >= 2:
        qs = [_word_cap(f"{tail} {f}", _MAX_WORDS_PER_GOOGLE_IMAGE_Q) for f in facets_intent[:5]]
    elif mm and len(mm) >= 2 and len(facets_intent) >= 2:
        qs = []
        for f in facets_intent[:5]:
            f_l = str(f).strip().lower()
            if not f_l:
                continue
            if f_l in _ALLOWED_FACETS:
                qs.extend(_prepend_searchapi_high_recall_queries(mm, f_l))
            qs.append(_word_cap(f"{mm} {f_l}", _MAX_WORDS_PER_GOOGLE_IMAGE_Q))
    elif tail:
        identity = tail
        if not itype:
            for facet in _default_discovery_facets(user_low=user_low):
                qs.append(_word_cap(f"{identity} {facet}", _MAX_WORDS_PER_GOOGLE_IMAGE_Q))
        else:
            facet = itype
            if itype == "cabin" and "interior" in user_low and "cabin" not in user_low:
                facet = "interior"
            qs.extend(_build_query_variants(identity=identity, facet=facet, user_low=user_low))
    elif mm and len(mm) >= 2:
        identity = mm
        if not itype:
            for facet in _default_discovery_facets(user_low=user_low):
                qs.append(_word_cap(f"{identity} {facet}", _MAX_WORDS_PER_GOOGLE_IMAGE_Q))
        else:
            facet = itype
            if itype == "cabin" and "interior" in user_low and "cabin" not in user_low:
                facet = "interior"
            qs.extend(_build_query_variants(identity=identity, facet=facet, user_low=user_low))
    else:
        return {"queries": []}

    qs = _uniq_cap(qs)
    # Same-facet pad only when user named a facet but variants were dropped (e.g. word cap).
    if itype and len(qs) < 3 and tail:
        facet = itype
        if itype == "cabin" and "interior" in user_low and "cabin" not in user_low:
            facet = "interior"
        for extra in _build_query_variants(identity=tail, facet=facet, user_low=user_low):
            if extra not in qs:
                qs.append(extra)
            if len(qs) >= 3:
                break
        qs = _uniq_cap(qs)

    qs = _pin_compact_google_image_queries_first(
        qs,
        user_low=user_low,
        model_identity=mm,
        intent_image_type=itype,
        intent_facets=facets_intent,
        canonical_tail=tail,
    )
    return {"queries": qs[:5]}


def format_queries_json_response(user_input: str, **kwargs: Any) -> str:
    """Single JSON object string for prompts / logging (``queries`` only)."""
    import json

    payload = generate_ultra_precise_google_image_queries_json(user_input, **kwargs)
    return json.dumps(payload, ensure_ascii=False)
