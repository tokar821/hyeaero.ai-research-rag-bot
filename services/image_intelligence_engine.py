"""
Aviation-grade Image Intelligence Engine (HyeAero.AI).

Deterministic pipeline: parse → resolve tail to canonical model (Phly / FAA when DB available) →
SearchAPI retrieval → per-image category + match_type → facet gates → tags → confidence → strict JSON.

Public entry points:
  - :func:`run_aircraft_image_intelligence` — ``{aircraft, image_type, images[{url, confidence, source, tags}], insight}``
  - :func:`run_image_intelligence` — legacy ``{aircraft, tail_number, images[{type, url, source, match_type, confidence}]}``

Does **not** invent URLs or aircraft identity; uncertain rows are dropped.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from rag.aviation_tail import normalize_tail_token

from services.consultant_aircraft_images import _model_tokens_match_strict
from services.consultant_image_search_orchestrator import classify_premium_aviation_intent
from services.image_query_decision_engine import generate_ultra_precise_google_image_queries_json
from services.searchapi_aircraft_images import (
    _all_registration_like_tokens,
    compose_manufacturer_model_phrase,
    extract_domain,
    fetch_ranked_searchapi_aircraft_images,
    normalize_aircraft_name,
    searchapi_aircraft_images_enabled,
    strip_domains as _strip_domains_blob,
)

logger = logging.getLogger(__name__)

_STOCK_NEEDLES = (
    "freepik",
    "shutterstock",
    "dreamstime",
    "adobe stock",
    "123rf",
    "istockphoto",
    "stock photo",
    "generative ai",
    "ai image",
)

# Higher = more trusted for brokerage-style sourcing (subset of SearchAPI domain bias).
_SOURCE_TIER: Tuple[Tuple[str, float], ...] = (
    ("jetphotos.", 0.08),
    ("planespotters.", 0.07),
    ("controller.com", 0.06),
    ("aircraftexchange", 0.05),
    ("globalair.", 0.04),
    ("airliners.", 0.04),
)


def _optional_db() -> Any:
    try:
        from api.main import get_db

        return get_db()
    except Exception:
        return None


def _faa_model_from_row(fr: Dict[str, Any]) -> str:
    mm = (fr.get("faa_reference_model") or "").strip()
    return normalize_aircraft_name(mm) if mm else ""


def _phly_model_from_row(r: Dict[str, Any]) -> str:
    mm = compose_manufacturer_model_phrase(
        str(r.get("manufacturer") or ""),
        str(r.get("model") or ""),
    )
    return normalize_aircraft_name(mm.strip()) if mm else ""


def resolve_aircraft_identity(
    *,
    tail: str,
    db: Any,
) -> Tuple[str, bool, str]:
    """
    Map tail → canonical marketing model string when authoritative.

    Returns ``(aircraft, authoritative, reason)`` where ``reason`` is empty on success.
    """
    if not tail:
        return "", False, "no_tail"
    if db is None:
        return "", False, "database_unavailable"

    models: List[str] = []
    try:
        from rag.phlydata_consultant_lookup import lookup_phlydata_aircraft_rows

        phly = lookup_phlydata_aircraft_rows(db, [tail])
        for r in (phly or [])[:4]:
            mm = _phly_model_from_row(r)
            if mm:
                models.append(mm)
    except Exception as e:
        logger.debug("Phly tail resolve skipped: %s", e)

    try:
        from services.faa_master_lookup import fetch_faa_master_owner_rows

        rows, _kind = fetch_faa_master_owner_rows(db, serial="", model=None, registration=tail)
        if rows:
            mm = _faa_model_from_row(rows[0])
            if mm:
                models.append(mm)
    except Exception as e:
        logger.debug("FAA tail resolve skipped: %s", e)

    uniq = list(dict.fromkeys(models))
    if len(uniq) > 1:
        return "", False, "ambiguous_model"
    if len(uniq) == 1:
        return uniq[0], True, ""
    return "", False, "unresolved_model"


def _blob_for_row(row: Dict[str, Any]) -> str:
    u = str(row.get("url") or "")
    t = str(row.get("description") or row.get("title") or "")
    p = str(row.get("page_url") or "")
    return f"{u} {t} {p}".lower()


def classify_visual_category(blob: str) -> str:
    """exterior | cabin | cockpit — heuristic from host metadata."""
    b = blob.lower()
    if any(x in b for x in ("cockpit", "flight deck", "flightdeck")):
        return "cockpit"
    if any(x in b for x in ("cabin", "interior", "salon", "seating", "galley", "layout")):
        return "cabin"
    if any(
        x in b
        for x in (
            "exterior",
            "ramp",
            "parked",
            "taxi",
            "landing",
            "takeoff",
            "approach",
            "walkaround",
        )
    ):
        return "exterior"
    return "cabin"


# (phrase in blob.lower(), tag label)
_VISUAL_TAG_NEEDLES: Tuple[Tuple[str, str], ...] = (
    ("club seating", "club seating"),
    ("club seat", "club seating"),
    ("divan", "divan"),
    ("berthing", "berthing"),
    ("galley", "galley"),
    ("lavatory", "lavatory"),
    ("shower", "shower"),
    ("fwd cabin", "fwd cabin"),
    ("aft cabin", "aft cabin"),
    ("forward cabin", "fwd cabin"),
    ("aft galley", "galley"),
    ("conference table", "conference table"),
    ("credenza", "credenza"),
    ("entertainment", "entertainment"),
    ("bulkhead", "bulkhead"),
    ("bedroom", "bedroom"),
    ("enclosed suite", "suite"),
    ("sky interior", "branded interior"),
    ("flight deck", "cockpit"),
)


def build_visual_image_tags(blob: str, category: str) -> List[str]:
    """Short brokerage tags from host metadata + coarse visual class."""
    low = (blob or "").lower().strip()
    tags: List[str] = []
    cat = (category or "").strip().lower()
    if cat in ("cabin", "cockpit", "exterior"):
        tags.append(cat)
    for needle, label in _VISUAL_TAG_NEEDLES:
        if needle in low and label not in tags:
            tags.append(label)
    return tags


def resolve_pipeline_image_type(intent: Dict[str, Any]) -> str:
    """Single ``image_type`` for API: cabin | cockpit | exterior | comparison."""
    if str(intent.get("type") or "").upper() == "COMPARISON":
        return "comparison"
    facets = intent.get("image_facets")
    if isinstance(facets, list) and len(facets) >= 2:
        return "comparison"
    it = str(intent.get("image_type") or "").strip().lower()
    if it == "interior":
        return "cabin"
    if it in ("cabin", "cockpit", "exterior"):
        return it
    return "cabin"


def _normalize_facet_token(t: str) -> str:
    x = (t or "").strip().lower()
    if x == "interior":
        return "cabin"
    return x if x in ("cabin", "cockpit", "exterior") else ""


def allowed_visual_categories_for_intent(intent: Dict[str, Any]) -> Optional[Set[str]]:
    """
    Categories permitted for this turn, or ``None`` = all (discovery).

    Strict single-facet requests drop wrong-facet frames entirely.
    """
    if str(intent.get("type") or "").upper() == "COMPARISON":
        return None
    facets_raw = intent.get("image_facets")
    facets: List[str] = []
    if isinstance(facets_raw, list):
        for f in facets_raw:
            n = _normalize_facet_token(str(f))
            if n:
                facets.append(n)
    it = _normalize_facet_token(str(intent.get("image_type") or ""))
    if len(facets) >= 2:
        return set(facets)
    if len(facets) == 1:
        return {facets[0]}
    if it:
        return {it}
    return None


def _model_only_fallback_intent(intent: Dict[str, Any], *, canonical_aircraft: str) -> Dict[str, Any]:
    """Strip tail so query engine emits model-keyed SearchAPI strings; keep facet hints."""
    i2 = copy.deepcopy(intent)
    i2["tail_number"] = ""
    i2["aircraft"] = (canonical_aircraft or "").strip()
    i2["type"] = "IMAGE_REQUEST"
    i2["suppress_image_search"] = False
    i2["validate_images"] = bool(i2.get("image_type") or i2.get("image_facets"))
    return i2


def _build_professional_insight(
    *,
    intent: Dict[str, Any],
    tail: str,
    aircraft_label: str,
    images: List[Dict[str, Any]],
    any_tail_match: bool,
    used_model_fallback: bool,
    pipeline_image_type: str,
    searchapi_on: bool,
    reason_early_exit: str,
) -> str:
    if reason_early_exit == "invalid":
        return (
            "That aircraft designation matches a blocked placeholder pattern; "
            "HyeAero does not attach gallery imagery to hypothetical or malformed type strings."
        )
    if reason_early_exit == "suppressed":
        return (
            "Image retrieval is suppressed for this request classification under brokerage safety rules; "
            "no external gallery was assembled."
        )
    if reason_early_exit == "ambiguous_model":
        return (
            "The tail resolves to conflicting canonical marketing types in the registry; "
            "visual retrieval is withheld until the aircraft identity is unambiguous."
        )
    if not searchapi_on:
        return (
            "Image search is not enabled in this environment, so no external gallery was retrieved. "
            "Enable SearchAPI credentials to activate the visual pipeline."
        )
    if reason_early_exit == "no_queries":
        return (
            "The request did not yield precision-safe image search strings under brokerage token rules; "
            "no gallery was assembled."
        )
    if not images:
        return (
            "No frames cleared tail/model cross-checks, stock filtering, and facet gates for this turn. "
            "Try a narrower facet (cabin vs cockpit vs exterior) or confirm the registration."
        )

    parts: List[str] = []
    if any_tail_match:
        parts.append(
            "Each frame ties to indexed metadata that references the requested registration, "
            "with model tokens checked against the resolved marketing type where available."
        )
    elif used_model_fallback and (tail or "").strip():
        parts.append(
            "Tail-specific public imagery that passed validation was sparse; "
            "the gallery is limited to representative shots of the resolved marketing model, "
            "with other registrations screened out."
        )
    elif aircraft_label:
        parts.append(
            "Imagery is constrained to the resolved marketing model with strict title, URL, and host checks."
        )
    else:
        parts.append("Imagery passed host and relevance filters for this request.")

    if pipeline_image_type in ("cabin", "cockpit"):
        parts.append(
            "Frames outside the requested passenger or flight-deck context were removed "
            "to protect cabin and cockpit accuracy."
        )
    elif pipeline_image_type == "exterior":
        parts.append("Non-ramp exterior and weakly-labeled cabin placeholders were deprioritized or removed.")

    return " ".join(parts)[:1200]


def _is_stock_or_non_aviation(blob: str) -> bool:
    return any(n in blob for n in _STOCK_NEEDLES)


def _domain_bonus(url: str) -> float:
    low = (url or "").lower()
    for frag, bonus in _SOURCE_TIER:
        if frag in low:
            return bonus
    return 0.0


def _other_tail_conflict(canonical_tail: str, blob: str) -> bool:
    canon = normalize_tail_token(canonical_tail)
    if not canon:
        return False
    blob_u = _strip_domains_blob(blob).upper()
    for t in _all_registration_like_tokens(blob or ""):
        nt = normalize_tail_token(t)
        if not nt or nt == canon:
            continue
        if nt in blob_u or nt.lower() in blob.lower():
            return True
    return False


def _match_type_and_confidence(
    *,
    row: Dict[str, Any],
    tail: str,
    canonical_aircraft: str,
    authoritative_model: bool,
) -> Tuple[str, float, bool]:
    """
    Returns ``(match_type, confidence, keep)``.

    ``keep`` is False when the row must be rejected (non-aviation, wrong tail, model bleed).
    """
    blob = _blob_for_row(row)
    url = str(row.get("url") or "")
    if not url.startswith("https://"):
        return "reject", 0.0, False
    if _is_stock_or_non_aviation(blob):
        return "reject", 0.0, False

    tail_n = normalize_tail_token(tail or "")
    tail_in = bool(tail_n) and (tail_n.lower() in blob or tail_n.upper() in _strip_domains_blob(blob).upper())

    if tail_n and _other_tail_conflict(tail_n, blob):
        return "reject", 0.0, False

    if authoritative_model and (canonical_aircraft or "").strip():
        if not _model_tokens_match_strict(blob, canonical_aircraft):
            if not tail_in:
                return "reject", 0.0, False

    bonus = _domain_bonus(url)

    if tail_in:
        return "tail_match", min(1.0, 0.92 + bonus), True
    if (canonical_aircraft or "").strip() and _model_tokens_match_strict(blob, canonical_aircraft):
        return "model_match", min(0.9, 0.72 + bonus), True
    if any(h in url.lower() for h, _ in _SOURCE_TIER):
        # Brokerage policy: do not ship anonymous “type” shots without tail or model evidence.
        return "generic_representative", 0.0, False

    return "reject", 0.0, False


def _sort_visual_candidates(
    candidates: List[Dict[str, Any]],
    *,
    allowed: Optional[Set[str]],
) -> List[Dict[str, Any]]:
    """Prefer correct facet, then tail match, then confidence."""

    def key(c: Dict[str, Any]) -> Tuple[int, int, float]:
        cat = str(c.get("type") or "")
        wrong_facet = 0
        if allowed is not None and cat not in allowed:
            wrong_facet = 2
        tail_pri = 0 if c.get("match_type") == "tail_match" else 1
        conf = float(c.get("confidence") or 0.0)
        return (wrong_facet, tail_pri, -conf)

    return sorted(candidates, key=key)


def _rows_to_visual_candidates(
    rows: List[Dict[str, Any]],
    *,
    tail: str,
    canonical: str,
    authoritative: bool,
    allowed_cats: Optional[Set[str]],
) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        url = str(row.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        blob = _blob_for_row(row)
        cat = classify_visual_category(blob)
        if allowed_cats is not None and cat not in allowed_cats:
            continue
        mt, conf, keep = _match_type_and_confidence(
            row=row,
            tail=tail,
            canonical_aircraft=canonical,
            authoritative_model=authoritative,
        )
        if not keep or mt == "reject" or conf <= 0:
            continue
        src = str(row.get("source") or extract_domain(url) or "web")
        tags = build_visual_image_tags(blob, cat)
        out.append(
            {
                "type": cat,
                "url": url,
                "source": src,
                "match_type": mt,
                "confidence": round(float(conf), 3),
                "tags": tags,
            }
        )
    return out


def _visual_intelligence_bundle(
    user_query: str,
    *,
    db: Any = None,
    max_images: int = 8,
) -> Dict[str, Any]:
    """
    Single fetch path returning V2 fields plus legacy images and ``tail_number``.

    Internal helper for :func:`run_image_intelligence` (avoids double SearchAPI).
    """
    raw = (user_query or "").strip()
    cap = max(1, min(24, int(max_images)))
    intent = classify_premium_aviation_intent(
        raw,
        required_tail=None,
        required_marketing_type=None,
        phly_rows=[],
    )
    pipeline_image_type = resolve_pipeline_image_type(intent)
    tail = normalize_tail_token(str(intent.get("tail_number") or ""))

    empty_legacy: List[Dict[str, Any]] = []

    if intent.get("type") == "INVALID":
        return {
            "aircraft": "",
            "tail_number": tail,
            "image_type": pipeline_image_type,
            "v2_images": [],
            "legacy_images": empty_legacy,
            "insight": _build_professional_insight(
                intent=intent,
                tail=tail,
                aircraft_label="",
                images=[],
                any_tail_match=False,
                used_model_fallback=False,
                pipeline_image_type=pipeline_image_type,
                searchapi_on=searchapi_aircraft_images_enabled(),
                reason_early_exit="invalid",
            ),
        }
    if intent.get("suppress_image_search"):
        return {
            "aircraft": "",
            "tail_number": tail,
            "image_type": pipeline_image_type,
            "v2_images": [],
            "legacy_images": empty_legacy,
            "insight": _build_professional_insight(
                intent=intent,
                tail=tail,
                aircraft_label="",
                images=[],
                any_tail_match=False,
                used_model_fallback=False,
                pipeline_image_type=pipeline_image_type,
                searchapi_on=searchapi_aircraft_images_enabled(),
                reason_early_exit="suppressed",
            ),
        }

    db = db if db is not None else _optional_db()

    canonical = ""
    authoritative = False
    reason = ""
    if tail:
        canonical, authoritative, reason = resolve_aircraft_identity(tail=tail, db=db)
        if reason == "ambiguous_model":
            return {
                "aircraft": "",
                "tail_number": tail,
                "image_type": pipeline_image_type,
                "v2_images": [],
                "legacy_images": empty_legacy,
                "insight": _build_professional_insight(
                    intent=intent,
                    tail=tail,
                    aircraft_label="",
                    images=[],
                    any_tail_match=False,
                    used_model_fallback=False,
                    pipeline_image_type=pipeline_image_type,
                    searchapi_on=searchapi_aircraft_images_enabled(),
                    reason_early_exit="ambiguous_model",
                ),
            }
    else:
        canonical = normalize_aircraft_name(str(intent.get("aircraft") or "").strip())
        authoritative = bool(canonical)

    aircraft_out = canonical if (authoritative or (not tail and bool(canonical))) else ""

    if not searchapi_aircraft_images_enabled():
        return {
            "aircraft": aircraft_out,
            "tail_number": tail,
            "image_type": pipeline_image_type,
            "v2_images": [],
            "legacy_images": empty_legacy,
            "insight": _build_professional_insight(
                intent=intent,
                tail=tail,
                aircraft_label=aircraft_out,
                images=[],
                any_tail_match=False,
                used_model_fallback=False,
                pipeline_image_type=pipeline_image_type,
                searchapi_on=False,
                reason_early_exit="",
            ),
        }

    allowed_cats = allowed_visual_categories_for_intent(intent)

    j = generate_ultra_precise_google_image_queries_json(
        raw,
        required_tail=tail or None,
        required_marketing_type=canonical if canonical else None,
        phly_rows=[],
        strict_tail_mode=bool(tail),
        mm_for_scoring=None,
        intent=intent,
    )
    queries = list(j.get("queries") or [])

    gallery_meta: Dict[str, Any] = {}

    def _run_fetch(qs: List[str], intent_for_premium: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not qs:
            return []
        rows, _meta = fetch_ranked_searchapi_aircraft_images(
            queries=qs,
            canonical_tail=None,
            strict_tail_mode=False,
            marketing_type_for_model_match=(canonical or "").strip() or None,
            max_out=max(12, cap * 3),
            user_query=raw,
            gallery_meta=gallery_meta,
            premium_intent=intent_for_premium,
        )
        return rows

    candidates: List[Dict[str, Any]] = []
    if queries:
        candidates = _rows_to_visual_candidates(
            _run_fetch(queries, intent),
            tail=tail,
            canonical=canonical,
            authoritative=authoritative,
            allowed_cats=allowed_cats,
        )

    if (
        not candidates
        and (tail or "").strip()
        and (canonical or "").strip()
        and authoritative
    ):
        i_fb = _model_only_fallback_intent(intent, canonical_aircraft=canonical)
        fb_q = generate_ultra_precise_google_image_queries_json(
            f"{canonical} {raw}".strip(),
            required_tail=None,
            required_marketing_type=canonical,
            phly_rows=[],
            strict_tail_mode=False,
            mm_for_scoring=None,
            intent=i_fb,
        )
        qs2 = list(fb_q.get("queries") or [])
        if qs2:
            candidates = _rows_to_visual_candidates(
                _run_fetch(qs2, i_fb),
                tail=tail,
                canonical=canonical,
                authoritative=authoritative,
                allowed_cats=allowed_cats,
            )

    candidates = _sort_visual_candidates(candidates, allowed=allowed_cats)
    candidates = [c for c in candidates if not allowed_cats or c["type"] in allowed_cats][:cap]

    any_tail_match = any(c.get("match_type") == "tail_match" for c in candidates)
    used_model_fallback = bool((tail or "").strip()) and bool(candidates) and not any_tail_match

    v2_images: List[Dict[str, Any]] = [
        {
            "url": c["url"],
            "confidence": c["confidence"],
            "source": c["source"],
            "tags": list(c.get("tags") or []),
        }
        for c in candidates
    ]
    legacy_images: List[Dict[str, Any]] = [
        {
            "type": c["type"],
            "url": c["url"],
            "source": c["source"],
            "match_type": c["match_type"],
            "confidence": c["confidence"],
        }
        for c in candidates
    ]

    insight = _build_professional_insight(
        intent=intent,
        tail=tail,
        aircraft_label=aircraft_out,
        images=v2_images,
        any_tail_match=any_tail_match,
        used_model_fallback=used_model_fallback,
        pipeline_image_type=pipeline_image_type,
        searchapi_on=True,
        reason_early_exit="" if queries else "no_queries",
    )

    return {
        "aircraft": aircraft_out,
        "tail_number": tail,
        "image_type": pipeline_image_type,
        "v2_images": v2_images,
        "legacy_images": legacy_images,
        "insight": insight,
    }


def run_aircraft_image_intelligence(
    user_query: str,
    *,
    db: Any = None,
    max_images: int = 8,
) -> Dict[str, Any]:
    """
    HyeAero Aircraft Image Intelligence — strict brokerage JSON.

    Returns ``aircraft``, ``image_type`` (``cabin`` | ``cockpit`` | ``exterior`` | ``comparison``),
    ``images`` with ``url``, ``confidence``, ``source``, ``tags``, and ``insight``.
    """
    b = _visual_intelligence_bundle(user_query, db=db, max_images=max_images)
    return {
        "aircraft": str(b.get("aircraft") or ""),
        "image_type": str(b.get("image_type") or "cabin"),
        "images": list(b.get("v2_images") or []),
        "insight": str(b.get("insight") or ""),
    }


def run_image_intelligence(
    user_query: str,
    *,
    db: Any = None,
    max_images: int = 8,
) -> Dict[str, Any]:
    """
    Legacy shape for tests and callers: ``tail_number`` plus per-image ``type`` / ``match_type``.

    Prefer :func:`run_aircraft_image_intelligence` for the HyeAero brokerage JSON contract.
    """
    b = _visual_intelligence_bundle(user_query, db=db, max_images=max_images)
    return {
        "aircraft": b.get("aircraft") or "",
        "tail_number": b.get("tail_number") or "",
        "images": list(b.get("legacy_images") or []),
    }
