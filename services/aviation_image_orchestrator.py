"""
Aviation Image Orchestrator — deterministic **rank → align → validate → fallback** pipeline.

Does not trust upstream ordering: always re-ranks ``filtered_images`` (or relevance-filtered
``raw_images``), always runs alignment, validates gates, optional bounded retries with stricter
aircraft anchor and optional ``search_fn`` refresh.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MIN_IMAGES = 3
_MIN_TOP_FOR_ALIGN = 3
_MAX_TOP = 6
_FINAL_MIN = 3
_FINAL_MAX = 5
_DEFAULT_RANK_MIN = 0.65
_RELAXED_RANK_MIN = 0.65
_MAX_RETRIES = 2


def _url_image_id(url: str) -> str:
    u = str(url or "").encode("utf-8", errors="ignore")
    return hashlib.sha256(u).hexdigest()[:16] if u else ""


def _dedupe_by_url(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for im in images or []:
        if not isinstance(im, dict):
            continue
        u = str(im.get("url") or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(im)
    return out


def _merge_rank_with_pool(
    ranked_meta: List[Dict[str, Any]],
    pool: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach full image rows to ranking metadata (preserves ``score``, etc.)."""
    by_id: Dict[str, Dict[str, Any]] = {}
    for im in pool or []:
        if not isinstance(im, dict):
            continue
        u = str(im.get("url") or "").strip()
        if not u:
            continue
        iid = str(im.get("image_id") or "").strip() or _url_image_id(u)
        by_id[iid] = im
    merged: List[Dict[str, Any]] = []
    for meta in ranked_meta or []:
        if not isinstance(meta, dict):
            continue
        iid = str(meta.get("image_id") or "").strip()
        base = by_id.get(iid)
        if not isinstance(base, dict):
            continue
        row = dict(base)
        for k in ("score", "aircraft_match", "visual_match", "source_quality", "reason"):
            if k in meta:
                row[k] = meta[k]
        merged.append(row)
    return merged


def _claim_terms(answer: str) -> List[str]:
    from services.image_answer_alignment_engine import _claim_style_terms

    return _claim_style_terms(answer or "")


def _exterior_only(blob: str) -> bool:
    from services.image_answer_alignment_engine import _exterior_only_blob

    return bool(_exterior_only_blob(blob))


def _blob_row(im: Dict[str, Any]) -> str:
    from services.image_answer_alignment_engine import _alignment_image_blob

    return _alignment_image_blob(im)


def _validate_stage(
    images: List[Dict[str, Any]],
    *,
    normalized_intent: Optional[Dict[str, Any]],
    aircraft_candidates: Optional[List[str]],
    answer_text: str,
    min_per_image_score: float,
) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    if len(images) < _MIN_IMAGES:
        issues.append("count_below_minimum")
        return False, issues

    norm = normalized_intent if isinstance(normalized_intent, dict) else {}
    vf = str(norm.get("visual_focus") or "").strip().lower()
    it = str(norm.get("intent_type") or "").strip().lower()
    cabin_intent = it in ("interior_visual", "cabin_search") or vf in (
        "interior",
        "cabin",
        "bedroom",
        "galley",
    )

    cands = [str(x).strip() for x in (aircraft_candidates or []) if str(x).strip()]
    try:
        from services.aviation_image_rank_filter_engine import _aircraft_match_score
    except Exception:
        _aircraft_match_score = None  # type: ignore

    dom_keys: List[str] = []
    for im in images:
        b = _blob_row(im)
        sc = float(im.get("score") or 0.0)
        if sc < float(min_per_image_score):
            issues.append("score_below_threshold")
            return False, issues

        if cands and _aircraft_match_score is not None:
            best = max(float(_aircraft_match_score(c, b)) for c in cands)
            if best < 0.65:
                issues.append("aircraft_match_fail")
                return False, issues

        if cabin_intent and vf != "exterior" and _exterior_only(b):
            issues.append("visual_focus_exterior_mismatch")
            return False, issues

        if cands and _aircraft_match_score is not None:
            best_c = cands[0]
            best_s = float(_aircraft_match_score(cands[0], b))
            for c in cands[1:]:
                s = float(_aircraft_match_score(c, b))
                if s > best_s:
                    best_s = s
                    best_c = c
            dom_keys.append(best_c.lower()[:48])

    if len(cands) >= 2 and len(set(dom_keys)) > 1 and it not in ("comparison",):
        issues.append("mixed_aircraft_cluster")
        return False, issues

    claims = _claim_terms(answer_text or "")
    if claims and not any(
        any(t in _blob_row(im) for t in claims) for im in images
    ):
        issues.append("claim_support_missing")
        return False, issues

    return True, issues


def _final_cleanup(images: List[Dict[str, Any]], *, min_score: float) -> List[Dict[str, Any]]:
    rows = _dedupe_by_url(images)
    rows = [r for r in rows if float(r.get("score") or 0.0) >= float(min_score)]
    rows.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
    out = rows[:_FINAL_MAX]
    return out


def orchestrate_aviation_image_pipeline(
    *,
    normalized_intent: Optional[Dict[str, Any]],
    aircraft_candidates: Optional[List[str]],
    answer_text: str,
    raw_images: List[Dict[str, Any]],
    filtered_images: List[Dict[str, Any]],
    search_fn: Optional[Callable[[Dict[str, Any]], List[Dict[str, Any]]]] = None,
    max_retries: int = _MAX_RETRIES,
) -> Dict[str, Any]:
    """
    Mandatory pipeline: rank → top pick → align → validate → bounded fallback.

    ``search_fn`` (optional): ``callable(query_payload) -> raw_images`` where ``query_payload``
    is the dict from :func:`rag.intent.generate_aviation_image_queries` (``{"queries":[...]}``),
    used on fallback to refresh results when upstream search is available.
    """
    decisions: Dict[str, Any] = {
        "ranking_applied": False,
        "alignment_applied": False,
        "fallback_used": False,
        "retries": 0,
    }

    from services.aviation_image_ranking_engine import rank_aviation_images_for_intent
    from services.image_answer_alignment_engine import align_images_with_consultant_answer
    from services.aviation_image_relevance_filter import evaluate_aviation_image_relevance

    def _build_pool(filt: List[Dict[str, Any]], raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pool = [dict(x) for x in (filt or []) if str(x.get("url") or "").strip()]
        if pool:
            return _dedupe_by_url(pool)
        out_raw: List[Dict[str, Any]] = []
        for im in raw or []:
            if not isinstance(im, dict) or not str(im.get("url") or "").strip():
                continue
            ev = evaluate_aviation_image_relevance(im)
            if ev.get("accepted"):
                out_raw.append(dict(im))
        return _dedupe_by_url(out_raw)

    def _refresh_pool_after_fallback() -> None:
        nonlocal pool, norm, strict_cands
        if cands:
            strict_cands = [cands[0]]
            na = dict(norm)
            na["aircraft"] = cands[0]
            norm = na
        if search_fn is not None:
            try:
                from rag.intent import generate_aviation_image_queries

                qpayload = generate_aviation_image_queries(norm)
                fresh = search_fn(qpayload)
                pool = _build_pool(fresh, fresh)
            except Exception as e:
                logger.debug("orchestrator search_fn fallback failed: %s", e)
                pool = _dedupe_by_url(list(raw_images or []) + list(filtered_images or []))
        else:
            pool = _dedupe_by_url(list(raw_images or []) + list(filtered_images or []))

    pool = _build_pool(filtered_images, raw_images)
    norm = dict(normalized_intent) if isinstance(normalized_intent, dict) else {}
    cands = [str(x).strip() for x in (aircraft_candidates or []) if str(x).strip()]
    strict_cands = list(cands)
    last_rank_floor = float(_DEFAULT_RANK_MIN)

    for attempt in range(max_retries + 1):
        # After any prior failure, relax ranking floor so duplicate down-ranks can still yield ≥3 rows.
        rank_floor = _RELAXED_RANK_MIN if (attempt > 0 or decisions.get("fallback_used")) else _DEFAULT_RANK_MIN
        last_rank_floor = float(rank_floor)
        ranked_meta = rank_aviation_images_for_intent(
            normalized_intent=norm,
            aircraft_candidates=list(strict_cands),
            images=pool,
            min_score=rank_floor,
            max_keep=_MAX_TOP,
        )
        decisions["ranking_applied"] = True

        if not ranked_meta:
            decisions["fallback_used"] = True
            _refresh_pool_after_fallback()
            decisions["retries"] = attempt
            continue

        merged = _merge_rank_with_pool(ranked_meta, pool)
        top = merged[:_MAX_TOP]
        if len(top) < _MIN_TOP_FOR_ALIGN:
            decisions["fallback_used"] = True
            _refresh_pool_after_fallback()
            decisions["retries"] = attempt
            continue

        align_out = align_images_with_consultant_answer(
            answer_text=answer_text or "",
            normalized_intent=norm,
            selected_images=top,
            aircraft_candidates=list(strict_cands or cands),
            image_pool=_dedupe_by_url(list(pool) + list(raw_images or [])),
        )
        decisions["alignment_applied"] = True
        aligned = list(align_out.get("final_images") or [])

        ok, _iss = _validate_stage(
            aligned,
            normalized_intent=norm,
            aircraft_candidates=list(strict_cands or cands),
            answer_text=answer_text or "",
            min_per_image_score=last_rank_floor,
        )
        if ok:
            final_images = _final_cleanup(aligned, min_score=last_rank_floor)
            if len(final_images) >= _MIN_IMAGES:
                decisions["retries"] = attempt
                return {"final_images": final_images, "pipeline_decisions": decisions}

        decisions["fallback_used"] = True
        _refresh_pool_after_fallback()
        decisions["retries"] = attempt

    if not decisions.get("alignment_applied"):
        align_images_with_consultant_answer(
            answer_text=answer_text or "",
            normalized_intent=norm,
            selected_images=[],
            aircraft_candidates=list(strict_cands or cands),
            image_pool=_dedupe_by_url(list(pool) + list(raw_images or [])),
        )
        decisions["alignment_applied"] = True

    decisions["retries"] = max_retries
    return {"final_images": [], "pipeline_decisions": decisions}
