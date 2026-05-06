"""
FINAL CONSULTANT PIPELINE CONTROLLER — intent + recovery → matching → answer → orchestrator.

Strict aircraft lock: candidates list is the only allowed model set for answer copy, pre-rank filtering,
and downstream rank/alignment (no injected generic fallbacks).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_VISUAL_QUERY_RE = re.compile(
    r"(?i)\b(show|see|interior|cabin|cockpit|inside|gallery|photo|photos|picture|pictures|"
    r"view|look\s+at|what\s+.*\s+like)\b",
)

_VISUAL_INTENT_TYPES = frozenset(
    {
        "interior_visual",
        "exterior_visual",
        "cockpit",
        "cabin_search",
        "generic_visual",
    }
)

_FORBIDDEN_CLARIFICATION_RE = re.compile(
    r"(?is)(\b(?:could\s+you|please)\s+(?:tell\s+me|specify)\s+(?:which|what)\s+aircraft\b|"
    r"\bcould\s+you\s+clarify\b.*\baircraft\b|"
    r"\bwhich\s+aircraft\s+(?:are\s+you|did\s+you|do\s+you)\b|"
    r"\bwhich\s+(?:plane|jet|aircraft)\s+(?:are\s+we|were\s+we|did\s+you)\b|"
    r"\bpick\s+(?:which|one)\s+aircraft\b)",
)

def _canonical_ulr_peer_list() -> List[str]:
    from rag.intent.aircraft_matching_engine import _ULR_FLAGSHIP_PEER_MODELS

    return list(_ULR_FLAGSHIP_PEER_MODELS)


def _ulr_anchor_blob(history: List[Dict[str, Any]], user_query: str, normalized_intent: Dict[str, Any]) -> str:
    from rag.intent.aircraft_matching_engine import _ULR_FLAGSHIP_ANCHOR

    blob = (
        _history_blob(history)
        + " "
        + (user_query or "")
        + " "
        + str(normalized_intent.get("aircraft") or "")
    )
    return blob if _ULR_FLAGSHIP_ANCHOR.search(blob) else ""


def filter_candidates_ulr_peer_lock(
    candidates: List[str],
    *,
    ulr_anchor_active: bool,
) -> List[str]:
    """
    When ULR anchor is active, **drop** Gulfstream G500/G600 tokens from the roster;
    other engine-resolved types (e.g. Falcon 900) are preserved.
    """
    base = [str(c).strip() for c in candidates if str(c).strip()]
    if not ulr_anchor_active:
        return base
    _g500 = re.compile(r"(?i)\b(?:gulfstream\s*)?g[-\s]?500\b")
    _g600 = re.compile(r"(?i)\b(?:gulfstream\s*)?g[-\s]?600\b")
    out: List[str] = []
    for s in base:
        low = s.lower()
        if "gulfstream" in low or re.search(r"(?i)\bg[-\s]?5|6\d\d\b", low):
            if _g500.search(s) or _g600.search(s):
                continue
        out.append(s)
    if not out:
        return _canonical_ulr_peer_list()
    return out


def _strip_off_list_aircraft_tokens(text: str, allowed: List[str]) -> str:
    """Remove marketing tokens for aircraft not present in ``allowed`` (answer lock)."""
    if not text or not allowed:
        return (text or "").strip()
    al = [a.strip() for a in allowed if str(a).strip()]
    if not al:
        return (text or "").strip()
    low_allow = " | ".join(a.lower() for a in al)
    t = text
    for pat, fam in (
        (r"(?i)\b(?:gulfstream\s*)?g[-\s]?500\b", "g500"),
        (r"(?i)\b(?:gulfstream\s*)?g[-\s]?600\b", "g600"),
        (r"(?i)\b(?:gulfstream\s*)?g[-\s]?550\b", "g550"),
        (r"(?i)\b(?:gulfstream\s*)?g[-\s]?700\b", "g700"),
    ):
        if fam in low_allow or any(fam in a for a in al):
            continue
        t = re.sub(pat, "", t)
    return " ".join(t.split())


_AIRCRAFT_SCANS_DESC: Sequence[Tuple[re.Pattern[str], Optional[str]]] = (
    (re.compile(r"\b(global\s*(?:express\s*)?(?:[\s*-]*)(?:xb?d?)?\s*7500)", re.I), "Bombardier Global 7500"),
    (re.compile(r"\b(global\s*8000|global\s*8k)", re.I), "Bombardier Global 8000"),
    (re.compile(r"\b(global\s*6000)", re.I), "Bombardier Global 6000"),
    (re.compile(r"\b(global\s*5000)", re.I), "Bombardier Global 5000"),
    (re.compile(r"\b(global\s*6500)", re.I), "Bombardier Global 6500"),
    (re.compile(r"\b(?:gulfstream\s*)?g\s*[-_]?\s*700\b|\bg700\b", re.I), "Gulfstream G700"),
    (re.compile(r"\b(?:gulfstream\s*)?g\s*[-_]?\s*650\b|\bg650\b", re.I), "Gulfstream G650"),
    (re.compile(r"\b(?:gulfstream\s*)?g\s*[-_]?\s*600\b|\bg600\b", re.I), "Gulfstream G600"),
    (re.compile(r"\b(?:gulfstream\s*)?g\s*[-_]?\s*550\b|\bg550\b", re.I), "Gulfstream G550"),
    (re.compile(r"\b(?:gulfstream\s*)?g\s*[-_]?\s*500\b|\bg500\b", re.I), "Gulfstream G500"),
    (re.compile(r"\b(?:gulfstream\s*)?\bgv\b", re.I), "Gulfstream GV family"),
    (re.compile(r"\b(falcon\s*9x\b|falcon\s*8x\b|falcon\s*7x\b)", re.I), None),
    (re.compile(r"\bfalcon\s*(?:[-_\s]*)900(?:lx|ex|b)?", re.I), "Dassault Falcon 900EX"),
    (re.compile(r"\bfalcon\s*(?:[-_\s]*)2000\b", re.I), "Dassault Falcon 2000"),
    (re.compile(r"\bfalcon\s*(?:[-_\s]*)(?:6x|five|five\s*x)\b", re.I), "Dassault Falcon 6X"),
    (re.compile(r"\bchallenger\s*850\b", re.I), "Bombardier Challenger 850"),
    (re.compile(r"\bchallenger\s*(?:605\b|604\b|650\b|350\b|300\b)", re.I), None),
    (
        re.compile(r"\bcitation\s*(?:[-_\s]*)(?:xj|latitude|longitude|hemisphere|citation\s*sj)", re.I),
        "Citation family",
    ),
    (re.compile(r"\bembraer\s*(?:praetor|legacy|lineage|phenom)", re.I), "Embraer business jet"),
    (re.compile(r"\bblearn?jet\b", re.I), "Learjet family"),
)


def _challenger_model_from_regex(m: Optional[re.Match[str]]) -> Optional[str]:
    if not m:
        return None
    blob = (m.group(0) or "").lower()
    if "650" in blob:
        return "Bombardier Challenger 650"
    if "605" in blob or "604" in blob:
        return "Bombardier Challenger 605"
    if "350" in blob:
        return "Bombardier Challenger 350"
    if "300" in blob:
        return "Bombardier Challenger 300"
    return None


def _falcon_x_from_regex(m: Optional[re.Match[str]]) -> Optional[str]:
    if not m:
        return None
    blob = (m.group(0) or "").lower()
    if "9x" in blob:
        return "Dassault Falcon 9X"
    if "8x" in blob:
        return "Dassault Falcon 8X"
    if "7x" in blob:
        return "Dassault Falcon 7X"
    return None


def _first_aircraft_in_text_chunk(tl: str) -> Optional[str]:
    if re.search(r"(?is)\bfalcon[^\n]{0,16}9000\b", tl) or re.search(
        r"(?is)\b9000\b[^\n]{0,8}\bfalcon\b", tl
    ):
        return "Dassault Falcon 900EX"
    for rx, canned in _AIRCRAFT_SCANS_DESC:
        m = rx.search(tl)
        if not m:
            continue
        if canned is None:
            if "challenger" in tl.lower():
                cm = _challenger_model_from_regex(m)
                if cm:
                    return cm
            if "falcon" in tl.lower():
                fm = _falcon_x_from_regex(m)
                if fm:
                    return fm
            continue
        return canned
    return None


def scan_thread_aircraft_mention(
    history: Optional[List[Dict[str, Any]]],
    *,
    max_turns: int = 36,
) -> Optional[str]:
    """History-only newest-first aircraft resolution — overrides query-level parsing."""
    if not history:
        return None
    for h in reversed(history[-max_turns:]):
        if not isinstance(h, dict):
            continue
        tl = str(h.get("content") or "")
        hit = _first_aircraft_in_text_chunk(tl)
        if hit:
            return hit
    return None


def scan_last_aircraft_mention(
    history: Optional[List[Dict[str, Any]]],
    user_query: str,
    *,
    max_turns: int = 36,
) -> Optional[str]:
    """Query first (current turn), then thread messages."""
    chunks: List[str] = []
    if user_query and str(user_query).strip():
        chunks.append(str(user_query).strip())
    if history:
        for h in reversed(history[-max_turns:]):
            if isinstance(h, dict):
                chunks.append(str(h.get("content") or ""))
    for tl in chunks:
        if not (tl or "").strip():
            continue
        hit = _first_aircraft_in_text_chunk(tl)
        if hit:
            return hit
    return None


def apply_aircraft_context_recovery(
    normalized_intent: Dict[str, Any],
    history: List[Dict[str, Any]],
    user_query: str,
) -> Dict[str, Any]:
    """
    Prefer **history-resolved aircraft** over the normalized query parse when history contains a firm mention.
    """
    out = dict(normalized_intent)
    thread_ac = scan_thread_aircraft_mention(history)
    if thread_ac:
        out["aircraft"] = thread_ac
        return out
    if str(out.get("aircraft") or "").strip():
        return out
    merged = scan_last_aircraft_mention(history, user_query)
    if merged:
        out["aircraft"] = merged
    return out


def enrich_matching_for_visual_lock(
    matching: Dict[str, Any],
    *,
    visual_trigger: bool,
    normalized_intent: Dict[str, Any],
    user_query: str,
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Visual flow: tolerate classifier ``hard_fail`` without inventing fallback aircraft —
    reuse engine-only lists and ULR peer roster when mandated.
    """
    from rag.intent.aircraft_matching_engine import (
        _ULR_FLAGSHIP_PEER_MODELS,
        run_aircraft_matching_engine,
    )

    m = dict(matching)
    ulr_blob_hit = bool(_ulr_anchor_blob(history, user_query, normalized_intent))

    if visual_trigger and m.get("hard_fail"):
        if ulr_blob_hit:
            m["aircraft_candidates"] = list(_ULR_FLAGSHIP_PEER_MODELS)
        else:
            m["aircraft_candidates"] = []
        m["hard_fail"] = False
        m["hard_fail_reason"] = None
        m["reasoning"] = "visual_intent_hard_fail_relaxed"

    picked = [str(x).strip() for x in (m.get("aircraft_candidates") or []) if str(x).strip()]

    if visual_trigger and not picked:
        m2 = run_aircraft_matching_engine(
            user_query or "",
            history=history,
            normalized_intent=normalized_intent,
        )
        if not m2.get("hard_fail"):
            picked = [str(x).strip() for x in (m2.get("aircraft_candidates") or []) if str(x).strip()]
            if picked:
                m.update({k: m2[k] for k in ("reasoning",) if k in m2})

    if visual_trigger and not picked:
        anchor = str(normalized_intent.get("aircraft") or "").strip()
        if anchor:
            m3 = run_aircraft_matching_engine(
                user_query or "",
                history=history,
                normalized_intent=normalized_intent,
                proposed_candidates=[anchor],
            )
            if not m3.get("hard_fail"):
                picked = [str(x).strip() for x in (m3.get("aircraft_candidates") or []) if str(x).strip()]

    ql = (user_query or "").lower()
    if visual_trigger and not picked and re.search(r"\bfalcon\b", ql) and re.search(r"\b9000\b", ql):
        mf = run_aircraft_matching_engine(
            user_query or "",
            history=history,
            normalized_intent=normalized_intent,
            proposed_candidates=["Dassault Falcon 900EX"],
        )
        if not mf.get("hard_fail"):
            picked = [str(x).strip() for x in (mf.get("aircraft_candidates") or []) if str(x).strip()]

    if visual_trigger and not picked and ulr_blob_hit:
        picked = list(_ULR_FLAGSHIP_PEER_MODELS)

    picked = filter_candidates_ulr_peer_lock(picked, ulr_anchor_active=ulr_blob_hit)
    m["aircraft_candidates"] = list(picked)
    return m


def _history_blob(history: Optional[List[Dict[str, Any]]], max_msgs: int = 12) -> str:
    if not history:
        return ""
    parts: List[str] = []
    for h in history[-max_msgs:]:
        if isinstance(h, dict):
            parts.append(str(h.get("content") or ""))
    return " ".join(parts).strip()


def _visual_trigger(user_query: str, history: Optional[List[Dict[str, Any]]], normalized_intent: Dict[str, Any]) -> bool:
    it = str(normalized_intent.get("intent_type") or "").strip()
    if it in _VISUAL_INTENT_TYPES:
        return True
    q = (user_query or "").strip()
    if _VISUAL_QUERY_RE.search(q):
        return True
    if _VISUAL_QUERY_RE.search(_history_blob(history)):
        return True
    return False


def _sanitize_answer_strict_aircraft(answer: str, candidates: List[str]) -> str:
    t = _FORBIDDEN_CLARIFICATION_RE.sub("", answer or "").strip()
    t = _strip_off_list_aircraft_tokens(t, candidates)
    cl = [c.lower() for c in candidates if str(c).strip()]
    if not cl:
        return " ".join(t.split())
    if "g500" in t.lower() and not any("g500" in c for c in cl):
        t = re.sub(r"(?i)\bg[-\s]?500\b(?:\s*[/]\s*g[-\s]?600)?", "", t)
    if "g600" in t.lower() and not any("g600" in c or " g600" in c for c in cl):
        t = re.sub(r"(?i)\bg[-\s]?600\b", "", t)
    return " ".join(t.split())


def _controlled_answer_text(
    user_query: str,
    *,
    normalized_intent: Dict[str, Any],
    matching: Dict[str, Any],
    visual_trigger: bool,
    strict_aircraft_only: bool = False,
) -> str:
    raw_cands = [str(x).strip() for x in (matching.get("aircraft_candidates") or []) if str(x).strip()]
    vf = str(normalized_intent.get("visual_focus") or "").strip() or "interior"
    qlow = (user_query or "").lower()

    if not raw_cands:
        body = "**Thread aircraft lock** is pending resolver output — gallery runs only once candidates are populated."
        if visual_trigger:
            body += f" **{vf.capitalize()}** visuals require a non-empty matched candidate list from the engine."
        return _sanitize_answer_strict_aircraft(body, raw_cands)

    lead = raw_cands[0]

    if strict_aircraft_only or visual_trigger:
        if (
            ("falcon" in qlow and "9000" in qlow)
            and raw_cands
            and any("falcon" in c.lower() and "900" in c.lower() for c in raw_cands)
        ):
            body = (
                f"There is no production **Falcon 9000** — the regulated family names are Falcon 900 / 900EX / 900LX. "
                f"**{lead}** is on-candidate wording for visuals now."
            )
        elif lead:
            body = (
                f"**{lead}** is the locked on-list reference — wording and imagery omit any aircraft outside "
                "`aircraft_candidates`."
            )
        else:
            body = f"Cabin visuals lock to **{lead}**."

        body += f" **{vf.capitalize()}** gallery rows are filtered to that roster only."
        body = _sanitize_answer_strict_aircraft(body, raw_cands)
        return " ".join(body.split())

    if "falcon" in qlow and "9000" in qlow and lead:
        body = (
            f"There is no production **Falcon 9000** — likely **Falcon 900 / 900EX / 900LX**. **{lead}** anchors wording."
        )
    elif lead:
        body = f"**{lead}** anchors this turn."
    else:
        body = "Focused guidance on your constraints."

    if visual_trigger:
        body += f" Gallery: **{vf}**."

    body = _sanitize_answer_strict_aircraft(body, raw_cands)
    if "cannot" in body.lower() and "image" in body.lower():
        body = re.sub(r"(?is)i\s+cannot\s+find\s+images[^.]*\.?", "", body).strip()
    return " ".join(body.split())


def _image_blob_ranking(im: Dict[str, Any]) -> str:
    parts = [
        str(im.get("title") or ""),
        str(im.get("description") or ""),
        str(im.get("url") or ""),
        str(im.get("source_domain") or ""),
    ]
    return " ".join(parts).strip().lower()


def filter_images_aircraft_precheck(
    images: List[Dict[str, Any]],
    aircraft_candidates: List[str],
    *,
    min_aircraft_signal: float = 0.65,
) -> List[Dict[str, Any]]:
    """Drop ranking inputs that cannot satisfy aircraft lock before scoring."""
    cands = [str(x).strip() for x in aircraft_candidates if str(x).strip()]
    if not cands:
        return [dict(x) for x in images if isinstance(x, dict)]
    try:
        from services.aviation_image_rank_filter_engine import _aircraft_match_score
    except Exception:
        return [dict(x) for x in images if isinstance(x, dict)]

    out: List[Dict[str, Any]] = []
    for im in images or []:
        if not isinstance(im, dict):
            continue
        b = _image_blob_ranking(im)
        best = max(float(_aircraft_match_score(c, b)) for c in cands)
        if best >= float(min_aircraft_signal):
            out.append(dict(im))
    return out


def _rows_for_ranking(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in raw or []:
        if not isinstance(r, dict):
            continue
        u = str(r.get("url") or r.get("image") or "").strip()
        if not u:
            continue
        title = str(r.get("title") or r.get("description") or "")
        desc = str(r.get("snippet") or r.get("description") or title)
        dom = str(r.get("source_domain") or "")
        if not dom:
            from urllib.parse import urlparse

            try:
                dom = (urlparse(u).netloc or "").lower()
            except Exception:
                dom = ""
        out.append(
            {
                "url": u,
                "title": title,
                "description": desc,
                "source_domain": dom,
                "source": str(r.get("source") or ""),
                "page_url": str(r.get("_source_page") or r.get("link") or ""),
            }
        )
    return out


def default_searchapi_fetch(
    user_query: str,
    normalized_intent: Dict[str, Any],
    matching: Dict[str, Any],
    queries: List[str],
) -> List[Dict[str, Any]]:
    try:
        from services.searchapi_aircraft_images import (
            fetch_ranked_searchapi_aircraft_images,
            searchapi_aircraft_images_enabled,
        )
    except Exception as e:
        logger.debug("searchapi import failed: %s", e)
        return []

    if not searchapi_aircraft_images_enabled():
        return []

    cands = list(matching.get("aircraft_candidates") or [])
    mm = (cands[0] if cands else None) or str(normalized_intent.get("aircraft") or "").strip() or None
    try:
        rows, _meta = fetch_ranked_searchapi_aircraft_images(
            queries=queries or [user_query],
            canonical_tail=None,
            strict_tail_mode=False,
            marketing_type_for_model_match=mm,
            per_query_results=12,
            max_out=8,
            user_query=user_query or "",
            gallery_meta={},
            premium_intent={},
        )
    except Exception as e:
        logger.debug("default_searchapi_fetch failed: %s", e)
        return []
    ranked = _rows_for_ranking(list(rows or []))
    return filter_images_aircraft_precheck(ranked, cands)


def generate_consultant_response(
    user_query: str,
    history: list,
) -> Dict[str, Any]:
    from rag.intent import (
        generate_aviation_image_queries,
        normalize_aviation_intent,
        run_aircraft_matching_engine,
    )
    from services.aviation_image_orchestrator import orchestrate_aviation_image_pipeline
    from services.aviation_image_relevance_filter import filter_aviation_images_by_relevance

    uq = (user_query or "").strip()
    hist = history if isinstance(history, list) else []

    normalized_intent: Dict[str, Any] = dict(normalize_aviation_intent(uq, hist))
    normalized_intent = apply_aircraft_context_recovery(normalized_intent, hist, uq)
    visual_pre = bool(_visual_trigger(uq, hist, normalized_intent))

    matching = run_aircraft_matching_engine(
        uq,
        history=hist,
        normalized_intent=normalized_intent,
    )
    matching = enrich_matching_for_visual_lock(
        matching,
        visual_trigger=visual_pre,
        normalized_intent=normalized_intent,
        user_query=uq,
        history=hist,
    )

    ulr_hit = bool(_ulr_anchor_blob(hist, uq, normalized_intent))
    c_locked = filter_candidates_ulr_peer_lock(
        list(matching.get("aircraft_candidates") or []),
        ulr_anchor_active=ulr_hit,
    )
    matching["aircraft_candidates"] = list(c_locked)
    candidates_meta = list(matching["aircraft_candidates"])

    if matching.get("hard_fail") and not visual_pre:
        answer = _controlled_answer_text(
            uq,
            normalized_intent=normalized_intent,
            matching=matching,
            visual_trigger=False,
            strict_aircraft_only=False,
        )
        return {
            "answer": answer,
            "images": [],
            "meta": {
                "visual_trigger": visual_pre,
                "pipeline_used": True,
                "aircraft_candidates": candidates_meta,
            },
        }

    answer_text = _controlled_answer_text(
        uq,
        normalized_intent=normalized_intent,
        matching=matching,
        visual_trigger=visual_pre,
        strict_aircraft_only=bool(visual_pre),
    )

    images_out: List[Dict[str, Any]] = []
    if visual_pre:
        qpayload = generate_aviation_image_queries(normalized_intent)
        queries = list(qpayload.get("queries") or [])
        raw_images = default_searchapi_fetch(uq, normalized_intent, matching, queries)
        raw_images = filter_images_aircraft_precheck(raw_images, candidates_meta)
        filtered_images = filter_aviation_images_by_relevance(raw_images)
        filtered_images = filter_images_aircraft_precheck(filtered_images, candidates_meta)

        def _search_fn(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
            qs = list(payload.get("queries") or queries)
            got = default_searchapi_fetch(uq, normalized_intent, matching, qs)
            return filter_images_aircraft_precheck(got, candidates_meta)

        orchestrated = orchestrate_aviation_image_pipeline(
            normalized_intent=normalized_intent,
            aircraft_candidates=list(matching.get("aircraft_candidates") or []),
            answer_text=answer_text,
            raw_images=raw_images,
            filtered_images=filtered_images,
            search_fn=_search_fn,
        )
        imgs = list(orchestrated.get("final_images") or [])
        if len(imgs) >= 3:
            images_out = imgs[:5]

    return {
        "answer": answer_text,
        "images": images_out,
        "meta": {
            "visual_trigger": visual_pre,
            "pipeline_used": True,
            "aircraft_candidates": candidates_meta,
        },
    }


__all__ = [
    "apply_aircraft_context_recovery",
    "default_searchapi_fetch",
    "enrich_matching_for_visual_lock",
    "filter_candidates_ulr_peer_lock",
    "filter_images_aircraft_precheck",
    "generate_consultant_response",
    "scan_last_aircraft_mention",
    "scan_thread_aircraft_mention",
]
