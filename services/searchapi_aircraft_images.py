"""
Bing Images aircraft gallery via SearchAPI.io (replaces Tavily *image* retrieval only).

Web snippet search remains on Tavily elsewhere; this module is scoped to image URLs only.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

from rag.aviation_tail import (
    find_loose_us_n_tail_tokens_in_text,
    find_strict_tail_candidates_in_text,
    normalize_tail_token,
)

logger = logging.getLogger(__name__)

# Host / TLD tokens that must never count as a second “registration” vs the canonical tail.
_IGNORED_TAIL_LIKE_TOKENS = frozenset(
    {
        "NET",
        "COM",
        "ORG",
        "EDU",
        "GOV",
        "IO",
        "CO",
        "TV",
        "INFO",
        "BIZ",
        "UK",
        "US",
        "EU",
        "ME",
        "CC",
        "AI",
        "INT",
    }
)

SEARCHAPI_SEARCH_URL = "https://www.searchapi.io/api/v1/search"

# Buyer-oriented tail gallery: listings first, then spotter sites.
_DOMAIN_SCORES_TAIL: Tuple[Tuple[str, int], ...] = (
    ("controller.com", 1000),
    ("aircraftexchange", 950),
    ("jetphotos.", 900),
    ("planespotters.", 850),
    ("globalair.", 700),
    ("avbuyer.", 200),
)
# Visual / type browsing: spotters first.
_DOMAIN_SCORES_MODEL: Tuple[Tuple[str, int], ...] = (
    ("jetphotos.", 1000),
    ("planespotters.", 950),
    ("controller.com", 800),
    ("aircraftexchange", 750),
    ("globalair.", 700),
    ("avbuyer.", 300),
)

# Longer keys first so ``G650ER`` wins over ``G650``.
NORMALIZATION_MAP: Tuple[Tuple[str, str], ...] = (
    ("G650ER", "Gulfstream G650ER"),
    ("G650", "Gulfstream G650"),
    ("G550", "Gulfstream G550"),
    ("G500", "Gulfstream G500"),
    ("G600", "Gulfstream G600"),
    ("G280", "Gulfstream G280"),
    ("CL350", "Challenger 350"),
    ("CL300", "Challenger 300"),
    ("CL650", "Challenger 650"),
    ("GLOBAL XRS", "Global Express XRS"),
    ("GLOBAL 6000", "Global 6000"),
    ("GLOBAL 7500", "Global 7500"),
    ("GLOBAL 8000", "Global 8000"),
    ("PHENOM 300", "Embraer Phenom 300"),
    ("PHENOM 100", "Embraer Phenom 100"),
)

INTENT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "cabin": ("cabin", "interior"),
    "cockpit": ("cockpit", "flight deck"),
    "exterior": ("exterior",),
}

MAX_PER_DOMAIN = 2
# Minimum tail match score (V3) before a row enters ranking / gallery output.
MIN_TAIL_MATCH_SCORE = 80
# User V3 conflict regex on domain-stripped text (US-style marks).
_N_TAIL_CONFLICT_V3 = re.compile(r"\bN\d{1,5}[A-Z]{0,2}\b")


def normalize_aircraft_name(name: str) -> str:
    """Map shorthand tokens (G650, CL350, …) to a canonical marketing string for image queries."""
    raw = (name or "").strip()
    if not raw:
        return raw
    name_upper = raw.upper()
    for key, val in NORMALIZATION_MAP:
        if key in name_upper:
            return val
    return raw


def extract_domain(url: str) -> str:
    try:
        return (urlparse(url or "").netloc or "").lower()
    except Exception:
        return ""


def classify_tail_match_confidence(score: int) -> Optional[str]:
    if score >= 120:
        return "confirmed"
    if score >= 80:
        return "probable"
    return None


def detect_query_image_intent(user_query: str) -> Optional[str]:
    """Single primary facet from the user message (deterministic)."""
    low = (user_query or "").lower()
    if not low.strip():
        return None
    # Prefer the most specific visual facet (cockpit / cabin before generic exterior).
    if any(w in low for w in INTENT_KEYWORDS["cockpit"]):
        return "cockpit"
    if any(w in low for w in INTENT_KEYWORDS["cabin"]):
        return "cabin"
    if any(w in low for w in INTENT_KEYWORDS["exterior"]):
        return "exterior"
    return None


def strip_domains(text: str) -> str:
    """
    Remove host-shaped ``TOKEN.COM`` / ``.NET`` / ``.ORG`` spans before tail substring / regex work
    so ``PLANESPOTTERS.NET`` never yields a fake ``NET`` tail.
    """
    s = re.sub(
        r"\b[A-Z0-9-]+\.(?:COM|NET|ORG)\b",
        " ",
        (text or "").upper(),
    )
    return re.sub(r"\s+", " ", s).strip()


def apply_intent_boost(score: float, result: Dict[str, Any], intent: Optional[str]) -> float:
    if not intent:
        return score
    link = str(result.get("link") or result.get("url") or result.get("_source_page") or "")
    text = f"{result.get('title') or ''} {link}".lower()
    for word in INTENT_KEYWORDS.get(intent, ()):
        if word in text:
            score += 40.0
            break
    if intent == "cabin" and "exterior" in text:
        score -= 40.0
    if intent == "cockpit" and "cabin" in text:
        score -= 20.0
    if intent == "exterior" and "cabin" in text:
        score -= 20.0
    return score


def _domain_score_for_query_mode(url: str, mode: str) -> int:
    low = (url or "").lower()
    table = _DOMAIN_SCORES_TAIL if mode == "tail" else _DOMAIN_SCORES_MODEL
    best = 0
    for frag, sc in table:
        if frag in low:
            best = max(best, sc)
    return best


def searchapi_aircraft_images_enabled() -> bool:
    return bool((os.getenv("SEARCHAPI_API_KEY") or "").strip())


def search_aircraft_images(query: str, *, num_results: int = 15, timeout: float = 28.0) -> List[Dict[str, str]]:
    """
    Reusable Bing Images search via SearchAPI.io.

    Returns normalized rows: ``{"url": str, "title": str, "source": str}``.
    ``source`` is a human-readable site label (SearchAPI ``source.name`` or host).
    """
    raw_q = (query or "").strip()
    if not raw_q:
        return []
    api_key = (os.getenv("SEARCHAPI_API_KEY") or "").strip()
    if not api_key:
        return []
    n = max(10, min(20, int(num_results)))
    params = {
        "engine": "bing_images",
        "q": raw_q,
        "api_key": api_key,
        "safe_search": "moderate",
    }
    # SearchAPI supports pagination; request enough images in one page when available.
    if n != 15:
        params["num"] = n
    try:
        r = requests.get(SEARCHAPI_SEARCH_URL, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning("SearchAPI Bing Images request failed: %s", e)
        return []

    images = data.get("images") if isinstance(data, dict) else None
    if not isinstance(images, list):
        return []

    out: List[Dict[str, str]] = []
    for item in images[:n]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        src = item.get("source") if isinstance(item.get("source"), dict) else {}
        src_name = str((src or {}).get("name") or "").strip()
        src_link = str((src or {}).get("link") or "").strip()
        orig = item.get("original") if isinstance(item.get("original"), dict) else {}
        url = str((orig or {}).get("link") or "").strip()
        if not url.startswith("https://"):
            continue
        host = ""
        try:
            host = (urlparse(url).netloc or "").lower()
        except Exception:
            host = ""
        source_label = src_name or host or "web"
        # ``_source_page`` is used only for strict tail verification (title/url/source page text).
        out.append(
            {
                "url": url,
                "title": title,
                "source": source_label,
                "_source_page": src_link,
            }
        )
    return out


def compose_manufacturer_model_phrase(manufacturer: str, model: str) -> str:
    """
    Join manufacturer + model without duplicated OEM/family tokens, then normalize shorthands.

    If the full manufacturer string is already contained in the model string, return the model only
    (e.g. ``Falcon`` + ``Falcon 2000`` → ``Falcon 2000`` → normalized ``Dassault Falcon 2000`` only when
    normalization applies).
    """
    man = (manufacturer or "").strip()
    mdl = (model or "").strip()
    if not man:
        return normalize_aircraft_name(mdl) if mdl else ""
    if not mdl:
        return normalize_aircraft_name(man) if man else ""
    if man.lower() in mdl.lower():
        return normalize_aircraft_name(mdl)
    m_parts = man.split()
    d_parts = mdl.split()
    while d_parts and m_parts and m_parts[-1].lower() == d_parts[0].lower():
        d_parts = d_parts[1:]
    out = " ".join(m_parts + d_parts).strip()
    return normalize_aircraft_name(out)


def _blob_stripped_for_tail_conflict_scan(blob: str) -> str:
    """Remove URL / hostname spans before registration extraction (avoids ``.net`` false tails)."""
    s = re.sub(r"https?://[^\s]+", " ", blob or "", flags=re.I)
    s = re.sub(
        r"\b(?:[a-z0-9-]+\.)+(?:com|net|org|io|co|uk|eu|gov|edu|info|biz|tv)\b",
        " ",
        s,
        flags=re.I,
    )
    s = strip_domains(s)
    return re.sub(r"\s+", " ", s).strip()


def _all_registration_like_tokens(blob: str) -> List[str]:
    clean = _blob_stripped_for_tail_conflict_scan(blob)
    seen: set[str] = set()
    out: List[str] = []
    for t in find_strict_tail_candidates_in_text(clean) + find_loose_us_n_tail_tokens_in_text(clean):
        u = normalize_tail_token(t)
        if u in seen:
            continue
        seen.add(u)
        out.append(t)
    return out


def compute_tail_match_score(result: Dict[str, Any], tail: str) -> int:
    """
    Deterministic tail relevance (V3): strong URL/title signals, consistency bonus, conflict penalties.

    Maps SearchAPI rows: ``image`` / ``url`` → image URL; ``link`` / ``_source_page`` → page link.
    """
    tail_u = normalize_tail_token(tail)
    if not tail_u:
        return -10_000

    title_raw = str(result.get("title") or "")
    snippet_raw = str(result.get("snippet") or "")
    link_raw = str(result.get("link") or result.get("_source_page") or result.get("source_page") or "")
    image_raw = str(result.get("image") or result.get("url") or "")

    tail_l = tail_u.lower()
    title_s = strip_domains(title_raw).lower()
    snippet_s = strip_domains(snippet_raw).lower()
    link_s = strip_domains(link_raw).lower()
    image_s = strip_domains(image_raw).lower()
    text_blob = f"{title_s} {snippet_s} {link_s}"

    score = 0

    def _mark_in_cdn_path(blob: str, mark: str) -> bool:
        """CDN image paths often omit the leading ``N`` (``.../807js_...`` vs ``N807JS``)."""
        if not mark or not blob:
            return False
        m = normalize_tail_token(mark)
        low = blob.lower()
        if m.lower() in low:
            return True
        if m.startswith("N") and len(m) >= 4:
            suf = m[1:].lower()
            if re.search(rf"(?<![a-z0-9]){re.escape(suf)}(?![a-z0-9])", low):
                return True
        return False

    # Image / CDN paths are often lowercase; compare case-insensitively on domain-stripped text.
    if tail_l in image_s or _mark_in_cdn_path(image_raw, tail_u):
        score += 120
    if tail_l in title_s:
        score += 90
    if tail_l in link_s:
        score += 70
    if tail_l in snippet_s:
        score += 50
    if tail_l in text_blob:
        score += 30

    hits = sum(
        [
            tail_l in image_s or _mark_in_cdn_path(image_raw, tail_u),
            tail_l in title_s,
            tail_l in link_s,
            tail_l in snippet_s,
        ]
    )
    if hits >= 2:
        score += 40

    if not title_raw.strip() and (tail_l in image_s or _mark_in_cdn_path(image_raw, tail_u)):
        score += 30

    conflict_blob = strip_domains(f"{title_raw} {snippet_raw} {link_raw}")
    conflict_u = conflict_blob.upper()
    detected = {normalize_tail_token(x) for x in _N_TAIL_CONFLICT_V3.findall(conflict_u)}
    for t in detected:
        if not t or t == tail_u or t in _IGNORED_TAIL_LIKE_TOKENS:
            continue
        score -= 200
    return score


def build_aircraft_image_search_queries(
    *,
    canonical_tail: Optional[str],
    manufacturer: Optional[str],
    model: Optional[str],
) -> List[str]:
    """
    Smart query fan-out: tail-specific site-biased strings vs structured make/model facets.

    Part 2 spec — tail branch uses four concrete patterns; model branch uses three.
    """
    tail = normalize_tail_token(canonical_tail or "")
    if tail:
        return [
            f"{tail} aircraft",
            f"{tail} jet",
            f"{tail} site:jetphotos.com",
            f"{tail} site:planespotters.net",
        ]
    man = (manufacturer or "").strip()
    mdl = (model or "").strip()
    mm = compose_manufacturer_model_phrase(man, mdl) if (man or mdl) else ""
    mm = normalize_aircraft_name(mm) if mm else ""
    if len(mm) < 2:
        return []
    return [
        f"{mm} aircraft exterior",
        f"{mm} cabin",
        f"{mm} private jet",
        f"{mm} cockpit",
        f"{mm} interior",
        f"{mm} walkaround",
    ]


def _domain_priority_score(url: str, *, mode: str = "model") -> int:
    """Backward-compatible name: delegates to tail vs model domain tables."""
    return _domain_score_for_query_mode(url, mode)


def _avbuyer_penalty(url: str, source: str) -> int:
    blob = f"{url} {source}".lower()
    return -400 if "avbuyer" in blob else 0


def _other_tails_in_blob(canonical_tail: str, blob: str) -> List[str]:
    """Any registration-like marks in ``blob`` other than the canonical tail → conflict."""
    canon = normalize_tail_token(canonical_tail)
    out: List[str] = []
    for t in _all_registration_like_tokens(blob or ""):
        nt = normalize_tail_token(t)
        if nt == canon:
            continue
        if nt in _IGNORED_TAIL_LIKE_TOKENS:
            continue
        out.append(t)
    return out


def strict_tail_search_hit_confirmed(canonical_tail: str, row: Dict[str, Any]) -> bool:
    """
    Legacy “confirmed” gate — now implemented via :func:`compute_tail_match_score`
    (``>= 120`` ≈ confirmed tier).
    """
    tail = normalize_tail_token(canonical_tail)
    if not tail:
        return False
    return classify_tail_match_confidence(compute_tail_match_score(row, tail)) == "confirmed"


def _model_match_score(blob: str, marketing_type: str) -> int:
    if not marketing_type or not blob:
        return 0
    low = blob.lower()
    mt = marketing_type.strip().lower()
    if len(mt) < 4:
        return 0
    if mt in low:
        return 350
    parts = [p for p in mt.split() if len(p) >= 4]
    hits = sum(1 for p in parts if p in low)
    return 220 if hits >= 2 else (120 if hits == 1 else 0)


def _dedupe_rows_by_url(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        u = (r.get("url") or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(r)
    return out


def fetch_ranked_searchapi_aircraft_images(
    *,
    queries: List[str],
    canonical_tail: Optional[str],
    strict_tail_mode: bool,
    marketing_type_for_model_match: Optional[str],
    per_query_results: int = 16,
    max_out: int = 5,
    user_query: str = "",
    gallery_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run Bing image queries, score (tail confidence or model match), rank, cap per-domain, return gallery rows.

    Returns ``(images, meta)`` where ``meta`` may include empty-state UX and tail confidence notes.
    """
    meta_out: Dict[str, Any] = {}
    mode = "tail" if (strict_tail_mode and (canonical_tail or "").strip()) else "model"
    intent = detect_query_image_intent(user_query)

    merged: List[Dict[str, Any]] = []
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        batch = search_aircraft_images(q, num_results=per_query_results)
        for b in batch:
            url = b.get("url") or ""
            title = b.get("title") or ""
            source = b.get("source") or ""
            page = (b.get("_source_page") or "").strip() if isinstance(b, dict) else ""
            row = {
                "url": url,
                "title": title,
                "source": source,
                "_source_page": page,
            }
            if strict_tail_mode and canonical_tail:
                ts = compute_tail_match_score(row, str(canonical_tail).strip())
                if ts < MIN_TAIL_MATCH_SCORE:
                    continue
                conf = classify_tail_match_confidence(ts)
                if conf is None:
                    continue
                row["_tail_match_score"] = ts
                row["_tail_confidence"] = conf
            elif (marketing_type_for_model_match or "").strip():
                blob_chk = f"{url} {title} {source} {page}"
                from services.consultant_aircraft_images import (
                    _derive_model_negative_tokens,
                    _model_tokens_match_strict,
                )

                mt = marketing_type_for_model_match.strip()
                low_bc = blob_chk.lower()
                neg = _derive_model_negative_tokens(mt)
                if neg and any(n.lower() in low_bc for n in neg):
                    continue
                if not _model_tokens_match_strict(blob_chk, mt):
                    continue
            merged.append(row)

    merged = _dedupe_rows_by_url(merged)

    canon_tail = (
        normalize_tail_token(str(canonical_tail).strip())
        if strict_tail_mode and (canonical_tail or "").strip()
        else ""
    )
    agreement_boost = 0.0
    if canon_tail:
        tail_hits = Counter()
        for r in merged:
            bag = (
                (r.get("title") or "")
                + (r.get("snippet") or "")
                + (r.get("link") or r.get("_source_page") or "")
                + (r.get("url") or r.get("image") or "")
            )
            if canon_tail in strip_domains(bag).upper():
                tail_hits[canon_tail] += 1
        agreement_boost = float(min(50, tail_hits[canon_tail] * 10))

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for row in merged:
        url = str(row.get("url") or "")
        title = str(row.get("title") or "")
        source = str(row.get("source") or "")
        page = str(row.get("_source_page") or "")
        blob = f"{url} {title} {source} {page}"
        score = float(_domain_score_for_query_mode(url, mode))
        score += float(_avbuyer_penalty(url, source))
        if strict_tail_mode and canonical_tail:
            ts = int(row.get("_tail_match_score") or compute_tail_match_score(row, str(canonical_tail).strip()))
            score += float(ts) * 0.35
            score += agreement_boost
        elif (marketing_type_for_model_match or "").strip():
            score += float(_model_match_score(blob, marketing_type_for_model_match))
        score = apply_intent_boost(score, row, intent)
        scored.append((score, row))

    scored.sort(key=lambda t: t[0], reverse=True)

    def _row_is_avbuyer(row: Dict[str, Any]) -> bool:
        return "avbuyer" in f"{row.get('url', '')}{row.get('source', '')}".lower()

    non_ab = [(s, r) for s, r in scored if not _row_is_avbuyer(r)]
    high_floor = 750 if mode == "model" else 700
    high_tier = sum(
        1 for s, r in non_ab if _domain_score_for_query_mode(str(r.get("url") or ""), mode) >= high_floor
    )
    if high_tier >= 3:
        scored = [(s, r) for s, r in scored if not _row_is_avbuyer(r)]

    # Tail: if any ``confirmed`` tier exists, drop ``probable``-only rows for the final list.
    if strict_tail_mode and canonical_tail:
        any_conf = any(
            (r.get("_tail_confidence") or "") == "confirmed" for _, r in scored
        )
        if any_conf:
            scored = [(s, r) for s, r in scored if (r.get("_tail_confidence") or "") == "confirmed"]

    domain_counts: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    for _, row in scored:
        url = str(row.get("url") or "").strip()
        if not url:
            continue
        dom = extract_domain(url) or "_unknown"
        if domain_counts.get(dom, 0) >= MAX_PER_DOMAIN:
            continue
        page_u = str(row.get("_source_page") or "").strip() or None
        item: Dict[str, Any] = {
            "url": url,
            "source": "searchapi_bing_images",
            "description": (str(row.get("title") or "").strip() or None),
            "page_url": page_u,
            "lookup_key": None,
        }
        if strict_tail_mode and canonical_tail:
            item["tail_match_confidence"] = row.get("_tail_confidence")
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
        out.append(item)
        if len(out) >= max_out:
            break

    meta_out["consultant_searchapi_gallery_mode"] = mode
    if strict_tail_mode and canonical_tail:
        if not out:
            meta_out["consultant_gallery_empty"] = True
            meta_out["consultant_gallery_message"] = "No verified images found for this aircraft."
            meta_out["consultant_gallery_suggestions"] = [
                "View similar aircraft by model",
                "Broaden search scope",
            ]
        elif all((r.get("tail_match_confidence") or "") == "probable" for r in out):
            meta_out["consultant_gallery_tail_confidence"] = "probable"
            meta_out["consultant_gallery_tail_note"] = (
                "Images may not exactly match this tail number"
            )

    if gallery_meta is not None:
        gallery_meta.update(meta_out)
    return out, meta_out


def resolve_queries_for_consultant_gallery(
    *,
    user_query: str,
    phly_rows: List[Dict[str, Any]],
    required_tail: Optional[str],
    strict_tail_mode: bool,
    required_marketing_type: Optional[str],
    strict_model_mode: bool,
) -> Tuple[List[str], Optional[str], Optional[str]]:
    """
    Returns (queries, canonical_tail_for_strict_filter, marketing_type_for_scoring).

    ``canonical_tail`` is set when strict tail gallery mode is active (user message contained tail).
    """
    tail = normalize_tail_token(required_tail or "")
    if strict_tail_mode and tail:
        return build_aircraft_image_search_queries(canonical_tail=tail, manufacturer=None, model=None), tail, None

    if strict_model_mode and (required_marketing_type or "").strip():
        mm = normalize_aircraft_name(str(required_marketing_type).strip())
        # Use the full marketing string for all facets (manufacturer/model may be one token).
        qs = build_aircraft_image_search_queries(canonical_tail=None, manufacturer="", model=mm)
        return qs, None, mm

    # Prefer Phly make/model when present (registry-resolved type for tail workflows).
    for r in phly_rows[:4]:
        man = (r.get("manufacturer") or "").strip()
        mdl = (r.get("model") or "").strip()
        if man or mdl:
            mm = compose_manufacturer_model_phrase(man, mdl)
            mm = normalize_aircraft_name(mm)
            qs = build_aircraft_image_search_queries(canonical_tail=None, manufacturer="", model=mm)
            return qs, None, mm or None

    # Last resort: infer manufacturer/model tokens from user text only (no tail → no strict tail mode here).
    try:
        from rag.consultant_query_expand import _detect_manufacturers, _detect_models

        blob = (user_query or "").strip()
        mans = _detect_manufacturers(blob.lower())
        mdls = _detect_models(blob)
        man = mans[0] if mans else ""
        mdl = mdls[0] if mdls else ""
        if man or mdl:
            mm = compose_manufacturer_model_phrase(man, mdl)
            mm = normalize_aircraft_name(mm)
            qs = build_aircraft_image_search_queries(canonical_tail=None, manufacturer="", model=mm)
            return qs, None, mm or None
    except Exception:
        pass

    # Ultra-generic fallback (still avoids inventing a tail).
    q = (user_query or "").strip()
    if len(q) >= 3:
        return [f"{q[:120]} aircraft exterior"], None, None
    return [], None, None
