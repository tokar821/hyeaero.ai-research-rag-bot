"""
Aircraft image gallery via SearchAPI.io (replaces Tavily *image* retrieval only).

Default engine is **Google Images Light** (`google_images_light`) — same `images[]` shape as
`google_images` per SearchAPI docs; override with ``SEARCHAPI_IMAGE_ENGINE``.

Web snippet search remains on Tavily elsewhere; this module is scoped to image URLs only.
Optional Tavily **domain** hints (``SEARCHAPI_TAVILY_DOMAIN_VERIFY``) add authority for unknown hosts.
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


def searchapi_image_engine() -> str:
    """``SEARCHAPI_IMAGE_ENGINE``: ``google_images_light`` (default), ``google_images``, or ``bing_images``."""
    v = (os.getenv("SEARCHAPI_IMAGE_ENGINE") or "google_images_light").strip().lower()
    if v in ("google_images", "google_images_light", "bing_images"):
        return v
    logger.warning("Unknown SEARCHAPI_IMAGE_ENGINE=%r; using google_images_light", v)
    return "google_images_light"


def searchapi_literal_user_query_mode() -> bool:
    """
    When true (default), consultant SearchAPI uses **one** ``q`` string taken from the user question
    (trimmed / whitespace-collapsed), not multi-query fan-out or marketing-type rewrites.
    Set ``SEARCHAPI_LITERAL_USER_QUERY=0`` to restore legacy Bing-style query fan-out.
    """
    return (os.getenv("SEARCHAPI_LITERAL_USER_QUERY") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def searchapi_gallery_source_label() -> str:
    e = searchapi_image_engine()
    if e.startswith("google"):
        return "searchapi_google_images"
    return "searchapi_bing_images"


def build_searchapi_image_request_params(*, q: str, num_results: int) -> Optional[Dict[str, Any]]:
    """Query params for GET ``/api/v1/search`` (Bing vs Google engines differ on safe-search keys)."""
    raw_q = " ".join((q or "").strip().split())
    if len(raw_q) < 1:
        return None
    api_key = (os.getenv("SEARCHAPI_API_KEY") or "").strip()
    if not api_key:
        return None
    engine = searchapi_image_engine()
    n = max(8, min(20, int(num_results)))
    params: Dict[str, Any] = {"engine": engine, "q": raw_q[:200], "api_key": api_key, "num": n}
    if engine == "bing_images":
        params["safe_search"] = "moderate"
    else:
        params["safe"] = (os.getenv("SEARCHAPI_IMAGE_SAFE") or "blur").strip() or "blur"
        params["gl"] = (os.getenv("SEARCHAPI_IMAGE_GL") or "us").strip() or "us"
        params["hl"] = (os.getenv("SEARCHAPI_IMAGE_HL") or "en").strip() or "en"
    return params


# Buyer-oriented tail gallery: listings first, then spotter sites.
_DOMAIN_SCORES_TAIL: Tuple[Tuple[str, int], ...] = (
    ("controller.com", 1000),
    ("aircraftexchange", 950),
    ("jetphotos.", 900),
    ("planespotters.", 850),
    ("globalair.", 700),
    ("avbuyer.", 200),
)
# Visual / type browsing: spotters first; then cabin/interior-rich hosts Google Images often returns
# (YouTube, Instagram, Reddit, Pinterest, charter/marketing sites) so they are not buried at score 0.
_DOMAIN_SCORES_MODEL: Tuple[Tuple[str, int], ...] = (
    ("jetphotos.", 1000),
    ("planespotters.", 950),
    ("controller.com", 800),
    ("aircraftexchange", 750),
    ("globalair.", 700),
    ("avbuyer.", 300),
    # Charter / OEM / news (interior marketing pages).
    ("aircharterservice.com", 720),
    ("privatejetcardcomparisons.com", 710),
    ("flexjet.com", 710),
    ("claylacy.com", 710),
    ("wingaviation", 700),
    ("skybirdaviation", 700),
    ("duncanaviation.com", 700),
    ("globalcharter.com", 700),
    ("pacmin.com", 700),
    ("simpleflying.com", 680),
    ("simpleflyingimages.com", 680),
    # Social / video thumbnails (common for cabin walkthroughs).
    ("ytimg.com", 700),
    ("youtube.com", 690),
    ("youtu.be", 690),
    ("cdninstagram.com", 690),
    ("instagram.com", 690),
    ("fbcdn.net", 680),
    ("facebook.com", 680),
    ("pinimg.com", 690),
    ("pinterest.com", 680),
    ("pinterest.", 680),
    ("redditmedia.com", 680),
    ("redditstatic.com", 670),
    ("redd.it", 680),
    ("reddit.com", 660),
    # OEM / manufacturer / aviation press (image may be on CDN; source page often carries the signal).
    ("dassault-aviation.com", 880),
    ("dassaultfalcon.com", 880),
    ("dassaultfalcon.", 880),
    ("falconjet.com", 870),
    ("gulfstream.com", 880),
    ("bombardier.com", 870),
    ("embraer.com", 860),
    ("bjtonline.com", 760),
    ("ainonline.com", 760),
    ("aviationweek.com", 740),
    ("flightglobal.com", 730),
    ("corporatejetinvestor.com", 720),
    ("flycorporate.com", 710),
)

# Longer keys first so ``G650ER`` wins over ``G650``.
NORMALIZATION_MAP: Tuple[Tuple[str, str], ...] = (
    ("G650ER", "Gulfstream G650ER"),
    ("G650", "Gulfstream G650"),
    ("G800", "Gulfstream G800"),
    ("G700", "Gulfstream G700"),
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

# Legacy default when env unset (tail galleries stay tight on duplicate CDNs).
_MAX_PER_DOMAIN_TAIL_DEFAULT = 2
_MAX_PER_DOMAIN_MODEL_DEFAULT = 48


def searchapi_preserve_google_image_rank_order() -> bool:
    """
    When true (default), **model-mode** gallery ranking follows SearchAPI / Google ``images[].position``
    (per query, in query order) after strict verification — not a rebuilt sort from our host priors.

    Within the first ``SEARCHAPI_AVIATION_RANKUP_WINDOW`` positions (per merged ``_api_rank``),
    results are **re-ordered by aviation-domain authority** while keeping Google order as the
    backbone (see :func:`searchapi_aviation_rankup_window`).

    Set ``SEARCHAPI_PRESERVE_GOOGLE_RANK_ORDER=0`` to restore domain-heavy ordering.
    """
    return (os.getenv("SEARCHAPI_PRESERVE_GOOGLE_RANK_ORDER") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def searchapi_image_relax_model_match() -> bool:
    """
    When true, SearchAPI model-mode gallery uses :func:`_model_tokens_match_searchapi_relaxed`
    instead of strict token boundaries (still rejects negative model tokens).

    Set ``SEARCHAPI_IMAGE_RELAX_MODEL_MATCH=1``.
    """
    return (os.getenv("SEARCHAPI_IMAGE_RELAX_MODEL_MATCH") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def searchapi_skip_premium_image_validation() -> bool:
    """When true, skip :func:`apply_premium_image_validation` for SearchAPI merged rows."""
    return (os.getenv("SEARCHAPI_SKIP_PREMIUM_IMAGE_VALIDATION") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _searchapi_tavily_domain_timeout_s() -> float:
    try:
        return max(4.0, min(30.0, float((os.getenv("SEARCHAPI_TAVILY_DOMAIN_TIMEOUT") or "9").strip())))
    except ValueError:
        return 9.0


def searchapi_aviation_rankup_window() -> int:
    """
    When preserve-Google rank is on, merge consecutive Google ranks into buckets of this size;
    within each bucket, **higher** aviation / OEM / spotter authority sorts **earlier**.

    Set ``SEARCHAPI_AVIATION_RANKUP_WINDOW=0`` to keep strict API position order (no rank-up).
    Default ``8`` — typical first screen of Google Images.
    """
    try:
        v = (os.getenv("SEARCHAPI_AVIATION_RANKUP_WINDOW") or "8").strip()
        if v.lower() in ("0", "off", "false", "no"):
            return 0
        return max(1, min(50, int(v)))
    except ValueError:
        return 8


def searchapi_max_images_per_domain(*, mode: str) -> int:
    """
    Max images kept per image **host** when building the gallery (dedupe spam CDNs).

    - ``SEARCHAPI_MAX_PER_IMAGE_DOMAIN``: single override for both modes (clamped 1–80).
    - Otherwise: ``2`` for strict tail mode, ``48`` for model mode (trust diverse Google results).
    """
    raw = (os.getenv("SEARCHAPI_MAX_PER_IMAGE_DOMAIN") or "").strip()
    if raw:
        try:
            return max(1, min(80, int(raw)))
        except ValueError:
            pass
    return _MAX_PER_DOMAIN_TAIL_DEFAULT if mode == "tail" else _MAX_PER_DOMAIN_MODEL_DEFAULT


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


def _domain_score_on_lowercase_blob(low: str, mode: str) -> int:
    table = _DOMAIN_SCORES_TAIL if mode == "tail" else _DOMAIN_SCORES_MODEL
    best = 0
    for frag, sc in table:
        if frag in low:
            best = max(best, sc)
    return best


def _domain_score_for_query_mode(url: str, mode: str) -> int:
    return _domain_score_on_lowercase_blob((url or "").lower(), mode)


def _off_topic_image_rankup_penalty(low: str) -> int:
    """
    Reduce aviation authority for obvious non-aviation hosts/pages so Google junk (parks, weddings)
    does not rank above OEM / trade press within the same rank-up window.
    """
    if not low:
        return 0
    pen = 0
    if ("park" in low or "parks." in low) and any(
        x in low for x in ("county", "state", "city.", "municipal", "recreation", "trail", "campground")
    ):
        pen += 520
    if any(x in low for x in ("wedding", "bridal", "engagement", "zillow", "realtor", "apartment", "condo listing")):
        pen += 480
    if "stockphoto" in low or "shutterstock" in low or "gettyimages" in low or "istockphoto" in low:
        pen += 350
    return min(pen, 900)


def aviation_rankup_authority_score(
    url: str,
    page: str,
    source: str,
    title: str,
    *,
    mode: str,
    tavily_host_boosts: Optional[Dict[str, int]] = None,
) -> int:
    """
    Single score for rank-up: best domain-table hit across image URL, source page, source label,
    and title, minus off-topic penalties, plus optional Tavily-derived boost for unknown domains.
    """
    low = f"{url} {page} {source} {title}".lower()
    base = _domain_score_on_lowercase_blob(low, mode)
    core = max(0, int(base) - int(_off_topic_image_rankup_penalty(low)))
    extra = 0
    if tavily_host_boosts:
        from services.tavily_aviation_domain_boost import normalize_source_host

        ph = normalize_source_host(extract_domain(page))
        uh = normalize_source_host(extract_domain(url))
        extra = max(int(tavily_host_boosts.get(ph, 0)), int(tavily_host_boosts.get(uh, 0)))
    return core + min(500, int(extra))


def searchapi_aircraft_images_enabled() -> bool:
    return bool((os.getenv("SEARCHAPI_API_KEY") or "").strip())


def search_aircraft_images(query: str, *, num_results: int = 15, timeout: float = 28.0) -> List[Dict[str, str]]:
    """
    Reusable image search via SearchAPI.io (Google Images / Light or Bing per ``SEARCHAPI_IMAGE_ENGINE``).

    Returns normalized rows aligned with SearchAPI ``images[]``:

    - ``url``: ``original.link`` (HTTPS image asset)
    - ``title``: image title
    - ``source``: ``source.name`` (fallback: image host)
    - ``_source_page``: ``source.link`` (result / listing page — used for strict matching)
    - ``_position``: 1-based ``position`` from the API (used to preserve Google rank order)
    """
    raw_q = (query or "").strip()
    if not raw_q:
        return []
    params = build_searchapi_image_request_params(q=raw_q, num_results=num_results)
    if not params:
        return []
    try:
        r = requests.get(SEARCHAPI_SEARCH_URL, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning("SearchAPI %s image request failed: %s", params.get("engine"), e)
        return []

    images = data.get("images") if isinstance(data, dict) else None
    if not isinstance(images, list):
        return []

    try:
        limit = int(params.get("num") or 15)
    except (TypeError, ValueError):
        limit = 15
    limit = max(1, min(50, limit))

    out: List[Dict[str, str]] = []
    for bi, item in enumerate(images[:limit]):
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
        pos_raw = item.get("position")
        try:
            pos_i = int(pos_raw) if pos_raw is not None and str(pos_raw).strip() != "" else bi + 1
        except (TypeError, ValueError):
            pos_i = bi + 1
        # ``_source_page`` is used only for strict tail verification (title/url/source page text).
        out.append(
            {
                "url": url,
                "title": title,
                "source": source_label,
                "_source_page": src_link,
                "_position": str(pos_i),
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
    # Compact ``G650 interior``-style strings first (high recall on Google Images), then legacy fan-out.
    try:
        from services.image_query_decision_engine import _prepend_searchapi_high_recall_queries

        pre = (
            _prepend_searchapi_high_recall_queries(mm, "cabin")
            + _prepend_searchapi_high_recall_queries(mm, "exterior")
            + _prepend_searchapi_high_recall_queries(mm, "cockpit")
        )
    except Exception:
        pre = []
    body = [
        f"{mm} aircraft exterior",
        f"{mm} cabin",
        f"{mm} private jet",
        f"{mm} cockpit",
        f"{mm} interior",
        f"{mm} walkaround",
    ]
    out: List[str] = []
    seen: set[str] = set()
    for q in pre + body:
        k = (q or "").strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(q.strip())
    return out


def _domain_priority_score(url: str, *, mode: str = "model") -> int:
    """Backward-compatible name: delegates to tail vs model domain tables."""
    return _domain_score_for_query_mode(url, mode)


def _avbuyer_penalty(url: str, source: str) -> int:
    blob = f"{url} {source}".lower()
    return -400 if "avbuyer" in blob else 0


def _generic_stock_image_penalty(url: str, title: str, source: str) -> float:
    """Down-rank obvious stock / AI / logistics spam that Bing/Google sometimes surface for jet queries."""
    blob = f"{url} {title} {source}".lower()
    needles = (
        "freepik",
        "shutterstock",
        "dreamstime",
        "adobe stock",
        "123rf",
        "istockphoto",
        "stock photo",
        "generative ai",
        "generative ia",
        "ai image",
        "needs a load aircraft",
        "mounting needs",
        "logistics mounting",
    )
    return -650.0 if any(x in blob for x in needles) else 0.0


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


def _marketing_type_for_scoring(
    user_query: str,
    phly_rows: List[Dict[str, Any]],
    required_marketing_type: Optional[str],
) -> Optional[str]:
    """Canonical marketing string for **ranking boosts** only (not used as SearchAPI ``q`` in literal mode)."""
    if (required_marketing_type or "").strip():
        return normalize_aircraft_name(str(required_marketing_type).strip())
    for r in (phly_rows or [])[:4]:
        man = (r.get("manufacturer") or "").strip()
        mdl = (r.get("model") or "").strip()
        if man or mdl:
            mm = compose_manufacturer_model_phrase(man, mdl)
            mm = normalize_aircraft_name(mm)
            return mm or None
    try:
        from rag.consultant_query_expand import _detect_manufacturers, _detect_models

        blob = (user_query or "").strip()
        mans = _detect_manufacturers(blob.lower())
        mdls = _detect_models(blob)
        man = mans[0] if mans else ""
        mdl = mdls[0] if mdls else ""
        if man or mdl:
            mm = compose_manufacturer_model_phrase(man, mdl)
            return normalize_aircraft_name(mm) if mm else None
    except Exception:
        pass
    return None


def _facet_addon_words_from_user_query(user_query: str, tail: str) -> List[str]:
    """Up to two allowed visual tokens from the user line when they also typed the registration."""
    tail_c = tail.replace(" ", "").upper()
    compact = re.sub(r"\s+", "", (user_query or "").upper())
    if not tail_c or tail_c not in compact:
        return []
    low = (user_query or "").lower()
    out: List[str] = []
    for w in ("exterior", "cockpit", "cabin", "interior", "galley", "ramp", "walkaround"):
        if w in low and w not in out:
            out.append(w)
        if len(out) >= 2:
            break
    return out


def _literal_single_search_q(
    user_query: str,
    *,
    strict_tail_mode: bool,
    required_tail: Optional[str],
    strict_model_mode: bool,
    required_marketing_type: Optional[str],
    mm_fallback: Optional[str],
) -> Optional[str]:
    """
    Single ``q`` for SearchAPI (literal / fallback paths).

    When ``required_tail`` resolves to a registration, **never** use vague English alone
    (*\"can I see that\"*) as ``q`` — image search is anchored on the tail (optionally plus
    facet words copied from the user line when they also typed the mark).
    """
    q = " ".join((user_query or "").strip().split())
    tail = normalize_tail_token(required_tail or "")
    if tail:
        addons = _facet_addon_words_from_user_query(q, tail)
        if addons:
            return f"{tail} {' '.join(addons)}".strip()[:200]
        # Registration-only: avoids unrelated Google Image hits from conversational text.
        return tail[:200]
    if len(q) >= 3:
        return q[:200]
    if strict_model_mode and (required_marketing_type or "").strip():
        return str(required_marketing_type).strip()[:200]
    if mm_fallback and len(mm_fallback.strip()) >= 3:
        return mm_fallback.strip()[:200]
    return None


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
    premium_intent: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run SearchAPI image search (Google or Bing), score, rank, cap per-domain, return gallery rows.

    Model mode (default): after strict verification on **image URL, title, source name, and source page
    link**, ordering follows Google’s ``position`` in **buckets** (see ``SEARCHAPI_AVIATION_RANKUP_WINDOW``):
    within each bucket, **aviation / OEM / trade-press** hosts rank above obvious off-topic pages while
    keeping overall API order. Set ``SEARCHAPI_PRESERVE_GOOGLE_RANK_ORDER=0`` for legacy host-heavy
    scoring. Set ``SEARCHAPI_AVIATION_RANKUP_WINDOW=0`` to disable rank-up inside preserve mode.

    Optional **image rank & filter** pass (``SEARCHAPI_IMAGE_RANK_FILTER_ENGINE=1``): deterministic
    broker-style filter on titles/URLs (houses/hotels, wrong section, weak aircraft match). Requires
    at least **two** surviving images or returns an **empty** gallery (precision over filler). Meta
    may include ``image_rank_filter_engine``.

    Returns ``(images, meta)`` where ``meta`` may include empty-state UX and tail confidence notes.
    """
    meta_out: Dict[str, Any] = {"consultant_searchapi_image_engine": searchapi_image_engine()}
    mode = "tail" if (strict_tail_mode and (canonical_tail or "").strip()) else "model"
    intent = detect_query_image_intent(user_query)
    preserve_google = (
        searchapi_preserve_google_image_rank_order()
        and not strict_tail_mode
        and bool((marketing_type_for_model_match or "").strip())
    )
    meta_out["searchapi_preserve_google_rank_order"] = bool(preserve_google)
    rankup_w = searchapi_aviation_rankup_window() if preserve_google else 0
    meta_out["searchapi_aviation_rankup_window"] = int(rankup_w)
    meta_out["searchapi_max_per_image_domain"] = searchapi_max_images_per_domain(mode=mode)

    merged: List[Dict[str, Any]] = []
    for qi, q in enumerate(queries):
        q = (q or "").strip()
        if not q:
            continue
        batch = search_aircraft_images(q, num_results=per_query_results)
        for bi, b in enumerate(batch):
            url = b.get("url") or ""
            title = b.get("title") or ""
            source = b.get("source") or ""
            page = (b.get("_source_page") or "").strip() if isinstance(b, dict) else ""
            try:
                pos_i = int(str((b or {}).get("_position") or (bi + 1)).strip())
            except ValueError:
                pos_i = bi + 1
            row = {
                "url": url,
                "title": title,
                "source": source,
                "_source_page": page,
                "_api_rank": qi * 10_000 + pos_i,
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
                # Model alignment: strict tokens by default; optional relaxed match (still rejects negatives).
                blob_chk = f"{url} {title} {source} {page}"
                from services.consultant_aircraft_images import (
                    _derive_model_negative_tokens,
                    _model_tokens_match_searchapi_relaxed,
                    _model_tokens_match_strict,
                )

                mt = marketing_type_for_model_match.strip()
                low_bc = blob_chk.lower()
                neg = _derive_model_negative_tokens(mt)
                if neg and any(n.lower() in low_bc for n in neg):
                    continue
                _match_fn = (
                    _model_tokens_match_searchapi_relaxed
                    if searchapi_image_relax_model_match()
                    else _model_tokens_match_strict
                )
                if not _match_fn(blob_chk, mt):
                    continue
            merged.append(row)

    merged = _dedupe_rows_by_url(merged)

    from services.consultant_image_search_orchestrator import (
        PREMIUM_VERIFIED_IMAGE_FAILURE,
        apply_premium_image_validation,
    )

    pre_validation_n = len(merged)
    pi = premium_intent or {}
    if searchapi_skip_premium_image_validation():
        _applied_premium = False
    else:
        merged, _applied_premium = apply_premium_image_validation(merged, pi)
    premium_stripped_all = bool(
        pi.get("validate_images") and pre_validation_n > 0 and len(merged) == 0
    )

    tavily_host_boosts: Dict[str, int] = {}
    meta_out["searchapi_tavily_domain_verify"] = False
    if preserve_google:
        from services.tavily_aviation_domain_boost import (
            prefetch_tavily_domain_boosts,
            searchapi_tavily_domain_max_lookups,
            searchapi_tavily_domain_verify_enabled,
        )

        if searchapi_tavily_domain_verify_enabled():
            meta_out["searchapi_tavily_domain_verify"] = True
            host_order: List[str] = []
            for r in merged:
                page_u = str(r.get("_source_page") or "")
                url_u = str(r.get("url") or "")
                tab = _domain_score_on_lowercase_blob(f"{page_u} {url_u}".lower(), mode)
                if tab >= 550:
                    continue
                dom = extract_domain(page_u) or extract_domain(url_u)
                if dom:
                    host_order.append(dom)
            tavily_host_boosts = prefetch_tavily_domain_boosts(
                host_order,
                max_lookups=searchapi_tavily_domain_max_lookups(),
                timeout_per_host=_searchapi_tavily_domain_timeout_s(),
            )
            meta_out["searchapi_tavily_domain_hosts_scored"] = len(tavily_host_boosts)

    _norm_host = None
    if tavily_host_boosts:
        from services.tavily_aviation_domain_boost import normalize_source_host as _norm_host

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
        if preserve_google:
            api_r = int(row.get("_api_rank") or 9_999_999)
            score = 5_000_000.0 - float(api_r)
            score += float(_model_match_score(blob, str(marketing_type_for_model_match).strip())) * 2.0
            score += float(_domain_score_for_query_mode(url, mode)) * 0.0001
            score += float(_avbuyer_penalty(url, source)) * 0.15
            score += _generic_stock_image_penalty(url, title, source)
            tw = 0
            if tavily_host_boosts and _norm_host is not None:
                tw = max(
                    int(tavily_host_boosts.get(_norm_host(extract_domain(page)), 0)),
                    int(tavily_host_boosts.get(_norm_host(extract_domain(url)), 0)),
                )
            score += float(tw) * 0.06
        else:
            score = float(_domain_score_for_query_mode(url, mode))
            score += float(_avbuyer_penalty(url, source))
            score += _generic_stock_image_penalty(url, title, source)
            if strict_tail_mode and canonical_tail:
                ts = int(row.get("_tail_match_score") or compute_tail_match_score(row, str(canonical_tail).strip()))
                score += float(ts) * 0.35
                score += agreement_boost
            elif (marketing_type_for_model_match or "").strip():
                score += float(_model_match_score(blob, marketing_type_for_model_match))
        score = apply_intent_boost(score, row, intent)
        scored.append((score, row))

    if preserve_google and rankup_w > 0:

        def _preserve_google_rankup_sort_key(t: Tuple[float, Dict[str, Any]]) -> Tuple[int, int, float]:
            s, row = t
            api_r = int(row.get("_api_rank") or 9_999_999)
            ar = aviation_rankup_authority_score(
                str(row.get("url") or ""),
                str(row.get("_source_page") or ""),
                str(row.get("source") or ""),
                str(row.get("title") or ""),
                mode=mode,
                tavily_host_boosts=tavily_host_boosts or None,
            )
            bucket = api_r // rankup_w
            # Ascending: smaller bucket first; within bucket higher aviation first; then higher composite.
            return (bucket, -ar, -s)

        scored.sort(key=_preserve_google_rankup_sort_key)
    else:
        scored.sort(key=lambda t: t[0], reverse=True)

    def _row_is_avbuyer(row: Dict[str, Any]) -> bool:
        return "avbuyer" in f"{row.get('url', '')}{row.get('source', '')}".lower()

    if not preserve_google:
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
        max_per_dom = searchapi_max_images_per_domain(mode=mode)
        if domain_counts.get(dom, 0) >= max_per_dom:
            continue
        page_u = str(row.get("_source_page") or "").strip() or None
        item: Dict[str, Any] = {
            "url": url,
            "source": searchapi_gallery_source_label(),
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

    if out and (marketing_type_for_model_match or canonical_tail or "").strip():
        from services.aviation_image_rank_filter_engine import (
            apply_rank_filter_to_gallery_items,
            searchapi_image_rank_filter_engine_enabled,
        )

        if searchapi_image_rank_filter_engine_enabled():
            _sec = intent or (premium_intent or {}).get("image_type") or "interior"
            _qi = {
                "aircraft": (marketing_type_for_model_match or canonical_tail or "").strip(),
                "section": str(_sec).strip() or "interior",
                "type": str((premium_intent or {}).get("image_type") or _sec or "interior"),
            }
            _prev_n = len(out)
            out = apply_rank_filter_to_gallery_items(
                gallery_items=out,
                query_intent=_qi,
                max_out=max_out,
                gallery_meta=gallery_meta,
            )
            if _prev_n and not out:
                meta_out["consultant_gallery_empty"] = True
                meta_out["consultant_gallery_message"] = (
                    "No verified images met quality and relevance thresholds."
                )

    meta_out["consultant_searchapi_gallery_mode"] = mode
    if strict_tail_mode and canonical_tail:
        if not out:
            meta_out["consultant_gallery_empty"] = True
            if premium_stripped_all:
                meta_out["consultant_gallery_message"] = PREMIUM_VERIFIED_IMAGE_FAILURE
            else:
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
    elif not out and premium_stripped_all:
        meta_out["consultant_gallery_empty"] = True
        meta_out["consultant_gallery_message"] = PREMIUM_VERIFIED_IMAGE_FAILURE

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
) -> Tuple[List[str], Optional[str], Optional[str], Dict[str, Any]]:
    """
    Returns (queries, canonical_tail_for_strict_filter, marketing_type_for_scoring, premium_intent).

    Default literal mode with **``SEARCHAPI_PRECISION_QUERIES``** on (default): 3–5 short orchestrated
    queries from classified intent (HyeAero-style). Set ``SEARCHAPI_PRECISION_QUERIES=0`` for a
    single spacing-normalized user-string ``q``.

    ``canonical_tail`` is set when strict tail gallery mode is active (user message contained tail).
    """
    from services.consultant_image_search_orchestrator import (
        build_precision_image_search_queries,
        classify_premium_aviation_intent,
        searchapi_precision_queries_enabled,
    )

    tail = normalize_tail_token(required_tail or "")
    # Whenever the pipeline knows a registration, image ``q`` strings must be tail-led even if
    # ``strict_tail_page_match`` was not set (defensive — avoids "can I see that" as SearchAPI ``q``).
    strict_for_queries = bool(strict_tail_mode or tail)
    mm_score = _marketing_type_for_scoring(user_query, phly_rows, required_marketing_type)
    intent = classify_premium_aviation_intent(
        user_query,
        required_tail=required_tail,
        required_marketing_type=required_marketing_type,
        phly_rows=phly_rows,
    )

    if searchapi_literal_user_query_mode():
        if searchapi_precision_queries_enabled():
            qs, _iq_meta = build_precision_image_search_queries(
                intent,
                user_query=user_query,
                strict_tail_mode=strict_for_queries,
                required_tail=required_tail,
                required_marketing_type=required_marketing_type,
                phly_rows=phly_rows,
                mm_for_scoring=mm_score,
            )
            if _iq_meta.get("image_query_engine"):
                intent["image_query_engine"] = _iq_meta["image_query_engine"]
            if not qs:
                # Decision engine intentionally returned no queries (INVALID model, buying-only, …).
                if intent.get("suppress_image_search") or intent.get("type") == "INVALID":
                    qs = []
                else:
                    sq = _literal_single_search_q(
                        user_query,
                        strict_tail_mode=strict_for_queries,
                        required_tail=required_tail,
                        strict_model_mode=strict_model_mode,
                        required_marketing_type=required_marketing_type,
                        mm_fallback=mm_score,
                    )
                    qs = [sq] if sq else []
            tail_out = tail if tail else None
            return qs, tail_out, mm_score, intent

        sq = _literal_single_search_q(
            user_query,
            strict_tail_mode=strict_for_queries,
            required_tail=required_tail,
            strict_model_mode=strict_model_mode,
            required_marketing_type=required_marketing_type,
            mm_fallback=mm_score,
        )
        if not sq:
            return [], None, mm_score, intent
        tail_out = tail if tail else None
        return [sq], tail_out, mm_score, intent

    # Legacy: multi-query fan-out (``SEARCHAPI_LITERAL_USER_QUERY=0``).
    if strict_tail_mode and tail:
        return (
            build_aircraft_image_search_queries(canonical_tail=tail, manufacturer=None, model=None),
            tail,
            None,
            intent,
        )

    if strict_model_mode and (required_marketing_type or "").strip():
        mm = normalize_aircraft_name(str(required_marketing_type).strip())
        qs = build_aircraft_image_search_queries(canonical_tail=None, manufacturer="", model=mm)
        return qs, None, mm, intent

    for r in (phly_rows or [])[:4]:
        man = (r.get("manufacturer") or "").strip()
        mdl = (r.get("model") or "").strip()
        if man or mdl:
            mm = compose_manufacturer_model_phrase(man, mdl)
            mm = normalize_aircraft_name(mm)
            qs = build_aircraft_image_search_queries(canonical_tail=None, manufacturer="", model=mm)
            return qs, None, mm or None, intent

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
            return qs, None, mm or None, intent
    except Exception:
        pass

    q = (user_query or "").strip()
    if len(q) >= 3:
        return [f"{q[:120]} aircraft exterior"], None, None, intent
    return [], None, None, intent
