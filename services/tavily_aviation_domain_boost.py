"""
Optional Tavily lookup to score **unknown** image source domains for aviation relevance.

We cannot list every OEM, blog, or charter site. When the static domain table is uncertain
(low score) and ``SEARCHAPI_TAVILY_DOMAIN_VERIFY=1`` (recommended in production), we run a
**single** Tavily search per host (cached) and derive a boost from result titles/snippets — no second LLM.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Tuple

from services.tavily_owner_hint import TAVILY_SEARCH_URL, clamp_tavily_query

logger = logging.getLogger(__name__)

_CACHE: Dict[str, Tuple[int, float]] = {}
# score_int, monotonic_ts

_CDN_NO_SITE_MEANING = (
    "ytimg.com",
    "googleusercontent.com",
    "ggpht.com",
    "gstatic.com",
    "fbcdn.net",
    "cdninstagram.com",
    "pinimg.com",
    "redditmedia.com",
    "redditstatic.com",
    "twimg.com",
    "akamaized.net",
    "cloudfront.net",
    "amazonaws.com",
)

_POS_HINTS: Tuple[str, ...] = (
    "aviation",
    "aircraft",
    "airplane",
    "aeroplane",
    "business jet",
    "private jet",
    "charter",
    "airliner",
    "cockpit",
    "cabin interior",
    "flight deck",
    "FBO",
    "turboprop",
    "helicopter",
    "planespotter",
    "jetphotos",
    "for sale aircraft",
    "airframe",
    "MRO",
    "OEM",
    "dassault",
    "gulfstream",
    "bombardier",
    "embraer",
    "cessna",
)

_NEG_HINTS: Tuple[str, ...] = (
    "county park",
    "state park",
    "wedding",
    "bridal",
    "real estate",
    "zillow",
    "recipe",
    "casino",
    "football",
    "basketball",
    "vacation rental",
    "cabin rental",
    "gatlinburg",
    "racing sim",
    "furniture store",
    "interior designer",
    "book review",
)


def searchapi_tavily_domain_verify_enabled() -> bool:
    return (os.getenv("SEARCHAPI_TAVILY_DOMAIN_VERIFY") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def searchapi_tavily_domain_max_lookups() -> int:
    try:
        return max(0, min(20, int((os.getenv("SEARCHAPI_TAVILY_DOMAIN_MAX_LOOKUPS") or "8").strip())))
    except ValueError:
        return 8


def searchapi_tavily_domain_cache_ttl_s() -> float:
    try:
        hrs = float((os.getenv("SEARCHAPI_TAVILY_DOMAIN_CACHE_TTL_HOURS") or "168").strip())
        return max(300.0, min(86400.0 * 90, hrs * 3600.0))
    except ValueError:
        return 86400.0 * 7


def _tavily_disabled() -> bool:
    return (os.getenv("TAVILY_DISABLED") or "").strip().lower() in ("1", "true", "yes", "on")


def _normalize_host(host: str) -> str:
    h = (host or "").strip().lower()
    if h.startswith("www."):
        h = h[4:]
    return h


def normalize_source_host(host: str) -> str:
    """Normalize hostname for cache / boost map keys (strip ``www.``, lower-case)."""
    return _normalize_host(host)


def _host_skipped_for_tavily(host: str) -> bool:
    h = _normalize_host(host)
    if not h or "." not in h:
        return True
    return any(h == s or h.endswith("." + s) for s in _CDN_NO_SITE_MEANING)


def _score_snippets_for_aviation(blob: str) -> int:
    low = (blob or "").lower()
    pos = sum(2 for k in _POS_HINTS if k in low)
    neg = sum(3 for k in _NEG_HINTS if k in low)
    raw = pos * 16 - neg * 22
    return max(0, min(420, raw))


def _tavily_search_domain_rest(*, api_key: str, host: str, timeout: float) -> Dict[str, Any]:
    import requests

    q = clamp_tavily_query(
        f"{host} website what is it about business aviation private jet aircraft charter"
    )
    body: Dict[str, Any] = {
        "api_key": api_key,
        "query": q,
        "max_results": 6,
        "search_depth": "basic",
    }
    to = max(4.0, min(25.0, float(timeout)))
    r = requests.post(TAVILY_SEARCH_URL, json=body, timeout=to)
    r.raise_for_status()
    return r.json()


def tavily_aviation_domain_boost(host: str, *, timeout: float = 10.0) -> int:
    """
    Return 0–420 extra authority for ``host`` from a Tavily search, or 0 if disabled / error / empty.
    Cached per host for ``SEARCHAPI_TAVILY_DOMAIN_CACHE_TTL_HOURS``.
    """
    h = _normalize_host(host)
    if not searchapi_tavily_domain_verify_enabled() or _tavily_disabled() or _host_skipped_for_tavily(h):
        return 0
    api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return 0

    now = time.monotonic()
    ttl = searchapi_tavily_domain_cache_ttl_s()
    hit = _CACHE.get(h)
    if hit is not None:
        sc, ts = hit
        if now - ts <= ttl:
            return int(sc)

    parts: List[str] = []
    try:
        data = _tavily_search_domain_rest(api_key=api_key, host=h, timeout=timeout)
        results = data.get("results") if isinstance(data, dict) else None
        if isinstance(results, list):
            for r in results[:8]:
                if not isinstance(r, dict):
                    continue
                parts.append(str(r.get("title") or ""))
                parts.append(str(r.get("content") or r.get("snippet") or ""))
                parts.append(str(r.get("url") or ""))
    except Exception as e:
        logger.debug("Tavily domain verify failed for %s: %s", h, e)
        _CACHE[h] = (0, now)
        return 0

    blob = " ".join(parts)
    # If Tavily did not return our host at all, stay conservative.
    if h not in blob.lower():
        _CACHE[h] = (0, now)
        return 0

    sc = _score_snippets_for_aviation(blob)
    _CACHE[h] = (int(sc), now)
    if sc:
        logger.info("Tavily domain verify: host=%s boost=%s", h, sc)
    return int(sc)


def prefetch_tavily_domain_boosts(
    hosts_in_order: List[str],
    *,
    max_lookups: int,
    timeout_per_host: float = 10.0,
) -> Dict[str, int]:
    """
    Run Tavily at most ``max_lookups`` times for the first distinct registrable hosts in order.
    Returns map normalized_host -> boost.
    """
    if not searchapi_tavily_domain_verify_enabled() or max_lookups <= 0:
        return {}
    out: Dict[str, int] = {}
    seen: set[str] = set()
    n = 0
    for raw in hosts_in_order:
        h = _normalize_host(raw)
        if not h or h in seen or _host_skipped_for_tavily(h):
            continue
        seen.add(h)
        out[h] = tavily_aviation_domain_boost(h, timeout=timeout_per_host)
        n += 1
        if n >= max_lookups:
            break
    return out
