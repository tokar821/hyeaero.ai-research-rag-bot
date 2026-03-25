"""
Aircraft-matched images for Ask Consultant: Tavily image search + optional listing-page og:image.

No user uploads, no AI image generation — only URLs from search / scraped listing pages.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from rag.consultant_market_lookup import _blob_matches_phly_aircraft_identity
from services.scrape_listing_image_lookup import (
    images_for_listing_row,
    listing_image_lookup_key,
)

logger = logging.getLogger(__name__)

_MAX_SCRAPE_IMAGES_PER_LISTING = 8
_MAX_LISTINGS_FOR_SCRAPE_GALLERY = 4
_MAX_SCRAPE_IMAGES_TOTAL = 12

# Same marketplace hosts as Tavily filter — only fetch HTML from these (SSRF safety).
_LISTING_IMAGE_HOST_MARKERS = (
    "aircraftexchange.com",
    "aircraftexchange",
    "controller.com",
    "trade-a-plane",
    "avbuyer.com",
    "avbuyer",
    "jetnet.com",
    "globalair.com",
    "aso.com",
    "avpay",
    "planesales",
    "planequest",
    "aircraft24.com",
    "aircraft24",
    "hangar67",
    "apn.com",
    "tap.",
    "planefax",
)

_OG_IMAGE_RE = re.compile(
    r'<meta\s[^>]*(?:property|name)\s*=\s*["\']og:image["\'][^>]*content\s*=\s*["\']([^"\']+)["\']',
    re.IGNORECASE | re.DOTALL,
)
_OG_IMAGE_RE_ALT = re.compile(
    r'<meta\s[^>]*content\s*=\s*["\']([^"\']+)["\'][^>]*(?:property|name)\s*=\s*["\']og:image["\']',
    re.IGNORECASE | re.DOTALL,
)

_USER_AGENT = "HyeAero-ConsultantImageBot/1.0 (+https://hyeaero.ai)"


def _safe_https_image_url(url: str) -> bool:
    u = (url or "").strip()
    if not u.startswith("https://"):
        return False
    if len(u) > 2048:
        return False
    low = u.lower()
    if any(x in low for x in ("javascript:", "data:", "<script")):
        return False
    return True


def _listing_url_allowed_for_fetch(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    if not host:
        return False
    return any(m in host for m in _LISTING_IMAGE_HOST_MARKERS)


def fetch_og_image_url(listing_page_url: str, *, timeout: float = 6.0) -> Optional[str]:
    """GET listing HTML (first chunk) and return og:image if present and https."""
    u = (listing_page_url or "").strip()
    if not u.startswith("http") or not _listing_url_allowed_for_fetch(u):
        return None
    try:
        r = requests.get(
            u,
            timeout=(min(3.0, timeout), timeout),
            headers={"User-Agent": _USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            stream=True,
        )
        r.raise_for_status()
        ct = (r.headers.get("Content-Type") or "").lower()
        if "html" not in ct and "text" not in ct and ct:
            return None
        raw = b""
        for chunk in r.iter_content(chunk_size=65536):
            raw += chunk
            if len(raw) >= 400_000:
                break
        text = raw.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.debug("og:image fetch failed for %s: %s", u[:80], e)
        return None
    m = _OG_IMAGE_RE.search(text) or _OG_IMAGE_RE_ALT.search(text)
    if not m:
        return None
    img = (m.group(1) or "").strip()
    if img.startswith("//"):
        img = "https:" + img
    if not _safe_https_image_url(img):
        return None
    return img


def _normalize_tavily_image(item: Any) -> Optional[Dict[str, Optional[str]]]:
    if isinstance(item, str):
        u = item.strip()
        return {"url": u, "description": None} if u else None
    if isinstance(item, dict):
        u = (item.get("url") or "").strip()
        if not u:
            return None
        d = item.get("description")
        desc = str(d).strip() if d else None
        return {"url": u, "description": desc}
    return None


def filter_tavily_images_for_phly(
    images: List[Any],
    phly_rows: List[Dict[str, Any]],
    *,
    max_out: int = 8,
    trust_tail_biased_search: bool = False,
) -> List[Dict[str, Optional[str]]]:
    """Keep Tavily image hits whose URL or description references PhlyData serial/tail.

    When ``trust_tail_biased_search`` is True (photo-focused Tavily query already included the
    tail/serial), accept HTTPS image URLs even if the path omits the registration — typical for CDNs.
    """
    out: List[Dict[str, Optional[str]]] = []
    if not images:
        return out
    if not phly_rows:
        for item in images:
            norm = _normalize_tavily_image(item)
            if norm and _safe_https_image_url(norm["url"] or ""):
                out.append(
                    {"url": norm["url"], "description": norm.get("description"), "source": "tavily"}
                )
            if len(out) >= max_out:
                break
        return out
    for item in images:
        norm = _normalize_tavily_image(item)
        if not norm:
            continue
        url = norm.get("url") or ""
        desc = norm.get("description") or ""
        if not _safe_https_image_url(url):
            continue
        blob = f"{url} {desc}"
        if trust_tail_biased_search:
            desc_out = desc or "Tavily image (tail-specific search; verify visually)"
            out.append({"url": url, "description": desc_out, "source": "tavily"})
        else:
            if not _blob_matches_phly_aircraft_identity(blob, phly_rows):
                continue
            out.append({"url": url, "description": desc, "source": "tavily"})
        if len(out) >= max_out:
            break
    return out


def build_consultant_aircraft_images(
    tavily_payload: Dict[str, Any],
    phly_rows: List[Dict[str, Any]],
    listing_urls: Optional[List[str]] = None,
    listing_rows: Optional[List[Dict[str, Any]]] = None,
    *,
    trust_tail_biased_tavily_images: bool = False,
    max_listing_og_fetches: int = 3,
    og_timeout: float = 6.0,
) -> List[Dict[str, Any]]:
    """
    Deduped list for API/UI: {url, source: tavily|scrape_gallery|listing_og, page_url?, description?, lookup_key?}.

    ``listing_rows`` should be Postgres-aligned dicts (source_platform, source_listing_id, listing_url)
    so images can be resolved from pre-extracted JSON (see scrape_listing_image_lookup).

    ``trust_tail_biased_tavily_images``: set True when Tavily was run with a **photo-focused query**
    that already included the quoted tail/serial (see :func:`build_aircraft_photo_focus_tavily_query`);
    CDN URLs often omit the registration in the path, so identity matching on URL alone would drop them.
    """
    seen: set[str] = set()
    final: List[Dict[str, Any]] = []

    raw_imgs = tavily_payload.get("images") if isinstance(tavily_payload, dict) else None
    if not isinstance(raw_imgs, list):
        raw_imgs = []
    tavily_cap = 14 if trust_tail_biased_tavily_images else 10
    for row in filter_tavily_images_for_phly(
        raw_imgs,
        phly_rows,
        max_out=tavily_cap,
        trust_tail_biased_search=trust_tail_biased_tavily_images,
    ):
        u = (row.get("url") or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        final.append(
            {
                "url": u,
                "source": row.get("source") or "tavily",
                "description": row.get("description"),
                "page_url": None,
                "lookup_key": None,
            }
        )

    # Pre-scraped gallery URLs (Controller / AircraftExchange) keyed by marketplace listing id.
    n_scrape = 0
    if listing_rows:
        for lr in listing_rows[:_MAX_LISTINGS_FOR_SCRAPE_GALLERY]:
            if not isinstance(lr, dict):
                continue
            if n_scrape >= _MAX_SCRAPE_IMAGES_TOTAL:
                break
            page = (lr.get("listing_url") or "").strip()
            lk = listing_image_lookup_key(lr)
            gal = images_for_listing_row(lr)
            per = 0
            for u in gal:
                if n_scrape >= _MAX_SCRAPE_IMAGES_TOTAL or per >= _MAX_SCRAPE_IMAGES_PER_LISTING:
                    break
                u = (u or "").strip()
                if not u or u in seen:
                    continue
                seen.add(u)
                final.append(
                    {
                        "url": u,
                        "source": "scrape_gallery",
                        "description": "Gallery image from saved marketplace scrape (matched listing row)",
                        "page_url": page or None,
                        "lookup_key": lk or None,
                    }
                )
                n_scrape += 1
                per += 1

    # URLs come from aircraft_listings rows already joined to this tail/serial — safe to scrape og:image.
    if listing_urls:
        n = 0
        for page in listing_urls:
            if n >= max_listing_og_fetches:
                break
            page = (page or "").strip()
            if not page.startswith("http"):
                continue
            img = fetch_og_image_url(page, timeout=og_timeout)
            n += 1
            if not img or img in seen:
                continue
            seen.add(img)
            final.append(
                {
                    "url": img,
                    "source": "listing_og",
                    "description": "Listing page preview image (og:image)",
                    "page_url": page,
                    "lookup_key": None,
                }
            )

    return final[:16]
