"""
Tail-specific marketing / broker listing pages (Virtual Hangar, etc.).

SearchAPI often returns CDN image URLs without cabin keywords; verification and cabin
facet filters use ``page_url`` + listing host to accept interior photos tied to the tail.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from rag.aviation_tail import normalize_tail_token

# Host fragments allowed for og:image HTML fetch (SSRF-safe allowlist extension).
MARKETING_LISTING_HOST_MARKERS: tuple[str, ...] = (
    "virtualhangar.com",
    "flyexclusive.com",
    "flyexclusive.",
    "flyjet.com",
    "flyjet.",
    "phly.com",
    "phly.",
    "controller.com",
    "aircraftexchange",
    "globalair.com",
    "avbuyer.",
    "trade-a-plane",
    "aso.com",
)

# CDN / asset hosts common on marketing listing pages (image URL may not mention the tail).
MARKETING_LISTING_CDN_MARKERS: tuple[str, ...] = (
    "website-files.com",
    "webflow.com",
    "cloudfront.net",
    "imgix.net",
)

_TAIL_IN_PAGE_PATH_RE = re.compile(
    r"(?i)/(?:aircraft|listing|inventory|jet|aircraft-detail|for-sale)/[^/?#]*"
    r"(n[1-9a-z][a-z0-9]{1,5})"
)

_EXTERIOR_PATH_MARKERS = (
    "exterior",
    "exteriortail",
    "ramp",
    "walkaround",
    "air-to-air",
    "takeoff",
    "landing",
    "spotting",
)

_CABIN_PATH_MARKERS = (
    "cabin",
    "interior",
    "salon",
    "galley",
    "seating",
    "layout",
    "divan",
    "lavatory",
    "website-files.com",
)

# Logos / widgets scraped from broker HTML (not aircraft cabin stills).
_JUNK_IMAGE_PATH_MARKERS = (
    "logo",
    "logos",
    "jetstimate",
    "uberjets",
    "uber-jets",
    "virtualhangar_logos",
    "white-",
    "banner",
    "favicon",
    "icon-",
    "sprite",
    "placeholder",
)

# Image hosts that are not the listing site's Webflow CDN (site chrome).
_DISALLOWED_IMAGE_HOST_FRAGMENTS = (
    "virtualhangar.io",
    "jetstimate.com",
)

_IMG_URL_IN_HTML_RE = re.compile(
    r"""https?://[^\s"'<>]+\.(?:jpg|jpeg|png|webp)(?:\?[^\s"'<>]*)?""",
    re.I,
)


def _host_allowed(url: str) -> bool:
    try:
        host = (urlparse(url).netloc or "").lower()
    except Exception:
        return False
    if not host:
        return False
    return any(m in host for m in MARKETING_LISTING_HOST_MARKERS)


def canonical_virtualhangar_tail_url(tail: str) -> str:
    t = normalize_tail_token(tail or "")
    if not t:
        return ""
    return f"https://virtualhangar.com/aircraft/{t.lower()}"


def discover_tail_marketing_listing_urls(
    tail: str,
    phly_rows: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """Ordered unique listing page URLs likely to carry cabin/interior photos for this tail."""
    t = normalize_tail_token(tail or "")
    if not t:
        return []
    seen: set[str] = set()
    out: List[str] = []

    def _add(u: str) -> None:
        u = (u or "").strip()
        if not u.startswith("http") or u.lower() in seen:
            return
        if not _host_allowed(u):
            return
        seen.add(u.lower())
        out.append(u)

    vh = canonical_virtualhangar_tail_url(t)
    if vh:
        _add(vh)

    for row in phly_rows or []:
        if not isinstance(row, dict):
            continue
        for key in ("listing_url", "url", "source_url"):
            _add(str(row.get(key) or ""))

    return out


def append_tail_marketing_cabin_queries(queries: List[str], tail: str) -> List[str]:
    """Prepend high-recall queries for broker listing sites (within word caps elsewhere)."""
    t = normalize_tail_token(tail or "")
    if not t:
        return list(queries or [])
    extras = [
        f"{t} site:virtualhangar.com",
        f"{t} virtual hangar cabin",
        f"{t} cabin interior",
        f"{t} interior",
    ]
    seen = {q.strip().lower() for q in queries if (q or "").strip()}
    out: List[str] = []
    for q in extras:
        k = q.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(q.strip())
    out.extend(queries or [])
    return out


def _tail_on_listing_page(page_url: str, tail: str) -> bool:
    t = normalize_tail_token(tail or "")
    if not t or not page_url:
        return False
    low = page_url.lower()
    if t.lower() in low:
        return True
    if t.startswith("N") and len(t) >= 4 and t[1:].lower() in low:
        return True
    m = _TAIL_IN_PAGE_PATH_RE.search(page_url)
    if m and normalize_tail_token(m.group(1)) == t:
        return True
    return False


def is_tail_marketing_listing_page(page_url: str, tail: str) -> bool:
    """True when ``page_url`` is a known listing host and references the tail."""
    page = (page_url or "").strip()
    if not page.startswith("http") or not _host_allowed(page):
        return False
    return _tail_on_listing_page(page, tail)


def _image_url_looks_exterior(url: str) -> bool:
    low = (url or "").lower()
    return any(m in low for m in _EXTERIOR_PATH_MARKERS)


def _image_url_looks_junk_marketing_asset(url: str) -> bool:
    low = (url or "").lower()
    if any(h in low for h in _DISALLOWED_IMAGE_HOST_FRAGMENTS):
        return True
    if any(j in low for j in _JUNK_IMAGE_PATH_MARKERS):
        return True
    if "wp-content/uploads" in low and "website-files.com" not in low:
        if "virtualhangar.com" in low or "virtualhangar.io" in low:
            if not re.search(
                r"(?i)\b(cabin|interior|salon|galley|aircraft|xls|citation|excel|560|n807)",
                low,
            ):
                return True
    return False


def _image_url_looks_cabin(url: str) -> bool:
    low = (url or "").lower()
    if _image_url_looks_exterior(low) or _image_url_looks_junk_marketing_asset(low):
        return False
    if "website-files.com" in low:
        return True
    return any(m in low for m in _CABIN_PATH_MARKERS if m != "website-files.com")


def fetch_marketing_listing_page_images(
    listing_page_url: str,
    *,
    tail: str = "",
    want_cabin: bool = True,
    max_images: int = 6,
    timeout: float = 10.0,
) -> List[str]:
    """
    Extract HTTPS image URLs embedded in broker listing HTML.

    ``og:image`` on Virtual Hangar is often an exterior S3 shot; cabin photos live in-page
    (typically ``website-files.com`` CDN paths).
    """
    from services.consultant_aircraft_images import fetch_listing_page_html

    page = (listing_page_url or "").strip()
    if not page.startswith("http") or not _host_allowed(page):
        return []

    text = fetch_listing_page_html(page, timeout=timeout)
    if not text:
        return []

    raw = list(dict.fromkeys(_IMG_URL_IN_HTML_RE.findall(text)))
    page_low = page.lower()

    if want_cabin and "virtualhangar.com" in page_low:
        vh: List[str] = []
        for u in raw:
            u = (u or "").strip()
            if not u.startswith("https://") or _image_url_looks_junk_marketing_asset(u):
                continue
            if "website-files.com" in u.lower() and not _image_url_looks_exterior(u):
                vh.append(u)
        return vh[: max(1, min(max_images, 4))]

    cabin: List[str] = []
    for u in raw:
        u = (u or "").strip()
        if not u.startswith("https://"):
            continue
        if _image_url_looks_exterior(u) or _image_url_looks_junk_marketing_asset(u):
            continue
        if want_cabin:
            if _image_url_looks_cabin(u):
                cabin.append(u)
        else:
            cabin.append(u)

    return cabin[: max(1, max_images)]


def row_is_tail_listing_cabin_candidate(row: Dict[str, Any], tail: str) -> bool:
    """
    Listing-page interior photo: tail on marketing page, image often on a generic CDN path.
    """
    t = normalize_tail_token(tail or "")
    if not t:
        return False
    page = str(row.get("page_url") or row.get("_source_page") or row.get("link") or "")
    if not is_tail_marketing_listing_page(page, t):
        return False
    url = str(row.get("url") or row.get("image") or "").lower()
    if _image_url_looks_exterior(url):
        return False
    if "website-files.com" in url:
        return True
    if _image_url_looks_junk_marketing_asset(url):
        return False
    if any(cdn in url for cdn in MARKETING_LISTING_CDN_MARKERS) and _image_url_looks_cabin(url):
        return True
    blob = " ".join(
        str(row.get(k) or "")
        for k in ("url", "title", "description", "source", "page_url", "_source_page")
    ).lower()
    if re.search(r"\b(interior|cabin|salon|seating|galley|layout)\b", blob, re.I) and _image_url_looks_cabin(url):
        return True
    if "virtualhangar" in page.lower() and _image_url_looks_cabin(url):
        return True
    return _tail_on_listing_page(page, t) and _image_url_looks_cabin(url)


def enrich_gallery_from_tail_marketing_listings(
    *,
    tail: str,
    phly_rows: Optional[List[Dict[str, Any]]],
    max_out: int = 5,
    facet: str = "cabin",
) -> List[Dict[str, Any]]:
    """
    Fetch og:image (and optional scrape gallery) from broker listing URLs for this tail.
    """
    from services.consultant_aircraft_images import fetch_og_image_url

    t = normalize_tail_token(tail or "")
    if not t:
        return []

    want_cabin = (facet or "").strip().lower() in ("cabin", "interior", "")
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for page in discover_tail_marketing_listing_urls(t, phly_rows)[:6]:
        imgs = fetch_marketing_listing_page_images(
            page,
            tail=t,
            want_cabin=want_cabin,
            max_images=max_out,
        )
        if not imgs and not want_cabin:
            og = fetch_og_image_url(page)
            if og:
                imgs = [og]
        if not imgs and want_cabin:
            og = fetch_og_image_url(page)
            if og and not _image_url_looks_exterior(og):
                imgs = [og]

        for img in imgs:
            if not img or img in seen:
                continue
            seen.add(img)
            desc = f"{t} cabin interior"
            if "virtualhangar" in page.lower():
                desc = f"{t} — Virtual Hangar listing cabin interior"
            row: Dict[str, Any] = {
                "url": img,
                "source": "listing_scrape",
                "description": desc,
                "title": desc,
                "page_url": page,
                "_source_page": page,
                "tail_match_confidence": "confirmed",
                "image_provenance": "tail_marketing_listing",
            }
            if want_cabin:
                try:
                    from services.broker_execution.gallery_visual_intent import row_matches_visual_facet

                    if not row_matches_visual_facet(row, "cabin"):
                        continue
                except Exception:
                    pass
            out.append(row)
            if len(out) >= max_out:
                break
        if len(out) >= max_out:
            break

    if not out:
        return []

    try:
        from services.aircraft_image_verification.pipeline import verify_gallery_images

        verified, _meta = verify_gallery_images(
            out,
            tail=t,
            model=None,
            section="cabin" if want_cabin else "interior",
            max_out=max_out,
        )
        return verified or out[:max_out]
    except Exception:
        return out[:max_out]


__all__ = [
    "MARKETING_LISTING_HOST_MARKERS",
    "MARKETING_LISTING_CDN_MARKERS",
    "append_tail_marketing_cabin_queries",
    "canonical_virtualhangar_tail_url",
    "discover_tail_marketing_listing_urls",
    "enrich_gallery_from_tail_marketing_listings",
    "fetch_marketing_listing_page_images",
    "is_tail_marketing_listing_page",
    "row_is_tail_listing_cabin_candidate",
]
