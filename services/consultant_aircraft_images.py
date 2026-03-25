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

# Reject obvious non-aviation image hits (Tavily image search is noisy next to fashion / lifestyle).
_OFF_TOPIC_IMAGE_BLOB = re.compile(
    r"catwalk|haute[-\s]couture|supermodel|menswear|womenswear|streetwear|"
    r"\bfashion\b|luxury\s+fashion|\b(?:womens|mens)\s*(?:wear|fashion)\b|"
    r"fashion\s*(?:week|show|house)|(?:paris|milan|london|ny|new\s*york)\s+fashion|"
    r"runway\s*(?:show|collection|look)|fashion\s*runway|\b(?:nyfw|mfw|pfw|lfw)\b|"
    r"\b(?:celebrity|red\s*carpet|street\s*style|makeup|hairstylist|beauty\s+look)\b|"
    r"vogue\.|elle\.com|wmagazine|harpersbazaar|harpers\s*bazaar|highsnobiety|"
    r"lookbook|outfit\s*of\s*the\s*day|ootd|\bmodel\s+(?:walks|wearing|in\s+heel)|"
    r"\b(?:she|he|they)\s+wear(?:s|ing)?\b|"
    r"handbag|handbags|clutch\s+bag|purse\b|stilettos?|(?:high[-\s])?heels\b|evening\s*gown|"
    r"photo-?shoot|photoshoot|editorial\s+shoot|spring\s+\d{4}\s+collection|"
    r"\bfw\d{2}\b|\bss\d{2}\b|ready[-\s]to[-\s]wear|pret[-\s]a[-\s]porter|"
    r"backstage\s+at|front\s*row|"
    r"zara\.|hm\.com|h&m|uniqlo|shein|farfetch|net-?a-?porter|"
    r"getty\s*images.*\b(fashion|runway|model)\b|shutterstock.*\bfashion\b",
    re.I,
)

# Hosts that almost always mean lifestyle / retail editorials (not aircraft photography).
_FASHION_EDITORIAL_HOST_FRAGMENTS = (
    "vogue.",
    "elle.com",
    "elle.fr",
    "harpersbazaar",
    "wmagazine",
    "wwd.com",
    "instyle.",
    "hypebeast.",
    "highsnobiety",
    "farfetch.",
    "net-a-porter",
    "matchesfashion",
    "ssense.",
    "therunway.com",  # retail, not ramp
    "whowhatwear.",
)

# URL path snippets common on magazine / retail CDNs serving fashion stories.
_FASHION_EDITORIAL_PATH_RE = re.compile(
    r"/(fashion|beauty|style|womens(?:wear)?|mens(?:wear)?|shopping|retail|"
    r"lifestyle|celebrity|street-?style|runway|looks?|trending-style)(/|$|\?)|"
    r"/(tag|tags|category|categories)/[^/\s]*(?:fashion|beauty|runway|style)|"
    r"/wiki/(fashion|haute_couture|supermodel)\b",
    re.I,
)

# CDN image URLs use opaque paths — tighten text when identity is not in URL/description.
_LIFESTYLE_EDITORIAL_DESCRIPTION_RE = re.compile(
    r"\b(?:handbag|purse|stiletto|heels?|gown|couture|lookbook|ootd|"
    r"photo-?shoot|editorial|nyfw|mfw|pfw|lfw|womenswear|menswear)\b",
    re.I,
)

# Strong positive signals (narrow "citation" / "cabin" vs academic & lifestyle noise).
_AIRCRAFT_IMAGE_BLOB_HINT = re.compile(
    r"\b(?:"
    r"aircraft|airplane|aeroplane|aviation|airliner|airliners|bizjet|biz\s*jet|business\s*jet|"
    r"private\s*jet|corporate\s*jet|helicopter|turboprop|propjet|widebody|narrowbody|"
    r"cockpit|flight\s*deck|(?:aircraft|jet|plane|bizjet)\s+cabin|\bcabin\s*(?:interior|seating|layout)|"
    r"interior\s*cabin|airframe|fuselage|winglets?|tail\s*fin|"
    r"jetphotos|planespotters|flightaware|flightradar|aircraftforsale|"
    r"cessna(?:\s+citation)?|gulfstream|bombardier|embraer|beechcraft|hawker|pilatus|phenom|"
    r"citation\s+(?:cj|excel|xls|latitude|longitude|mustang|sovereign|hemisphere|ascend|x)\w*|"
    r"\bcj\d+[a-z]?\b|"
    r"challenger|falcon|learjet|global\s*(?:5000|6000|6500|7500|8000|express)?|"
    r"boeing|airbus|b737|b787|a220|a320|crj|erj|dash\s*8|king\s*air"
    r")\b",
    re.I,
)

# Hosts that overwhelmingly serve aviation photography / fleet imagery (still subject to off-topic reject).
_KNOWN_AVIATION_IMAGE_HOST_FRAGMENTS = (
    "jetphotos.com",
    "jetphotos.net",
    "airliners.net",
    "planespotters.net",
    "planepictures.net",
    "airplane-pictures.net",
    "flightaware.com",
    "flightradar24",
    "airport-data.com",
    "aviation-safety.net",
    "flightglobal.com",
    "ainonline.com",
    "aviationweek.com",
    "simpleflying.com",
    "verticalmag.com",
)

# Listing / OEM PR CDNs — still not fashion editorials; used when tail is not in opaque image path.
_TRUSTED_LISTING_OR_MARKET_IMAGE_HOST_FRAGMENTS = (
    "controller.com",
    "aircraftexchange.com",
    "globalair.com",
    "avbuyer.com",
    "trade-a-plane",
    "jetnet.com",
    "aso.com",
    "avpay",
    "planesales",
    "planequest",
    "aircraft24.com",
    "hangar67",
    "businessaircraft.bombardier.com",
    "gulfstream.com",
    "txtav.com",
    "embraer.com",
)

_TRUSTED_AIRCRAFT_IMAGE_HOST_FRAGMENTS = (
    _KNOWN_AVIATION_IMAGE_HOST_FRAGMENTS + _TRUSTED_LISTING_OR_MARKET_IMAGE_HOST_FRAGMENTS
)

# People / runway / styling cues — reject unless tail/serial clearly appears in URL+description (id match).
_HUMAN_FASHION_TABLOID_RISK = re.compile(
    r"\b(?:wearing|wears|wore|styled\s+by|dress(?:ed)?\s+by|outfit|ootd)\b|"
    r"\bwalk(?:s|ed|ing)?\s+(?:the\s+)?runway\b|"
    r"\b(?:supermodel|top\s+model)\b|"
    r"\b(?:celebrity|celebrities)\s+(?:style|fashion|looks?|in)\b|"
    r"\b(?:fashion|couture)\s+(?:week|show|house|brand|night)\b|"
    r"\b(?:mens|womens)\s*wear\s+(?:show|collection|brand)\b|"
    r"\b(?:vogue|elle|bazaar|wwd)\s+(?:cover|party|gala|issue)\b|"
    r"\bfw\s?\d{2}\b|\bss\s?\d{2}\w?\b|"
    r"\b(?:spring|summer|fall|autumn|winter)\s+\d{4}\s+(?:collection|show)\b|"
    r"\bfront\s*row\b|\bbackstage\s+at\b|"
    r"\brunway\s+(?:collection|look)\b|"
    r"\b(?:designer|luxury\s+brand)\s+(?:\w+\s+){0,3}(?:show|collection)\b|"
    r"\bcelebrities?\s+at\s+\w+\s+fashion\b",
    re.I,
)

# US civil registration: N + alphanumeric, must include at least one digit — avoids "new", "next", "nick".
_US_N_REG_IN_BLOB = re.compile(r"\bN(?=[A-Z0-9]*\d)[A-Z0-9]{1,5}\b", re.I)

_SUSPICIOUS_IMAGE_PAGE_URL = re.compile(
    r"\.(?:html?|php|asp)(?:\?|$)|/(?:article|articles|news|story|stories|blog|magazine|"
    r"runway|fashion|beauty|style|celebrity|entertainment|living|shopping)/",
    re.I,
)

# If present, do not treat path as "fashion editorial" (aviation / planespotting pages).
_AV_PATH_OVERRIDE_RE = re.compile(
    r"/(aviation|aircraft|airplanes?|bizjet|jets?|flightaware|flightradar|planespotter|"
    r"airliners|military-?aviation|general-?aviation)(/|$|\?)",
    re.I,
)

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


def _sanitize_tavily_image_description(desc: Optional[str]) -> Optional[str]:
    """Drop Tavily placeholder / disclaimer text so broken images do not show huge alt strings."""
    if desc is None:
        return None
    d = str(desc).strip()
    if not d:
        return None
    low = d.lower()
    if low.startswith("tavily") or "tavily image" in low:
        return None
    if any(
        x in low
        for x in (
            "verify visually",
            "verify",
            "tail-specific",
            "tail specific",
            "third-party",
            "third party",
            "image url",
            "tavily",
            "may not be this",
            "may not be",
            "before showing",
            "unverified image",
            "unverified",
            "not verified",
        )
    ):
        return None
    # Narrative disclaimers that are not real alt text
    if "search;" in low and "verify" in low:
        return None
    if len(d) > 160:
        return d[:157].rstrip() + "…"
    return d


def _consultant_image_page_likely_fashion_editorial(url: str) -> bool:
    """True when the image URL's host/path usually indicates magazine or retail, not aircraft."""
    try:
        p = urlparse(url)
    except Exception:
        return False
    host = (p.netloc or "").lower()
    path_q = f"{p.path or ''}?{p.query or ''}".lower()
    if _AV_PATH_OVERRIDE_RE.search(path_q):
        return False
    if any(h in host for h in _FASHION_EDITORIAL_HOST_FRAGMENTS):
        return True
    if _FASHION_EDITORIAL_PATH_RE.search(path_q):
        return True
    return False


def _image_url_suspicious_for_gallery(url: str) -> bool:
    """Article / CMS pages are often not hotlinkable image assets."""
    u = (url or "").strip().lower()
    if not u:
        return True
    return bool(_SUSPICIOUS_IMAGE_PAGE_URL.search(u))


def _blob_matches_phly_manufacturer_or_model(blob: str, phly_rows: List[Dict[str, Any]]) -> bool:
    """Typed make/model from PhlyData appears in URL or description — tightens trust-tail CDN results."""
    if not blob or not phly_rows:
        return False
    low = blob.lower()
    compact = re.sub(r"[^a-z0-9]", "", blob).lower()
    for r in phly_rows:
        for key in ("manufacturer", "model"):
            val = (r.get(key) or "").strip()
            if len(val) < 4:
                continue
            if val.lower() in low:
                return True
            vc = re.sub(r"[^a-z0-9]", "", val).lower()
            if len(vc) >= 4 and vc in compact:
                return True
    return False


def _tavily_image_blob_is_off_topic(blob: str) -> bool:
    b = (blob or "").strip()
    if not b:
        return False
    return bool(_OFF_TOPIC_IMAGE_BLOB.search(b))


def _tavily_image_blob_human_fashion_tabloid_risk(blob: str) -> bool:
    b = (blob or "").strip()
    if not b:
        return False
    return bool(_HUMAN_FASHION_TABLOID_RISK.search(b))


def _url_host_matches_trusted_aircraft_media(url: str) -> bool:
    low = (url or "").lower()
    return any(h in low for h in _TRUSTED_AIRCRAFT_IMAGE_HOST_FRAGMENTS)


def _tavily_image_blob_has_aircraft_signal(blob: str) -> bool:
    """True when URL/description/host plausibly relates to aircraft (for relaxed tail-biased filter)."""
    b = (blob or "").strip()
    if not b:
        return False
    low = b.lower()
    if _AIRCRAFT_IMAGE_BLOB_HINT.search(b):
        return True
    if _US_N_REG_IN_BLOB.search(b):
        return True
    if any(h in low for h in _KNOWN_AVIATION_IMAGE_HOST_FRAGMENTS):
        return True
    if any(h in low for h in _TRUSTED_LISTING_OR_MARKET_IMAGE_HOST_FRAGMENTS):
        return True
    return False


def _tavily_aircraft_context_strong_enough_for_cdn_match(blob: str) -> bool:
    """Stricter than :func:`_tavily_image_blob_has_aircraft_signal` — avoids 'make/model word' false positives."""
    if not blob.strip():
        return False
    low = blob.lower()
    if _US_N_REG_IN_BLOB.search(blob):
        return True
    if any(h in low for h in _KNOWN_AVIATION_IMAGE_HOST_FRAGMENTS):
        return True
    if any(h in low for h in _TRUSTED_LISTING_OR_MARKET_IMAGE_HOST_FRAGMENTS):
        return True
    return bool(
        re.search(
            r"\b(?:aircraft|airplane|aeroplane|aviation|airliner|bizjet|business\s*jet|private\s*jet|"
            r"corporate\s*jet|cockpit|flight\s*deck|jetphotos|planespotters|airframe|fuselage|winglets?)\b",
            blob,
            re.I,
        )
    )


def _blob_suggests_lifestyle_without_aviation_anchor(blob: str, phly_rows: List[Dict[str, Any]]) -> bool:
    """Opaque CDN URLs: drop hits whose description is clearly lifestyle if identity is not tied to the aircraft."""
    if not blob or not _LIFESTYLE_EDITORIAL_DESCRIPTION_RE.search(blob):
        return False
    if phly_rows and _blob_matches_phly_aircraft_identity(blob, phly_rows):
        return False
    if _tavily_image_blob_has_aircraft_signal(blob):
        return False
    return True


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
            if not norm:
                continue
            url = (norm.get("url") or "").strip()
            desc = norm.get("description") or ""
            if not _safe_https_image_url(url):
                continue
            blob = f"{url} {desc}"
            if _consultant_image_page_likely_fashion_editorial(url):
                continue
            if _blob_suggests_lifestyle_without_aviation_anchor(blob, []):
                continue
            if _tavily_image_blob_is_off_topic(blob):
                continue
            if _tavily_image_blob_human_fashion_tabloid_risk(blob):
                continue
            if _image_url_suspicious_for_gallery(url):
                continue
            if not _tavily_image_blob_has_aircraft_signal(blob):
                continue
            dclean = _sanitize_tavily_image_description(desc)
            out.append({"url": url, "description": dclean, "source": "tavily"})
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
        low = blob.lower()
        if _tavily_image_blob_is_off_topic(blob):
            continue
        id_match = _blob_matches_phly_aircraft_identity(blob, phly_rows)
        if _tavily_image_blob_human_fashion_tabloid_risk(blob) and not id_match:
            continue
        if trust_tail_biased_search:
            if _image_url_suspicious_for_gallery(url) and not id_match:
                continue
            trusted_host = _url_host_matches_trusted_aircraft_media(url)
            if id_match:
                desc_out = _sanitize_tavily_image_description(desc) or "Aircraft (web search)"
                out.append({"url": url, "description": desc_out, "source": "tavily"})
            elif trusted_host and _tavily_image_blob_has_aircraft_signal(blob):
                # Opaque CDN paths on planespotter / listing hosts only — no tail-in-URL guesses from lifestyle CDNs.
                desc_out = _sanitize_tavily_image_description(desc) or "Aircraft (web search)"
                out.append({"url": url, "description": desc_out, "source": "tavily"})
            elif _tavily_image_blob_has_aircraft_signal(blob) and not _consultant_image_page_likely_fashion_editorial(
                url
            ):
                # Photo-focused Tavily query already quoted the tail — allow any host with strong aviation cues
                # (still blocked by off_topic + fashion_tabloid_risk above).
                desc_out = _sanitize_tavily_image_description(desc) or "Aircraft (web search)"
                out.append({"url": url, "description": desc_out, "source": "tavily"})
        else:
            if not id_match:
                continue
            dclean = _sanitize_tavily_image_description(desc)
            out.append({"url": url, "description": dclean, "source": "tavily"})
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
