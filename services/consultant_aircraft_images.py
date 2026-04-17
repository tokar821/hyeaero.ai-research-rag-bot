"""
Aircraft-matched images for Ask Consultant: SearchAPI (Bing Images) when ``SEARCHAPI_API_KEY``
is set, otherwise Tavily image blobs + optional listing-page og:image.

No user uploads, no AI image generation — only URLs from search / scraped listing pages.
"""

from __future__ import annotations

import logging
import os
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

_TAIL_TOKEN_ALNUM = re.compile(r"[^A-Z0-9]+")


def _norm_alnum_tokens(s: str) -> str:
    """Uppercase string with non-alnum replaced by spaces (stable token boundaries)."""
    return _TAIL_TOKEN_ALNUM.sub(" ", (s or "").upper()).strip()


def _tail_token_present(text: str, tail: str) -> bool:
    """
    True if ``tail`` appears as its own token in text, rejecting near-matches.

    Example: tail=N807JS → accept "... N807JS ..." but reject N807JT / N807JSX.
    """
    t = _norm_alnum_tokens(tail)
    if not t or len(t) < 3:
        return False
    blob = _norm_alnum_tokens(text)
    if not blob:
        return False
    # Token-boundary match in a normalized alnum space-separated string.
    return re.search(rf"(?<![A-Z0-9]){re.escape(t)}(?![A-Z0-9])", blob) is not None


_AVIATION_HOST_PRIORITY: List[tuple[str, int]] = [
    ("jetphotos.", 100),
    ("planespotters.", 90),
    ("airliners.", 80),
    ("airport-data.", 70),
    ("aviation-safety.", 70),
]


def _host_priority_score(url: str) -> int:
    host = ""
    try:
        host = (urlparse((url or "").strip()).netloc or "").lower()
    except Exception:
        host = ""
    if not host:
        return 0
    for frag, score in _AVIATION_HOST_PRIORITY:
        if frag in host:
            return score
    return 10


def _derive_model_positive_tokens(marketing_type: str) -> List[str]:
    """
    Positive model tokens for strict model mode (title/alt/desc/url).
    Includes common shorthand (e.g. Falcon 2000 → F2000).
    """
    mt = (marketing_type or "").strip()
    if not mt:
        return []
    out: List[str] = []
    out.append(mt)
    # Also accept model-only token if marketing_type includes manufacturer.
    parts = mt.split()
    if len(parts) >= 2:
        out.append(" ".join(parts[-2:]))  # last two words often "Falcon 2000", "Phenom 300"
    low = mt.lower()
    m = re.search(r"\bfalcon\s*(\d{3,4})\b", low)
    if m:
        out.append(f"F{m.group(1)}")
    m = re.search(r"\bchallenger\s*(\d{3})\b", low)
    if m:
        out.append(f"CL{m.group(1)}")
    m = re.search(r"\bglobal\s*(\d{4})\b", low)
    if m:
        out.append(f"Global {m.group(1)}")
    m = re.search(r"\bg\s*[-.]?\s*(\d{3,4})\b", low)
    if m and "gulfstream" in low:
        out.append(f"G{m.group(1)}")
    # Dedup normalized
    seen: set[str] = set()
    uniq: List[str] = []
    for t in out:
        t2 = t.strip()
        if not t2:
            continue
        key = t2.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t2)
    return uniq


def _derive_model_negative_tokens(marketing_type: str) -> List[str]:
    """
    Conservative negative tokens to prevent common family bleed.
    Only implemented for the most confusion-prone families.
    """
    mt = (marketing_type or "").strip().lower()
    if not mt:
        return []
    neg: List[str] = []
    if "falcon" in mt:
        if "2000" in mt:
            neg += [
                "Falcon 900",
                "Falcon 50",
                "Falcon 7X",
                "Falcon 8X",
                "Falcon 6X",
                "Falcon 10X",
            ]
        if "900" in mt:
            neg += ["Falcon 2000", "Falcon 7X", "Falcon 8X", "Falcon 6X", "Falcon 10X"]
    if "phenom" in mt:
        if "300" in mt:
            neg += ["Phenom 100"]
        if "100" in mt:
            neg += ["Phenom 300"]
    if "gulfstream" in mt and "650" in mt:
        neg += ["G550", "G450", "G500", "G600"]
    if "global" in mt and "7500" in mt:
        neg += ["Global 5000", "Global 6000", "Global 6500", "Global 8000"]
    if "challenger" in mt and "350" in mt:
        neg += ["Challenger 300", "Challenger 650", "Challenger 600", "Challenger 605", "Challenger 604"]
    # Unique and non-empty
    seen: set[str] = set()
    out: List[str] = []
    for t in neg:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def _model_tokens_match_strict(blob: str, marketing_type: str) -> bool:
    """Strict model match: require positive token, reject any negative token."""
    b = (blob or "")
    b_l = b.lower()
    pos = _derive_model_positive_tokens(marketing_type)
    neg = _derive_model_negative_tokens(marketing_type)
    if neg and any(n.lower() in b_l for n in neg):
        return False
    # Require at least one positive token as a substring (title/alt/url often includes it).
    return any(p.lower() in b_l for p in pos if len(p) >= 4)


def _iter_tavily_result_linked_images_for_model(
    tavily_payload: Dict[str, Any],
    *,
    required_marketing_type: str,
    cap_results: int = 20,
    cap_images_per_result: int = 8,
) -> List[Dict[str, Optional[str]]]:
    """
    Pull **source-linked** images from Tavily results whose page text matches the required model
    (strict model tokens, Option B-style page verification).
    """
    out: List[Dict[str, Optional[str]]] = []
    results = tavily_payload.get("results") if isinstance(tavily_payload, dict) else None
    if not isinstance(results, list) or not (required_marketing_type or "").strip():
        return out
    mt = str(required_marketing_type).strip()
    for r in results[:cap_results]:
        if not isinstance(r, dict):
            continue
        page_url = str(r.get("url") or "").strip()
        title = str(r.get("title") or "").strip()
        content = str(r.get("content") or "").strip()
        snippet = str(r.get("snippet") or "").strip()
        raw = str(r.get("raw_content") or "").strip()
        page_blob = " ".join(x for x in (page_url, title, snippet, content, raw) if x)
        if not _model_tokens_match_strict(page_blob, mt):
            continue
        imgs = r.get("images")
        if not isinstance(imgs, list):
            continue
        n = 0
        for im in imgs:
            if n >= cap_images_per_result:
                break
            norm = _normalize_tavily_image(im)
            if not norm:
                continue
            u = (norm.get("url") or "").strip()
            desc = norm.get("description") or ""
            if not _safe_https_image_url(u):
                continue
            blob = f"{u} {desc} {page_url} {title}"
            if _consultant_image_page_likely_fashion_editorial(u):
                continue
            if _tavily_image_blob_is_off_topic(blob):
                continue
            if _tavily_image_blob_human_fashion_tabloid_risk(blob):
                continue
            if _image_url_suspicious_for_gallery(u):
                continue
            if not _tavily_image_blob_has_aircraft_signal(blob):
                continue
            out.append(
                {
                    "url": u,
                    "description": _sanitize_tavily_image_description(desc),
                    "source": "tavily",
                    "page_url": page_url or None,
                }
            )
            n += 1
    return out


def _iter_tavily_result_linked_images(
    tavily_payload: Dict[str, Any],
    *,
    required_tail: str,
    cap_results: int = 20,
    cap_images_per_result: int = 8,
) -> List[Dict[str, Optional[str]]]:
    """
    Pull **source-linked** images from Tavily results whose page text mentions the required tail.
    (Option B: tail must be present on the source page, not necessarily in the image asset URL.)
    """
    out: List[Dict[str, Optional[str]]] = []
    results = tavily_payload.get("results") if isinstance(tavily_payload, dict) else None
    if not isinstance(results, list) or not required_tail:
        return out
    for r in results[:cap_results]:
        if not isinstance(r, dict):
            continue
        page_url = str(r.get("url") or "").strip()
        title = str(r.get("title") or "").strip()
        content = str(r.get("content") or "").strip()
        snippet = str(r.get("snippet") or "").strip()
        raw = str(r.get("raw_content") or "").strip()
        page_blob = " ".join(x for x in (page_url, title, snippet, content, raw) if x)
        if not _tail_token_present(page_blob, required_tail):
            continue
        imgs = r.get("images")
        if not isinstance(imgs, list):
            continue
        n = 0
        for im in imgs:
            if n >= cap_images_per_result:
                break
            norm = _normalize_tavily_image(im)
            if not norm:
                continue
            u = (norm.get("url") or "").strip()
            desc = norm.get("description") or ""
            if not _safe_https_image_url(u):
                continue
            blob = f"{u} {desc} {page_url} {title}"
            if _consultant_image_page_likely_fashion_editorial(u):
                continue
            if _tavily_image_blob_is_off_topic(blob):
                continue
            if _tavily_image_blob_human_fashion_tabloid_risk(blob):
                continue
            if _image_url_suspicious_for_gallery(u):
                continue
            if not _tavily_image_blob_has_aircraft_signal(blob):
                continue
            # Reject CDN paths that explicitly embed a different N-number than the requested tail.
            if _image_url_embeds_conflicting_us_tail(u, required_tail):
                continue
            out.append(
                {
                    "url": u,
                    "description": _sanitize_tavily_image_description(desc),
                    "source": "tavily",
                    "page_url": page_url or None,
                }
            )
            n += 1
    return out

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
_MAX_SCRAPE_IMAGES_TOTAL = 24

_GALLERY_CAP_MIN = 3
_GALLERY_CAP_MAX = 48


def consultant_gallery_image_cap(explicit_max: Optional[int] = None) -> int:
    """
    Max images returned in the consultant gallery (Tavily + listing sources).

    Default **5** (typical 3–5 visible photos per reply). Override with
    ``CONSULTANT_GALLERY_MAX_IMAGES`` (clamped ``_GALLERY_CAP_MIN``–``_GALLERY_CAP_MAX``).
    """
    if explicit_max is not None:
        try:
            return max(_GALLERY_CAP_MIN, min(_GALLERY_CAP_MAX, int(explicit_max)))
        except (TypeError, ValueError):
            pass
    raw = (os.getenv("CONSULTANT_GALLERY_MAX_IMAGES") or "5").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 5
    return max(_GALLERY_CAP_MIN, min(_GALLERY_CAP_MAX, n))

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


_US_TAIL_IN_TEXT = re.compile(r"\b(N[1-9A-Z][A-Z0-9]{1,5})\b", re.I)


def _norm_us_tail(s: str) -> str:
    return re.sub(r"[\s\-]+", "", (s or "").strip().upper())


def _phly_primary_us_tail(phly_rows: List[Dict[str, Any]]) -> str:
    for r in phly_rows[:4]:
        reg = (r.get("registration_number") or "").strip()
        u = _norm_us_tail(reg)
        if u.startswith("N") and len(u) >= 4 and re.search(r"\d", u):
            return u
    return ""


def _image_url_embeds_conflicting_us_tail(url: str, canonical_tail: str) -> bool:
    """
    True when the image URL text contains a different U.S. N-number than the resolved aircraft
    (e.g. article path ``...N503EA...`` while answering ``N508JA``).
    """
    if not canonical_tail or not url:
        return False
    canon = _norm_us_tail(canonical_tail)
    found = {m.group(1).upper() for m in _US_TAIL_IN_TEXT.finditer(url)}
    if not found:
        return False
    return any(t != canon for t in found)


_STOP_PHLY_MODEL_TOKENS = frozenset(
    {
        "the",
        "inc",
        "llc",
        "corp",
        "aircraft",
        "jet",
        "jets",
        "plane",
        "model",
        "series",
        "type",
    }
)


def _phly_rows_have_marketing_make_model(phly_rows: List[Dict[str, Any]]) -> bool:
    for r in phly_rows[:4]:
        if (r.get("manufacturer") or "").strip() and (r.get("model") or "").strip():
            return True
    return False


def _phly_model_or_type_in_blob(blob: str, phly_rows: List[Dict[str, Any]]) -> bool:
    """
    True when image URL/description text overlaps Phly **manufacturer + model** tokens.

    Reduces wrong-type jets when Tavily returns generic planespotter hits.
    """
    if not blob or not phly_rows:
        return False
    low = blob.lower()
    compact = re.sub(r"[^a-z0-9]+", "", low)
    for r in phly_rows[:4]:
        man = (r.get("manufacturer") or "").strip()
        mdl = (r.get("model") or "").strip()
        if not man and not mdl:
            continue
        tokens: List[str] = []
        for chunk in (man.lower(), mdl.lower()):
            for w in re.split(r"[\s/\-]+", chunk):
                w = w.strip()
                if len(w) >= 3 and w not in _STOP_PHLY_MODEL_TOKENS:
                    tokens.append(w)
        if not tokens:
            continue
        seen_t: set[str] = set()
        uniq: List[str] = []
        for t in tokens:
            if t not in seen_t:
                seen_t.add(t)
                uniq.append(t)
        hits = sum(1 for t in uniq if t in low)
        hits += sum(1 for t in uniq if len(t) >= 5 and t in compact)
        if hits >= 2:
            return True
        long_toks = [t for t in uniq if len(t) >= 5]
        if any(t in low for t in long_toks):
            return True
        combined = f"{man} {mdl}".strip().lower()
        if len(combined) >= 8 and combined in low:
            return True
    return False


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
        canon_us = _phly_primary_us_tail(phly_rows)
        if canon_us and _image_url_embeds_conflicting_us_tail(url, canon_us):
            continue
        blob = f"{url} {desc}"
        low = blob.lower()
        if _tavily_image_blob_is_off_topic(blob):
            continue
        id_match = _blob_matches_phly_aircraft_identity(blob, phly_rows)
        if _tavily_image_blob_human_fashion_tabloid_risk(blob) and not id_match:
            continue
        restrict_by_mm = trust_tail_biased_search and _phly_rows_have_marketing_make_model(phly_rows)

        if trust_tail_biased_search:
            if _image_url_suspicious_for_gallery(url) and not id_match:
                continue
            trusted_host = _url_host_matches_trusted_aircraft_media(url)
            if id_match:
                desc_out = _sanitize_tavily_image_description(desc) or "Aircraft (web search)"
                out.append({"url": url, "description": desc_out, "source": "tavily"})
            elif trusted_host and _tavily_image_blob_has_aircraft_signal(blob):
                if restrict_by_mm and not _phly_model_or_type_in_blob(blob, phly_rows):
                    continue
                # Opaque CDN paths on planespotter / listing hosts only — no tail-in-URL guesses from lifestyle CDNs.
                desc_out = _sanitize_tavily_image_description(desc) or "Aircraft (web search)"
                out.append({"url": url, "description": desc_out, "source": "tavily"})
            elif _tavily_image_blob_has_aircraft_signal(blob) and not _consultant_image_page_likely_fashion_editorial(
                url
            ):
                if restrict_by_mm and not _phly_model_or_type_in_blob(blob, phly_rows):
                    continue
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
    required_tail: Optional[str] = None,
    strict_tail_page_match: bool = False,
    required_marketing_type: Optional[str] = None,
    strict_model_title_alt_match: bool = False,
    max_listing_og_fetches: int = 6,
    og_timeout: float = 6.0,
    max_gallery_images: Optional[int] = None,
    user_query: str = "",
    history: Optional[List[Dict[str, str]]] = None,
    gallery_meta_out: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Deduped list for API/UI: {url, source: tavily|scrape_gallery|listing_og, page_url?, description?, lookup_key?}.

    ``listing_rows`` should be Postgres-aligned dicts (source_platform, source_listing_id, listing_url)
    so images can be resolved from pre-extracted JSON (see scrape_listing_image_lookup).

    ``trust_tail_biased_tavily_images``: set True when Tavily was run with a **photo-focused query**
    that already included the quoted tail/serial (see :func:`build_aircraft_photo_focus_tavily_query`);
    CDN URLs often omit the registration in the path, so identity matching on URL alone would drop them.

    ``max_gallery_images``: optional cap; if omitted, uses :func:`consultant_gallery_image_cap` (env
    ``CONSULTANT_GALLERY_MAX_IMAGES``, default 5, max 48).

    ``user_query`` / ``history``: optional thread text for SearchAPI query fan-out when ``SEARCHAPI_API_KEY`` is set
    (image search only; Tavily web snippets stay elsewhere).
    """
    seen: set[str] = set()
    final: List[Dict[str, Any]] = []

    cap = consultant_gallery_image_cap(max_gallery_images)

    # --- SearchAPI (Bing Images) replaces Tavily *image* hits when configured ---
    try:
        from services.searchapi_aircraft_images import (
            fetch_ranked_searchapi_aircraft_images,
            resolve_queries_for_consultant_gallery,
            searchapi_aircraft_images_enabled,
        )

        if searchapi_aircraft_images_enabled():
            strict_tail = bool(strict_tail_page_match and (required_tail or "").strip())
            rt = str(required_tail).strip() if required_tail else None
            queries, canon_tail, mm_for_score = resolve_queries_for_consultant_gallery(
                user_query=user_query or "",
                phly_rows=phly_rows,
                required_tail=rt,
                strict_tail_mode=strict_tail,
                required_marketing_type=(required_marketing_type or None),
                strict_model_mode=bool(
                    strict_model_title_alt_match and (required_marketing_type or "").strip()
                ),
            )
            if queries:
                _sea_meta: Dict[str, Any] = gallery_meta_out if gallery_meta_out is not None else {}
                sea, _ = fetch_ranked_searchapi_aircraft_images(
                    queries=queries,
                    canonical_tail=canon_tail if strict_tail else None,
                    strict_tail_mode=strict_tail,
                    marketing_type_for_model_match=(
                        (required_marketing_type or mm_for_score or "").strip()
                        if not strict_tail
                        else None
                    ),
                    max_out=cap,
                    user_query=user_query or "",
                    gallery_meta=_sea_meta,
                )
                if strict_model_title_alt_match and (required_marketing_type or "").strip():
                    mt = str(required_marketing_type).strip()
                    sea = [
                        r
                        for r in sea
                        if _model_tokens_match_strict(
                            f"{(r.get('url') or '')} {(r.get('description') or '')}",
                            mt,
                        )
                    ]
                # Strict tail: SearchAPI-only, no listing/model substitution (Part 3).
                if strict_tail:
                    return sea[:cap]
                # Strict model: no listing galleries mixed in (same policy as Tavily strict-model path).
                if strict_model_title_alt_match and (required_marketing_type or "").strip():
                    return sea[:cap]
                for r in sea:
                    u = (r.get("url") or "").strip()
                    if not u or u in seen:
                        continue
                    seen.add(u)
                    final.append(r)
                if len(final) >= cap:
                    return final[:cap]
                # Fall through: append listing scrape + og:image using the same dedupe rules as Tavily mode.
                tavily_payload = dict(tavily_payload or {})
                tavily_payload["images"] = []
    except Exception as _sea_e:
        logger.debug("SearchAPI gallery path skipped: %s", _sea_e)

    # Strict tail mode (Option B): only accept images that are linked to pages mentioning the tail token.
    if strict_tail_page_match and (required_tail or "").strip():
        for row in _iter_tavily_result_linked_images(
            tavily_payload,
            required_tail=str(required_tail).strip(),
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
                    # Source-linked: the UI "Open" link should go to the result page, not the raw CDN image.
                    "page_url": row.get("page_url"),
                    "lookup_key": None,
                }
            )
            if len(final) >= cap:
                return final[:cap]
        # No tail-verified images found → do not fall back to model/listing imagery in strict-tail mode.
        return final[:cap]

    # Strict model mode (Option B-like): try page-verified source-linked images first (URLs/alt often omit the model).
    if strict_model_title_alt_match and (required_marketing_type or "").strip():
        mt = str(required_marketing_type).strip()
        rows = _iter_tavily_result_linked_images_for_model(
            tavily_payload,
            required_marketing_type=mt,
        )
        rows = sorted(rows, key=lambda r: _host_priority_score((r.get("url") or "").strip()), reverse=True)
        for row in rows:
            u = (row.get("url") or "").strip()
            if not u or u in seen:
                continue
            seen.add(u)
            final.append(
                {
                    "url": u,
                    "source": row.get("source") or "tavily",
                    "description": row.get("description"),
                    "page_url": row.get("page_url"),
                    "lookup_key": None,
                }
            )
            if len(final) >= cap:
                return final[:cap]

    raw_imgs = tavily_payload.get("images") if isinstance(tavily_payload, dict) else None
    if not isinstance(raw_imgs, list):
        raw_imgs = []
    # Pull enough Tavily candidates so filtering can still fill ``cap`` after drops.
    tavily_cap = min(max(cap, 10), 32 if trust_tail_biased_tavily_images else 24)
    tavily_rows = filter_tavily_images_for_phly(
        raw_imgs,
        phly_rows,
        max_out=tavily_cap,
        trust_tail_biased_search=trust_tail_biased_tavily_images,
    )
    # Strict model mode: enforce title/alt/url contains the model token, reject common confusables.
    if strict_model_title_alt_match and (required_marketing_type or "").strip():
        mt = str(required_marketing_type).strip()
        filtered: List[Dict[str, Optional[str]]] = []
        for row in tavily_rows:
            u = (row.get("url") or "").strip()
            d = row.get("description") or ""
            blob = f"{u} {d}"
            if not _model_tokens_match_strict(blob, mt):
                continue
            filtered.append(row)
        tavily_rows = filtered

    # Rank by aviation host priority first, then keep stable order.
    tavily_rows = sorted(
        tavily_rows,
        key=lambda r: _host_priority_score((r.get("url") or "").strip()),
        reverse=True,
    )

    for row in tavily_rows:
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
        if len(final) >= cap:
            return final[:cap]

    # Pre-scraped gallery URLs (Controller / AircraftExchange) keyed by marketplace listing id.
    n_scrape = 0
    if strict_model_title_alt_match or strict_tail_page_match:
        # When user asked for strict-tail or strict-model matching, do not substitute listing-gallery images.
        listing_rows = None
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
    if strict_model_title_alt_match or strict_tail_page_match:
        listing_urls = None
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

    return final[:cap]
