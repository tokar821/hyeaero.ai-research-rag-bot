"""
Server-side fetch of consultant gallery image URLs for PDF export.

Browsers cannot read most third-party image URLs (no CORS); the frontend calls this endpoint
with the same HTTPS URLs returned by Ask Consultant.
"""

from __future__ import annotations

import ipaddress
import logging
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

_MAX_BYTES = 4_500_000
_TIMEOUT = (5.0, 20.0)

# Host substrings for URLs we expect from Tavily / listing galleries (SSRF guard).
_ALLOWED_HOST_MARKERS = (
    "jetphotos.com",
    "jetphotos.net",
    "airliners.net",
    "planespotters.net",
    "planepictures.net",
    "airplane-pictures.net",
    "flightaware.com",
    "flightradar24",
    "airport-data.com",
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
    "bombardier.com",
    "gulfstream.com",
    "txtav.com",
    "embraer.com",
    "cessna.com",
    "cloudinary.com",
    "cloudfront.net",
    "amazonaws.com",
    "amazon.com",
    "akamaized.net",
    "azureedge.net",
    "fastly.net",
    "imgix.net",
    "googleusercontent.com",
    "ggpht.com",
    "twimg.com",
    "wp.com",
    "wordpress.com",
    "simpleflyingimages.com",
    "simpleflying.com",
    "bjtonline.com",
    "cirrusaircraft.com",
    "flyusa.com",
    "mira-aviation.com",
    "bluebird-aviation.com",
    "grupooneair.com",
    "aviation-safety.net",
    "abpic.co.uk",
    "abpic.net",
    "aviationweek.com",
    "ainonline.com",
    "flightglobal.com",
    # Common “aircraft guide / news / operator” pages that Tavily often returns as image hosts.
    "globalmilitary.net",
    "businessworld.in",
    "navalnews.com",
    "albajet.com",
    "aircharterservice.com",
    "celebrityprivatejettracker.com",
    "squarespace-cdn.com",
    "shopify.com",
    "cdninstagram.com",
    "fbcdn.net",
    "digitaloceanspaces.com",
    "r2.cloudflarestorage.com",
)


def consultant_report_image_url_allowed(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    u = url.strip()
    if not u.startswith("https://"):
        return False
    if len(u) > 2048:
        return False
    try:
        p = urlparse(u)
    except Exception:
        return False
    host = (p.netloc or "").lower().split(":")[0]
    if not host or host.startswith("127.") or host == "localhost":
        return False
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
            return False
    except ValueError:
        pass
    return any(m in host for m in _ALLOWED_HOST_MARKERS)


def fetch_consultant_report_image(url: str) -> Optional[Tuple[bytes, str]]:
    """
    Download image bytes. Returns (body, content_type) or None on failure.
    """
    if not consultant_report_image_url_allowed(url):
        return None
    try:
        r = requests.get(
            url,
            timeout=_TIMEOUT,
            headers={
                "User-Agent": "HyeAero-Consultant-PDF/1.0 (image export; +https://hyeaero.ai)",
                "Accept": "image/*,*/*;q=0.8",
            },
            stream=True,
            allow_redirects=True,
        )
        r.raise_for_status()
        ct = (r.headers.get("Content-Type") or "application/octet-stream").split(";")[0].strip().lower()
        if ct and not ct.startswith("image/") and ct != "application/octet-stream":
            return None
        chunks: list[bytes] = []
        total = 0
        for chunk in r.iter_content(chunk_size=65536):
            if not chunk:
                continue
            total += len(chunk)
            if total > _MAX_BYTES:
                return None
            chunks.append(chunk)
        body = b"".join(chunks)
        if not body:
            return None
        if not ct.startswith("image/"):
            if body[:8] == b"\x89PNG\r\n\x1a\n":
                ct = "image/png"
            elif body[:2] == b"\xff\xd8":
                ct = "image/jpeg"
            elif body[:4] == b"RIFF" and body[8:12] == b"WEBP":
                ct = "image/webp"
            else:
                ct = "image/jpeg"
        return body, ct
    except Exception as e:
        logger.debug("consultant report image fetch failed: %s", e)
        return None
