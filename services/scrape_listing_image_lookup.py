"""
Map consultant / Postgres listing rows to pre-extracted gallery image URLs.

JSON is produced by etl-pipeline/scripts/extract_scrape_image_urls.py (one file per platform).
Each listing row includes ``lookup_key``, ``listing_page_url`` (from ``og:url``), and a top-level
``by_lookup_key`` map (deduped URLs if the same listing id appears in multiple scrape batches).
Extra fields inside ``by_lookup_key`` values are ignored; only ``image_urls`` is read for lookup.

Lookup key (stable join to consultant / DB)
-------------------------------------------
``"{platform}:{marketplace_listing_id}"`` e.g. ``controller:250049455``, ``aircraftexchange:5289``.

Resolve from a listing row (in order):

1. ``normalize_listing_source_platform(source_platform)`` + ``source_listing_id`` (or id parsed from URL).
2. If ``source_platform`` is missing or unknown, infer platform from ``listing_url`` host/path
   (Controller.com / AircraftExchange) and parse the id from the URL.

Use :func:`lookup_images_by_listing_url` when you only have a URL string.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_index: Dict[str, Dict[str, List[str]]] = {}
_loaded: bool = False

_CONTROLLER_ID_RE = re.compile(r"/listing/for-sale/(\d+)/", re.IGNORECASE)
_AE_ID_RE = re.compile(r"/details/(\d+)/", re.IGNORECASE)

# Normalize aircraft_listings.source_platform values to JSON platform folder names.
_PLATFORM_ALIASES = {
    "controller": "controller",
    "controller.com": "controller",
    "aircraftexchange": "aircraftexchange",
    "aircraft exchange": "aircraftexchange",
    "aircraft_exchange": "aircraftexchange",
    "aircraft-exchange": "aircraftexchange",
    "ae": "aircraftexchange",
    "iada": "aircraftexchange",
}


def normalize_listing_source_platform(raw: str) -> str:
    s = (raw or "").strip().lower()
    if not s:
        return ""
    if s in _PLATFORM_ALIASES:
        return _PLATFORM_ALIASES[s]
    if "controller" in s:
        return "controller"
    if "aircraftexchange" in s or "aircraft exchange" in s:
        return "aircraftexchange"
    return s


def _as_str_id(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _default_json_dir() -> Path:
    env = (os.getenv("SCRAPE_IMAGE_URLS_DIR") or "").strip()
    if env:
        return Path(env).resolve()
    root = Path(__file__).resolve().parents[2]
    return root / "etl-pipeline" / "store" / "derived" / "scrape_image_urls"


def infer_platform_from_listing_url(listing_url: str) -> str:
    """Return ``controller`` or ``aircraftexchange`` when URL implies it; else ``""``."""
    u = (listing_url or "").strip().lower()
    if not u:
        return ""
    if "controller.com" in u or "/listing/for-sale/" in u:
        return "controller"
    if "aircraftexchange.com" in u:
        return "aircraftexchange"
    return ""


def parse_external_listing_id(listing_url: str, source_platform: str) -> Optional[str]:
    """When source_listing_id is null in DB, derive marketplace listing id from URL."""
    u = (listing_url or "").strip()
    if not u:
        return None
    plat = (source_platform or "").strip().lower()
    if plat == "controller" or "controller.com" in u.lower():
        m = _CONTROLLER_ID_RE.search(u)
        return m.group(1) if m else None
    if plat == "aircraftexchange" or "aircraftexchange.com" in u.lower():
        m = _AE_ID_RE.search(u)
        return m.group(1) if m else None
    return None


def _load_index() -> None:
    global _loaded, _index
    with _lock:
        if _loaded:
            return
        base = _default_json_dir()
        if not base.is_dir():
            logger.debug("scrape_listing_image_lookup: directory missing: %s", base)
            _loaded = True
            return
        merged: Dict[str, Dict[str, List[str]]] = {}
        for path in sorted(base.glob("*_image_urls.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("scrape_listing_image_lookup: skip %s (%s)", path.name, e)
                continue
            plat = (data.get("platform") or "").strip()
            by_lk = data.get("by_lookup_key")
            if isinstance(by_lk, dict) and plat:
                for lk, row in by_lk.items():
                    if not isinstance(lk, str) or not lk.startswith(f"{plat}:"):
                        continue
                    if not isinstance(row, dict):
                        continue
                    urls = row.get("image_urls")
                    if isinstance(urls, list):
                        merged[lk] = {"image_urls": [str(u) for u in urls if u]}
                continue
            for row in data.get("files") or []:
                if not isinstance(row, dict):
                    continue
                lk = row.get("lookup_key")
                if not isinstance(lk, str) or not lk:
                    pid = row.get("page_id")
                    if plat and pid is not None:
                        lk = f"{plat}:{pid}"
                    else:
                        continue
                urls = row.get("image_urls")
                if isinstance(urls, list):
                    merged[lk] = {"image_urls": [str(u) for u in urls if u]}
        _index = merged
        _loaded = True
        logger.info(
            "scrape_listing_image_lookup: loaded %d lookup keys from %s",
            len(_index),
            base,
        )


def lookup_images(lookup_key: str) -> List[str]:
    _load_index()
    row = _index.get(lookup_key)
    if not row:
        return []
    return list(row.get("image_urls") or [])


def listing_image_lookup_key(row: Dict[str, Any]) -> str:
    """Stable key matching JSON index: ``{platform}:{marketplace_listing_id}``."""
    plat_raw = (row.get("source_platform") or "").strip()
    plat = normalize_listing_source_platform(plat_raw)
    url = (row.get("listing_url") or "").strip()
    if plat not in ("controller", "aircraftexchange"):
        plat = infer_platform_from_listing_url(url)
    if plat not in ("controller", "aircraftexchange"):
        return ""
    sid = (row.get("source_listing_id") or "").strip()
    if not sid:
        sid = parse_external_listing_id(url, plat) or ""
    if not plat or not sid:
        return ""
    return f"{plat}:{sid}"


def lookup_images_by_listing_url(listing_url: str) -> List[str]:
    """Resolve gallery URLs using only ``listing_url`` (no DB platform column required)."""
    u = (listing_url or "").strip()
    if not u:
        return []
    plat = infer_platform_from_listing_url(u)
    if plat not in ("controller", "aircraftexchange"):
        return []
    sid = parse_external_listing_id(u, plat) or ""
    if not sid:
        return []
    return lookup_images(f"{plat}:{sid}")


def images_for_listing_row(row: Dict[str, Any]) -> List[str]:
    lk = listing_image_lookup_key(row)
    if lk:
        return lookup_images(lk)
    return lookup_images_by_listing_url((row.get("listing_url") or "").strip())


def reset_cache_for_tests() -> None:
    global _loaded, _index
    with _lock:
        _loaded = False
        _index = {}
