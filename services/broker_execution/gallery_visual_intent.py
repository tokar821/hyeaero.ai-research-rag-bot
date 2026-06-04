"""
Gallery visual-intent filter — cabin/cockpit/exterior discipline for SearchAPI results.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_CABIN_BLOB_RE = re.compile(
    r"(?is)\b(?:aircraft\s+cabin|bizjet\s+cabin|jet\s+cabin|vip\s+cabin|"
    r"cabin\s+interior|interior\s+cabin|galley|divan|berth|lavatory|"
    r"club\s+seating|seating\s+layout|salon)\b"
)
_EXTERIOR_BLOB_RE = re.compile(
    r"(?is)\b(?:ramp\b|taxiing|taxi\b|takeoff|landing\b|walkaround|"
    r"air[- ]to[- ]air|parked\s+on\s+ramp|exterior\b|winglets?\b|"
    r"planespotting|spotter)\b"
)
_COCKPIT_BLOB_RE = re.compile(r"(?is)\b(?:cockpit|flight\s*deck|flightdeck)\b")


def resolve_gallery_visual_intent(
    user_query: str,
    premium_intent: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolve primary visual facet (cabin, cockpit, exterior, or any)."""
    try:
        from services.searchapi_aircraft_images import detect_query_image_intent

        intent = detect_query_image_intent(user_query)
        if intent:
            return intent
    except Exception:
        pass
    pi = premium_intent if isinstance(premium_intent, dict) else {}
    it = str(pi.get("image_type") or "").strip().lower()
    if it in ("cabin", "interior", "cockpit", "exterior"):
        return "interior" if it == "interior" else it
    facets = pi.get("image_facets")
    if isinstance(facets, list) and facets:
        f0 = str(facets[0]).strip().lower()
        if f0 in ("cabin", "interior", "cockpit", "exterior"):
            return "interior" if f0 == "interior" else f0
    return "any"


def _row_blob(row: Dict[str, Any]) -> str:
    return " ".join(
        str(row.get(k) or "")
        for k in ("url", "title", "description", "source", "page_url", "_source_page", "caption")
    ).lower()


def row_matches_visual_facet(row: Dict[str, Any], facet: str) -> bool:
    """True when image metadata supports the requested facet."""
    f = (facet or "any").strip().lower()
    if f in ("any", ""):
        return True
    if f == "interior":
        f = "cabin"
    blob = _row_blob(row)
    if f == "cabin":
        if _CABIN_BLOB_RE.search(blob):
            return True
        try:
            from services.tail_marketing_listing_images import row_is_tail_listing_cabin_candidate

            page = str(row.get("page_url") or row.get("_source_page") or "")
            tail_m = re.search(
                r"(?i)/aircraft/(n[1-9a-z][a-z0-9]{1,5})|(?:\b)(n[1-9a-z][a-z0-9]{1,5})(?:\b)",
                page,
            )
            if tail_m:
                tail_tok = tail_m.group(1) or tail_m.group(2)
                if row_is_tail_listing_cabin_candidate(row, tail_tok):
                    return True
        except Exception:
            pass
        if _EXTERIOR_BLOB_RE.search(blob) and not _CABIN_BLOB_RE.search(blob):
            return False
        if _COCKPIT_BLOB_RE.search(blob) and not _CABIN_BLOB_RE.search(blob):
            return False
        # Spotter pages without cabin cues are usually ramp/exterior shots.
        if re.search(r"jetphotos|planespotters|airliners\.net", blob, re.I):
            return False
        return bool(re.search(r"\binterior\b", blob, re.I))
    if f == "cockpit":
        return bool(_COCKPIT_BLOB_RE.search(blob)) and not (
            _CABIN_BLOB_RE.search(blob) and not _COCKPIT_BLOB_RE.search(blob)
        )
    if f == "exterior":
        return bool(_EXTERIOR_BLOB_RE.search(blob)) or not _CABIN_BLOB_RE.search(blob)
    return True


def filter_gallery_by_visual_intent(
    images: List[Dict[str, Any]],
    *,
    facet: str,
    max_out: int = 12,
) -> List[Dict[str, Any]]:
    """
    Keep only rows matching the facet; prefer facet matches first.

    When cabin is requested and nothing matches, return [] (caller may run model cabin fallback).
    """
    f = (facet or "any").strip().lower()
    if f in ("any", ""):
        return images[:max_out]

    matched: List[Dict[str, Any]] = []
    rest: List[Dict[str, Any]] = []
    for row in images or []:
        if not isinstance(row, dict):
            continue
        if row_matches_visual_facet(row, f):
            matched.append(row)
        else:
            rest.append(row)

    if f == "cabin":
        out = matched[:max_out]
        return out

    out = matched + rest
    return out[:max_out]


__all__ = [
    "filter_gallery_by_visual_intent",
    "resolve_gallery_visual_intent",
    "row_matches_visual_facet",
]
