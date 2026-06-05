"""
Image verification tiers — verified / likely / unverified / rejected for tail galleries.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_PLACEHOLDER_URL_RE = re.compile(
    r"(?is)(?:"
    r"/g650\.png|/g550\.png|/g280\.png|/challenger\.png|"
    r"/placeholder|/default[_-]?image|"
    r"tailimages/|tail_images_fixed|matched\+images|"
    r"/n\d{1,5}[a-z]{0,3}\.png|"
    r"map\.view\.|subscription/features|/static/images/premium/"
    r")"
)
_TRUSTED_TAIL_SOURCE_RE = re.compile(
    r"(?is)(?:jetphotos|planespotters|aircraft\.com|virtualhangar|flightradar|flightaware|"
    r"aviapages|airliners\.net|globalair)"
)
_EXTERIOR_BLOB_RE = re.compile(
    r"(?is)\b(?:exterior|ramp|taxi|takeoff|landing|airborne|winglets|fuselage|"
    r"planespotters|jetphotos|spotting)\b"
)
_CABIN_BLOB_RE = re.compile(
    r"(?is)\b(?:cabin|interior|salon|galley|divan|seating|lavatory|lopa)\b|lopa|/interior"
)


def is_rejected_gallery_image_url(url: str) -> bool:
    """True for placeholder / marketing junk assets that must never surface in galleries."""
    u = (url or "").strip()
    if not u:
        return True
    return bool(_PLACEHOLDER_URL_RE.search(u))


def filter_rejected_gallery_rows(
    images: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop rows whose image URL is a known placeholder or junk marketing asset."""
    out: List[Dict[str, Any]] = []
    for row in images or []:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url") or row.get("image") or row.get("image_url") or "")
        if is_rejected_gallery_image_url(url):
            continue
        try:
            from services.tail_marketing_listing_images import _image_url_looks_junk_marketing_asset

            if _image_url_looks_junk_marketing_asset(url):
                continue
        except Exception:
            pass
        out.append(row)
    return out


def classify_image_trust_tier(
    row: Dict[str, Any],
    *,
    tail: str = "",
    user_query: str = "",
) -> str:
    """
    Return one of: verified, likely, unverified, rejected.
    """
    if not isinstance(row, dict):
        return "rejected"

    url = str(row.get("url") or row.get("image_url") or "").strip()
    prov = str(row.get("image_provenance") or row.get("gallery_label") or "").lower()
    facet = str(row.get("visual_facet") or "").lower()
    blob = " ".join(
        str(row.get(k) or "")
        for k in ("description", "title", "url", "page_url", "source", "gallery_label")
    ).lower()
    tail_tok = (tail or "").strip().upper()

    if _PLACEHOLDER_URL_RE.search(url) or re.search(r"(?is)g650\.png|challenger\.png", blob):
        return "rejected"

    wants_cabin = bool(re.search(r"(?is)\bcabin|cabine|interior\b", user_query or ""))
    is_exterior = facet == "exterior" or _EXTERIOR_BLOB_RE.search(blob)
    is_cabin = facet == "cabin" or _CABIN_BLOB_RE.search(blob)

    if wants_cabin and is_exterior and not is_cabin:
        return "rejected"

    if prov in ("tail_exact_listing_cabin", "tail_exact_listing") or "listing cabin (exact tail)" in blob:
        if wants_cabin and not is_cabin and "cabin" not in prov:
            return "likely"
        return "verified"

    if tail_tok and tail_tok.lower() in blob and is_cabin:
        return "verified" if "listing" in blob or "virtualhangar" in blob else "likely"

    if tail_tok and tail_tok.lower() in blob:
        if _TRUSTED_TAIL_SOURCE_RE.search(blob) or _TRUSTED_TAIL_SOURCE_RE.search(url):
            return "likely"
        if is_exterior or re.search(r"(?is)\b(?:aircraft|jet|aviation|photo)\b", blob):
            return "likely"
        return "unverified"

    if prov.startswith("representative") or "representative" in blob:
        return "unverified"

    return "unverified"


def filter_and_tier_gallery_images(
    images: List[Dict[str, Any]],
    *,
    tail: str = "",
    user_query: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Reject bad rows; attach trust_tier; return counts."""
    counts = {"verified": 0, "likely": 0, "unverified": 0, "rejected": 0}
    out: List[Dict[str, Any]] = []
    for row in images or []:
        if not isinstance(row, dict):
            continue
        tier = classify_image_trust_tier(row, tail=tail, user_query=user_query)
        counts[tier] = counts.get(tier, 0) + 1
        if tier == "rejected":
            continue
        item = dict(row)
        item["trust_tier"] = tier
        out.append(item)
    return out, counts


def render_gallery_tier_prose(
    query: str,
    *,
    tail: str,
    model: str,
    counts: Dict[str, int],
    total_before_filter: int,
) -> str:
    label = tail or "this tail"
    head = f"**{label}**" + (f" ({model})" if model else "")
    v = int(counts.get("verified", 0) or 0)
    lk = int(counts.get("likely", 0) or 0)
    u = int(counts.get("unverified", 0) or 0)
    rej = int(counts.get("rejected", 0) or 0)
    shown = v + lk + u

    wants_cockpit = bool(re.search(r"(?is)\bcockpit\b", query or ""))
    wants_cabin = bool(re.search(r"(?is)\bcabin|cabine|interior\b", query or ""))
    if wants_cockpit:
        facet = "cockpit"
    elif wants_cabin:
        facet = "cabin"
    else:
        facet = "exterior/photo"

    if shown == 0:
        return (
            f"No **{facet}** photos for {head} this turn. "
            "I'd check JetPhotos, the listing page, or FlightAware — caption must match this registration."
        )

    if wants_cockpit:
        noun = "cockpit photo"
    elif wants_cabin:
        noun = "cabin photo"
    else:
        noun = "photo"
    if shown != 1:
        noun += "s"
    lead = f"{head} — **{shown}** {noun} below."
    if v:
        lead += " Tail-tied sources."
    elif lk:
        lead += " Confirm registration on the source page before you treat these as this aircraft."
    return lead


__all__ = [
    "classify_image_trust_tier",
    "filter_and_tier_gallery_images",
    "filter_rejected_gallery_rows",
    "is_rejected_gallery_image_url",
    "render_gallery_tier_prose",
]
