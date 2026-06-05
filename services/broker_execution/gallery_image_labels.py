"""
Stable gallery labels for UI — tail-exact vs listing vs representative (model/facet).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_TAIL_MARKETING_PROVENANCE = frozenset(
    {"tail_marketing_listing", "tail_exact_listing_cabin", "tail_exact_listing"}
)

_REPRESENTATIVE_PROVENANCE = frozenset(
    {"representative_model_cabin", "representative_model", "model_representative_cabin"}
)


def _infer_visual_facet(row: Dict[str, Any], *, user_query: str = "") -> str:
    prov = str(row.get("image_provenance") or "").lower()
    blob = " ".join(
        str(row.get(k) or "")
        for k in ("description", "title", "url", "page_url", "source")
    ).lower()
    if prov in _REPRESENTATIVE_PROVENANCE and "cabin" in prov:
        return "cabin"
    if re.search(r"(?is)\b(cabin|interior|salon|galley|seating|divan)\b", blob):
        return "cabin"
    if re.search(r"(?is)\b(cockpit|flight\s*deck|avionics)\b", blob):
        return "cockpit"
    if re.search(r"(?is)\b(exterior|ramp|taxi|airborne|fuselage|winglets)\b", blob):
        return "exterior"
    q = (user_query or "").lower()
    if re.search(r"(?is)\bcabin|interior\b", q):
        return "cabin"
    if re.search(r"(?is)\bcockpit\b", q):
        return "cockpit"
    if re.search(r"(?is)\bexterior\b", q):
        return "exterior"
    return "general"


def resolve_gallery_row_label(
    row: Dict[str, Any],
    *,
    tail: str = "",
    gallery_meta: Optional[Dict[str, Any]] = None,
    user_query: str = "",
) -> Dict[str, str]:
    """Return ``gallery_label``, ``image_provenance``, ``visual_facet`` for one image row."""
    meta = gallery_meta if isinstance(gallery_meta, dict) else {}
    prov = str(row.get("image_provenance") or "").strip().lower()
    source = str(row.get("source") or "").strip().lower()
    desc = str(row.get("description") or row.get("title") or "").lower()
    page = str(row.get("page_url") or row.get("_source_page") or "").lower()
    tail_tok = (tail or "").strip().upper()
    facet = _infer_visual_facet(row, user_query=user_query)

    if prov in _TAIL_MARKETING_PROVENANCE or (
        source == "listing_scrape"
        and tail_tok
        and (tail_tok.lower() in desc or "virtualhangar" in page or "website-files.com" in str(row.get("url") or ""))
    ):
        label = "Listing cabin (exact tail)" if facet == "cabin" else "Listing photo (exact tail)"
        return {
            "gallery_label": label,
            "image_provenance": "tail_exact_listing_cabin" if facet == "cabin" else "tail_exact_listing",
            "visual_facet": facet,
        }

    if meta.get("consultant_tail_listing_cabin_enriched") and source in (
        "listing_scrape",
        "listing_og",
    ):
        return {
            "gallery_label": "Listing cabin (exact tail)",
            "image_provenance": "tail_exact_listing_cabin",
            "visual_facet": "cabin",
        }

    if prov in _REPRESENTATIVE_PROVENANCE or meta.get("consultant_cabin_image_tier") == "representative_model":
        if facet == "cabin" or "cabin" in prov:
            return {
                "gallery_label": "Representative cabin (model)",
                "image_provenance": "representative_model_cabin",
                "visual_facet": "cabin",
            }
        return {
            "gallery_label": "Representative exterior (model)",
            "image_provenance": "representative_model",
            "visual_facet": facet if facet != "general" else "exterior",
        }

    if meta.get("consultant_tail_led_fallback_to_model_images"):
        return {
            "gallery_label": "Representative (model — tail photos sparse)",
            "image_provenance": "representative_model",
            "visual_facet": facet,
        }

    tail_conf = str(row.get("tail_match_confidence") or meta.get("consultant_gallery_tail_confidence") or "")
    if tail_tok and (tail_tok.lower() in desc or tail_conf in ("confirmed", "high", "exact")):
        if facet == "cabin":
            short = "Cabin"
        elif facet == "exterior":
            short = "Exterior"
        elif facet == "cockpit":
            short = "Cockpit"
        else:
            short = "Photo"
        return {
            "gallery_label": short,
            "image_provenance": "tail_exact",
            "visual_facet": facet,
        }

    if source == "scrape_gallery":
        return {
            "gallery_label": "Marketplace gallery",
            "image_provenance": "marketplace_gallery",
            "visual_facet": facet,
        }

    if source in ("listing_og", "listing_scrape"):
        return {
            "gallery_label": "Listing preview",
            "image_provenance": "listing_preview",
            "visual_facet": facet,
        }

    if facet == "cabin":
        web_label = "Web search — cabin"
    elif facet == "exterior":
        web_label = "Web search — exterior"
    elif facet == "cockpit":
        web_label = "Web search — cockpit"
    else:
        web_label = "Web search"
    return {
        "gallery_label": web_label,
        "image_provenance": prov or "web_search",
        "visual_facet": facet,
    }


def annotate_consultant_gallery_images(
    images: List[Dict[str, Any]],
    *,
    tail: str = "",
    gallery_meta: Optional[Dict[str, Any]] = None,
    user_query: str = "",
) -> List[Dict[str, Any]]:
    """Attach UI-stable label fields to each gallery row (mutates copies)."""
    if not images:
        return []
    out: List[Dict[str, Any]] = []
    for row in images:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        labels = resolve_gallery_row_label(
            item,
            tail=tail,
            gallery_meta=gallery_meta,
            user_query=user_query,
        )
        item.update(labels)
        out.append(item)
    return out


__all__ = [
    "annotate_consultant_gallery_images",
    "resolve_gallery_row_label",
]
