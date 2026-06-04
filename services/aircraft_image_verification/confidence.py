"""
Image confidence scoring (0–1) — internal telemetry; gates user-visible gallery.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.aircraft_image_verification.rejection import ImageVerificationContext
from services.aircraft_image_verification.source_ranking import classify_source_tier, source_trust_component


@dataclass
class ImageConfidenceBreakdown:
    model: str = ""
    tail: str = ""
    match_type: str = ""  # tail_exact | model_exact | none
    identity: float = 0.0
    source_trust: float = 0.0
    section_fit: float = 0.0
    listing_verification: float = 0.0
    total: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_type": self.match_type,
            "identity": round(self.identity, 4),
            "source_trust": round(self.source_trust, 4),
            "section_fit": round(self.section_fit, 4),
            "listing_verification": round(self.listing_verification, 4),
            "total": round(self.total, 4),
            "notes": list(self.notes),
        }


def _blob(row: Dict[str, Any]) -> str:
    return " ".join(
        str(row.get(k) or "")
        for k in ("url", "title", "source", "description", "_source_page", "page_url", "snippet")
    ).lower()


def _section_fit_score(blob: str, section: str, row: Optional[Dict[str, Any]] = None) -> float:
    from services.aircraft_image_verification.rejection import (
        _CABIN_STRONG,
        _COCKPIT_STRONG,
        _EXTERIOR_STRONG,
    )

    sec = (section or "exterior").strip().lower()
    if sec == "interior":
        sec = "cabin"
    if sec == "cabin":
        if _CABIN_STRONG.search(blob):
            return 1.0
        try:
            from services.tail_marketing_listing_images import row_is_tail_listing_cabin_candidate

            page = str(row.get("page_url") or row.get("_source_page") or "")
            tail_m = re.search(
                r"(?i)/aircraft/(n[1-9a-z][a-z0-9]{1,5})|(?:\b)(n[1-9a-z][a-z0-9]{1,5})(?:\b)",
                page,
            )
            if tail_m and row_is_tail_listing_cabin_candidate(row, tail_m.group(1) or tail_m.group(2)):
                return 0.92
        except Exception:
            pass
        if re.search(r"jetphotos|planespotters", blob, re.I):
            return 0.55
        if _COCKPIT_STRONG.search(blob) or _EXTERIOR_STRONG.search(blob):
            return 0.2
        return 0.45
    if sec == "cockpit":
        if _COCKPIT_STRONG.search(blob):
            return 1.0
        if _CABIN_STRONG.search(blob) and not _COCKPIT_STRONG.search(blob):
            return 0.15
        return 0.4
    if sec == "exterior":
        if _EXTERIOR_STRONG.search(blob):
            return 1.0
        if _CABIN_STRONG.search(blob) and not _EXTERIOR_STRONG.search(blob):
            return 0.25
        return 0.55
    return 0.5


def score_image_confidence(
    row: Dict[str, Any],
    ctx: ImageVerificationContext,
) -> ImageConfidenceBreakdown:
    """
    Composite confidence in [0, 1].

    Weights (sum ≈ 1.0):
      identity 0.45 | source 0.25 | section 0.15 | listing verify 0.15
    """
    blob = _blob(row)
    url = str(row.get("url") or row.get("imageUrl") or "")
    page = str(row.get("_source_page") or row.get("page_url") or "")
    title = str(row.get("title") or row.get("description") or "")

    bd = ImageConfidenceBreakdown()
    notes: List[str] = []

    # --- identity ---
    if ctx.tail:
        bd.match_type = "tail_exact"
        bd.tail = ctx.tail
        try:
            from services.tail_marketing_listing_images import is_tail_marketing_listing_page

            if is_tail_marketing_listing_page(page, ctx.tail):
                bd.identity = 0.45
                notes.append("tail_marketing_listing_page")
        except Exception:
            pass
        if bd.identity < 0.4:
            try:
                from services.searchapi_aircraft_images import (
                    classify_tail_match_confidence,
                    compute_tail_match_score,
                )

                ts = int(row.get("_tail_match_score") or compute_tail_match_score(row, ctx.tail))
                conf = row.get("_tail_confidence") or classify_tail_match_confidence(ts)
                if conf == "confirmed":
                    bd.identity = 0.45
                    notes.append("tail_match_confirmed")
                elif conf == "probable":
                    bd.identity = 0.34
                    notes.append("tail_match_probable")
                else:
                    bd.identity = 0.22
                    notes.append("tail_match_weak")
            except Exception:
                from rag.aviation_tail import normalize_tail_token
                from services.searchapi_aircraft_images import strip_domains

                t = normalize_tail_token(ctx.tail)
                if t in strip_domains(blob).upper():
                    bd.identity = 0.38
                else:
                    bd.identity = 0.15
    elif ctx.model:
        bd.match_type = "model_exact"
        bd.model = ctx.model
        try:
            from services.consultant_aircraft_images import _model_tokens_match_strict

            if _model_tokens_match_strict(blob, ctx.model):
                bd.identity = 0.43
                notes.append("model_token_strict_match")
            else:
                bd.identity = 0.12
        except Exception:
            ml = ctx.model.lower()
            bd.identity = 0.38 if ml in blob else 0.12
    else:
        bd.match_type = "none"
        bd.identity = 0.0

    # --- source trust ---
    tier, raw = classify_source_tier(
        url=url,
        page_url=page,
        source_label=str(row.get("source") or ""),
        title=title,
    )
    trust = source_trust_component(tier, raw)
    bd.source_trust = round(0.25 * trust, 4)
    notes.append(f"source_tier:{tier.value}")

    # --- section ---
    sec_score = _section_fit_score(blob, ctx.section, row=row)
    bd.section_fit = round(0.15 * sec_score, 4)

    # --- listing / page verification bonus ---
    listing = 0.0
    if tier.value in ("manufacturer", "operator", "verified_listing"):
        if ctx.tail and ctx.tail.lower() in blob:
            listing = 0.15
        elif ctx.model and ctx.model.lower() in blob:
            listing = 0.12
        elif page and any(x in page.lower() for x in ("controller.com", "jetphotos", "planespotters")):
            listing = 0.08
    bd.listing_verification = listing
    if listing > 0:
        notes.append("listing_or_oem_page_verified")

    bd.total = round(min(1.0, bd.identity + bd.source_trust + bd.section_fit + bd.listing_verification), 4)
    bd.notes = notes
    return bd
