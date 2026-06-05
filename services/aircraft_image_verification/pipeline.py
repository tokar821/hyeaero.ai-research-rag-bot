"""
Strict aircraft image verification pipeline.

Only verified images (confidence ≥ threshold) with exact tail OR exact model match
are returned to the user-facing gallery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from services.aircraft_image_verification.confidence import (
    ImageConfidenceBreakdown,
    score_image_confidence,
)
from services.aircraft_image_verification.constants import (
    VERIFIED_FAILURE_MESSAGE,
    verification_min_confidence,
)
from services.aircraft_image_verification.rejection import (
    ImageVerificationContext,
    evaluate_rejection,
)
from services.aircraft_image_verification.source_ranking import SourceTier, classify_source_tier

logger = logging.getLogger(__name__)


@dataclass
class RejectionRecord:
    url: str
    reason: str
    stage: str = "verification"

    def to_dict(self) -> Dict[str, Any]:
        return {"url": self.url, "reason": self.reason, "stage": self.stage}


@dataclass
class ImageVerificationAudit:
    tail: Optional[str] = None
    model: Optional[str] = None
    section: str = "exterior"
    min_confidence: float = 0.7
    input_count: int = 0
    verified_count: int = 0
    rejected: List[RejectionRecord] = field(default_factory=list)
    scoring: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pipeline_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tail": self.tail,
            "model": self.model,
            "section": self.section,
            "min_confidence": self.min_confidence,
            "input_count": self.input_count,
            "verified_count": self.verified_count,
            "rejected": [r.to_dict() for r in self.rejected],
            "scoring": dict(self.scoring),
            "pipeline_confidence": round(self.pipeline_confidence, 4),
        }


@dataclass
class VerificationPipelineResult:
    images: List[Dict[str, Any]]
    audit: ImageVerificationAudit
    empty: bool = False
    message: str = ""

    def to_meta(self) -> Dict[str, Any]:
        return {
            "aircraft_image_verification": self.audit.to_dict(),
            "consultant_gallery_empty": self.empty,
            "consultant_gallery_message": self.message if self.empty else "",
        }


def _row_url(row: Dict[str, Any]) -> str:
    return str(row.get("url") or row.get("imageUrl") or row.get("image") or "").strip()


def verify_aircraft_image_rows(
    rows: List[Dict[str, Any]],
    *,
    tail: Optional[str] = None,
    model: Optional[str] = None,
    section: str = "exterior",
    min_confidence: Optional[float] = None,
    max_out: int = 5,
) -> VerificationPipelineResult:
    """
    Verify SearchAPI-style merged rows before gallery assembly.

    Requires ``tail`` or ``model`` — generic unanchored searches return empty.
    """
    threshold = verification_min_confidence() if min_confidence is None else float(min_confidence)
    ctx = ImageVerificationContext(
        tail=(tail or "").strip() or None,
        model=(model or "").strip() or None,
        section=(section or "exterior").strip() or "exterior",
    )

    audit = ImageVerificationAudit(
        tail=ctx.tail,
        model=ctx.model,
        section=ctx.section,
        min_confidence=threshold,
        input_count=len(rows or []),
    )

    if not ctx.tail and not ctx.model:
        audit.rejected.append(
            RejectionRecord("", "missing_aircraft_identity", stage="preflight")
        )
        return VerificationPipelineResult(
            images=[],
            audit=audit,
            empty=True,
            message=VERIFIED_FAILURE_MESSAGE,
        )

    scored: List[Tuple[float, Dict[str, Any], ImageConfidenceBreakdown]] = []

    for row in rows or []:
        url = _row_url(row)
        reason = evaluate_rejection(row, ctx)
        if reason:
            audit.rejected.append(RejectionRecord(url, reason))
            continue

        bd = score_image_confidence(row, ctx)
        audit.scoring[url or f"row_{len(audit.scoring)}"] = bd.to_dict()

        effective = bd.total
        tail_conf = str(
            row.get("tail_match_confidence") or row.get("_tail_confidence") or ""
        ).strip().lower()
        if ctx.tail and tail_conf == "confirmed":
            effective = max(effective, threshold)
        elif bd.match_type in ("model_exact", "tail_exact") and effective >= 0.64:
            effective = max(effective, threshold)

        if effective < threshold:
            audit.rejected.append(
                RejectionRecord(url, f"confidence_below_threshold:{effective:.3f}")
            )
            continue

        out_row = dict(row)
        out_row["_verification_confidence"] = effective
        out_row["_verification_match_type"] = bd.match_type
        out_row["_verification_breakdown"] = bd.to_dict()
        tier, _ = classify_source_tier(
            url=url,
            page_url=str(row.get("_source_page") or row.get("page_url") or ""),
            source_label=str(row.get("source") or ""),
            title=str(row.get("title") or row.get("description") or ""),
        )
        out_row["_verification_source_tier"] = tier.value
        scored.append((effective, out_row, bd))

    scored.sort(key=lambda t: (t[0], t[2].source_trust), reverse=True)
    verified = [r for _, r, _ in scored[: max(1, max_out)]]

    audit.verified_count = len(verified)
    audit.pipeline_confidence = scored[0][0] if scored else 0.0

    if not verified:
        logger.info(
            "IMAGE_VERIFICATION_EMPTY tail=%s model=%s in=%d rejected=%d",
            ctx.tail,
            ctx.model,
            audit.input_count,
            len(audit.rejected),
        )
        return VerificationPipelineResult(
            images=[],
            audit=audit,
            empty=True,
            message=VERIFIED_FAILURE_MESSAGE,
        )

    return VerificationPipelineResult(
        images=verified,
        audit=audit,
        empty=False,
        message="",
    )


def verify_gallery_images(
    gallery_items: List[Dict[str, Any]],
    *,
    tail: Optional[str] = None,
    model: Optional[str] = None,
    section: str = "exterior",
    min_confidence: Optional[float] = None,
    max_out: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Verify consultant gallery items (``url``, ``description``, ``page_url``, …).

    Returns ``(verified_gallery_items, meta_dict)``.
    """
    rows_in: List[Dict[str, Any]] = []
    for it in gallery_items or []:
        rows_in.append(
            {
                "url": it.get("url"),
                "title": it.get("description") or "",
                "source": it.get("source") or "",
                "_source_page": it.get("page_url") or "",
                "page_url": it.get("page_url"),
                "tail_match_confidence": it.get("tail_match_confidence"),
                "_gallery_item": it,
            }
        )

    result = verify_aircraft_image_rows(
        rows_in,
        tail=tail,
        model=model,
        section=section,
        min_confidence=min_confidence,
        max_out=max_out,
    )

    out: List[Dict[str, Any]] = []
    for row in result.images:
        gi = row.get("_gallery_item")
        if isinstance(gi, dict):
            item = dict(gi)
        else:
            item = {
                "url": row.get("url"),
                "source": row.get("source"),
                "description": row.get("title") or row.get("description"),
                "page_url": row.get("page_url") or row.get("_source_page"),
            }
        item["verification_confidence"] = row.get("_verification_confidence")
        item["verification_match_type"] = row.get("_verification_match_type")
        item["verification_source_tier"] = row.get("_verification_source_tier")
        out.append(item)

    meta = result.to_meta()
    if result.empty:
        meta["consultant_gallery_suggestions"] = [
            "Confirm tail number or aircraft model spelling",
            "Try exterior ramp photos from a listing URL",
        ]
    return out, meta


def fallback_handling(
    *,
    had_candidates: bool,
    verification_result: VerificationPipelineResult,
) -> Dict[str, Any]:
    """
    Standard empty-state payload when verification yields no images.
    """
    if verification_result.images:
        return {
            "success": True,
            "images": verification_result.images,
            "confidence": verification_result.audit.pipeline_confidence,
            "verification_audit": verification_result.audit.to_dict(),
        }
    return {
        "success": False,
        "message": VERIFIED_FAILURE_MESSAGE,
        "images": [],
        "confidence": 0.0,
        "had_candidates": had_candidates,
        "verification_audit": verification_result.audit.to_dict(),
    }
