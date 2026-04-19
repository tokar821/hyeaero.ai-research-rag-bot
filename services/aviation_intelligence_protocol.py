"""
Aviation intelligence protocol — structured envelope (not conversational).

Status codes:
  INVALID        — aircraft / request invalid for retrieval (e.g. non-existent model pattern)
  AMBIGUOUS      — tail resolves to conflicting canonical models in registry
  NO_VISUAL      — turn is not a visual retrieval request
  OK             — visual pipeline executed and at least one verified image row returned
  RETRIEVAL_FAILED — visual request valid but zero images passed validation / search
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.consultant_image_search_orchestrator import classify_premium_aviation_intent
from services.image_intelligence_engine import _optional_db, resolve_aircraft_identity
from services.searchapi_aircraft_images import normalize_aircraft_name


def _norm_gallery_image(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map consultant gallery row → intelligence block item (minimal strict fields)."""
    url = str(row.get("url") or "").strip()
    src = str(row.get("source") or "").strip() or "unknown"
    desc = str(row.get("description") or "").strip()
    page = str(row.get("page_url") or "").strip()
    out: Dict[str, Any] = {"url": url, "source": src}
    if desc:
        out["description"] = desc
    if page:
        out["page_url"] = page
    return out


def build_aviation_intelligence_envelope(
    *,
    user_query: str,
    user_wants_gallery: bool,
    phly_rows: Optional[List[Dict[str, Any]]],
    aircraft_images: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a strict, JSON-serializable envelope for this turn (no prose).

    Aircraft validation runs before status **OK** / **RETRIEVAL_FAILED** (tail ambiguity → **AMBIGUOUS**).
    """
    raw = (user_query or "").strip()
    intent = classify_premium_aviation_intent(
        raw,
        required_tail=None,
        required_marketing_type=None,
        phly_rows=phly_rows or [],
    )
    tail_key = str(intent.get("tail_number") or "").strip()

    blocks: List[Dict[str, Any]] = []
    reason_code = ""
    aircraft = ""
    image_pipeline_executed = False

    if not user_wants_gallery:
        blocks.append({"type": "status", "status": "NO_VISUAL", "reason_code": "not_visual_intent"})
        return {
            "status": "NO_VISUAL",
            "reason_code": "not_visual_intent",
            "aircraft": "",
            "tail_number": tail_key,
            "image_pipeline_executed": False,
            "blocks": blocks,
        }

    if intent.get("type") == "INVALID":
        blocks.append({"type": "status", "status": "INVALID", "reason_code": "invalid_aircraft_query"})
        return {
            "status": "INVALID",
            "reason_code": "invalid_aircraft_query",
            "aircraft": "",
            "tail_number": tail_key,
            "image_pipeline_executed": False,
            "blocks": blocks,
        }

    if intent.get("suppress_image_search"):
        blocks.append({"type": "status", "status": "NO_VISUAL", "reason_code": "image_search_suppressed"})
        return {
            "status": "NO_VISUAL",
            "reason_code": "image_search_suppressed",
            "aircraft": "",
            "tail_number": tail_key,
            "image_pipeline_executed": False,
            "blocks": blocks,
        }

    db = _optional_db()
    image_pipeline_executed = True

    if tail_key:
        canon, authoritative, reason = resolve_aircraft_identity(tail=tail_key, db=db)
        if reason == "ambiguous_model":
            blocks.append({"type": "status", "status": "AMBIGUOUS", "reason_code": "ambiguous_model"})
            return {
                "status": "AMBIGUOUS",
                "reason_code": "ambiguous_model",
                "aircraft": "",
                "tail_number": tail_key,
                "image_pipeline_executed": True,
                "blocks": blocks,
            }
        aircraft = canon if authoritative else ""
    else:
        aircraft = normalize_aircraft_name(str(intent.get("aircraft") or "").strip())

    imgs = [_norm_gallery_image(r) for r in (aircraft_images or []) if str(r.get("url") or "").strip()]
    if imgs:
        blocks.append({"type": "status", "status": "OK", "reason_code": ""})
        blocks.append({"type": "verified_images", "count": len(imgs), "images": imgs})
        return {
            "status": "OK",
            "reason_code": "",
            "aircraft": aircraft,
            "tail_number": tail_key,
            "image_pipeline_executed": True,
            "blocks": blocks,
        }

    blocks.append({"type": "status", "status": "RETRIEVAL_FAILED", "reason_code": "zero_verified_images"})
    blocks.append({"type": "verified_images", "count": 0, "images": []})
    return {
        "status": "RETRIEVAL_FAILED",
        "reason_code": "zero_verified_images",
        "aircraft": aircraft,
        "tail_number": tail_key,
        "image_pipeline_executed": True,
        "blocks": blocks,
    }
