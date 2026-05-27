"""
Per-turn image context — prevent cross-turn gallery contamination (e.g. stale tail imagery).
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional


def advisory_image_context_patch(query: str) -> Dict[str, Any]:
    """
    Keys merged into ``data_used`` at the start of an advisory turn.

    Clears prior-turn gallery payloads so N807JS (or any tail) cannot leak into unrelated missions.
    """
    q = (query or "").strip()
    turn_id = hashlib.sha256(q.encode("utf-8")).hexdigest()[:16] if q else "empty"
    return {
        "consultant_aircraft_images": [],
        "aircraft_images": [],
        "consultant_gallery_meta": {},
        "consultant_image_context_turn_id": turn_id,
        "visual_memory_cleared": True,
        "gallery_tail_anchor": None,
        "gallery_model_anchor": None,
    }


def merge_fresh_gallery_patch(
    existing: Optional[Dict[str, Any]],
    images: List[Dict[str, Any]],
    *,
    meta: Optional[Dict[str, Any]] = None,
    tail_anchor: Optional[str] = None,
    model_anchor: Optional[str] = None,
) -> Dict[str, Any]:
    """Replace (not append) gallery slots for the active turn."""
    base = dict(existing or {})
    base["consultant_aircraft_images"] = list(images or [])
    base["aircraft_images"] = list(images or [])
    if meta is not None:
        base["consultant_gallery_meta"] = dict(meta)
    if tail_anchor:
        base["gallery_tail_anchor"] = tail_anchor
    if model_anchor:
        base["gallery_model_anchor"] = model_anchor
    return base


def gallery_allowed_for_query(query: str, *, intent: Optional[Dict[str, Any]] = None) -> bool:
    """Single gate for SearchAPI / Tavily gallery — advisory turns stay text-first."""
    from services.orchestration.image_trust_policy import should_activate_image_trust

    return should_activate_image_trust(query, intent=intent)
