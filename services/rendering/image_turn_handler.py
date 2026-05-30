"""
Image-turn handler — SearchAPI intelligence when configured; fail-closed prose otherwise.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def resolve_image_turn(
    query: str,
    *,
    tavily_payload: Optional[Dict[str, Any]] = None,
    phly_rows: Optional[List[Dict[str, Any]]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    data_used: Optional[Dict[str, Any]] = None,
    max_images: int = 6,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Return (answer prose, aircraft_images, patch metadata) for explicit image requests.
    """
    patch: Dict[str, Any] = {}
    images: List[Dict[str, Any]] = []

    try:
        from services.searchapi_aircraft_images import searchapi_aircraft_images_enabled

        if searchapi_aircraft_images_enabled():
            from services.image_intelligence_engine import run_aircraft_image_intelligence

            intel = run_aircraft_image_intelligence(query, max_images=max_images)
            patch["image_intelligence"] = {
                k: intel.get(k) for k in ("aircraft", "image_type", "insight") if k in intel
            }
            for row in intel.get("images") or []:
                if not isinstance(row, dict):
                    continue
                url = str(row.get("url") or "").strip()
                if not url:
                    continue
                images.append(
                    {
                        "url": url,
                        "source": row.get("source") or "searchapi",
                        "confidence": row.get("confidence"),
                        "label": intel.get("aircraft") or row.get("label"),
                        "tags": row.get("tags"),
                    }
                )
    except Exception:
        pass

    if not images:
        try:
            from services.consultant_aircraft_images import build_consultant_aircraft_images

            gallery_meta: Dict[str, Any] = {}
            images = build_consultant_aircraft_images(
                tavily_payload or {},
                phly_rows or [],
                user_query=query,
                history=history,
                gallery_meta_out=gallery_meta,
                max_gallery_images=max_images,
            )
            patch["consultant_gallery_meta"] = gallery_meta
        except Exception:
            images = []

    from services.rendering.image_prose import format_image_verification_response

    answer = format_image_verification_response(query, images, data_used=data_used)
    patch["image_turn_resolved"] = True
    patch["image_count"] = len(images)
    return answer, images, patch


__all__ = ["resolve_image_turn"]
