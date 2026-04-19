"""
HyeAero.AI **image ranking & filter** layer: deterministic text/URL heuristics over SearchAPI rows.

This is **not** CLIP (no vision embeddings here). ``semantic_match`` is a **text-relevance proxy**
from titles/URLs/snippets vs a synthetic intent line. Optional env disables the pass.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_HOUSE_HOTEL = re.compile(
    r"\b("
    r"house\b|home\b|airbnb|vrbo|zillow|realtor|hotel\b|resort\b|motel\b|"
    r"wood\s+cabin|log\s+cabin|cottage\b|bed\s+and\s+breakfast|\bbnb\b"
    r")\b",
    re.I,
)
_RENDER_CGI = re.compile(
    r"\b("
    r"cg[ii]\b|3d\s*render|rendering\b|cartoon|anime|minecraft|sims\b|"
    r"unreal\s*engine|blender\s*render|digital\s*art|vector\s*art|"
    r"illustration\s*only|concept\s*art"
    r")\b",
    re.I,
)
_WATERMARK_STOCK = re.compile(
    r"\b(shutterstock|gettyimages|istock|dreamstime|alamy|depositphotos|"
    r"watermark|stock\s*photo)\b",
    re.I,
)
_BLURRY = re.compile(r"\b(blurry|low\s*res|pixelated|thumbnail\s*only)\b", re.I)

_EXTERIOR_STRONG = re.compile(
    r"\b("
    r"ramp\b|airside|takeoff|landing\b|touchdown|air[- ]to[- ]air|"
    r"planespotting|spotter|jetphotos|rotate\b|departure|arrival\b|"
    r"gear\s*down|wheels?\s*down|taxiing|taxi\b"
    r")\b",
    re.I,
)
_COCKPIT_STRONG = re.compile(
    r"\b("
    r"cockpit|flight\s*deck|flightdeck|pfd\b|mfd\b|fms\b|"
    r"instrument\s*panel|glass\s*cockpit|throttle|yoke\b|pedestal"
    r")\b",
    re.I,
)
_CABIN_STRONG = re.compile(
    r"\b("
    r"cabin\b|interior\b|galley|divan|berth|lavatory|lav\b|"
    r"seating\s*layout|club\s*seating|inflight\s*cabin|vip\s*cabin"
    r")\b",
    re.I,
)


def searchapi_image_rank_filter_engine_enabled() -> bool:
    """
    Post-filters SearchAPI gallery rows. Default **off** (strict heuristics can empty short titles).

    Set ``SEARCHAPI_IMAGE_RANK_FILTER_ENGINE=1`` to enable broker-style re-ranking.
    """
    return (os.getenv("SEARCHAPI_IMAGE_RANK_FILTER_ENGINE") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _blob(row: Dict[str, Any]) -> str:
    return " ".join(
        str(row.get(k) or "")
        for k in ("url", "title", "source", "alt", "description", "_source_page", "page_url")
    ).lower()


def _hard_reject_reason(blob: str, section: str) -> Optional[str]:
    if _HOUSE_HOTEL.search(blob):
        return "house_hotel_residential"
    if _RENDER_CGI.search(blob):
        return "cgi_render_cartoon"
    sec = (section or "").strip().lower()
    if sec in ("cabin", "interior", "bedroom", "lavatory"):
        if _EXTERIOR_STRONG.search(blob) and not _CABIN_STRONG.search(blob):
            return "exterior_when_interior_requested"
        if _COCKPIT_STRONG.search(blob) and not _CABIN_STRONG.search(blob):
            return "cockpit_when_cabin_requested"
    if sec == "cockpit":
        if _CABIN_STRONG.search(blob) and not _COCKPIT_STRONG.search(blob):
            return "cabin_when_cockpit_requested"
        if _EXTERIOR_STRONG.search(blob) and not _COCKPIT_STRONG.search(blob):
            return "exterior_when_cockpit_requested"
    if sec == "exterior":
        if _CABIN_STRONG.search(blob) and not _EXTERIOR_STRONG.search(blob) and not _COCKPIT_STRONG.search(blob):
            # interior-only marketing page with no exterior cues
            if not re.search(r"\b(exterior|ramp|parked|taxi|flight|airport|sky)\b", blob, re.I):
                return "interior_when_exterior_requested"
    return None


def _aircraft_match_score(aircraft: str, blob: str) -> float:
    ac = (aircraft or "").strip()
    if not ac:
        return 0.45
    low = blob
    ac_l = ac.lower()
    if ac_l in low:
        return 1.0
    try:
        from services.consultant_aircraft_images import (
            _derive_model_positive_tokens,
            _model_positive_token_matches_blob,
        )

        pos = _derive_model_positive_tokens(ac)
        hits = [p for p in pos if len(p) >= 3 and _model_positive_token_matches_blob(blob, p)]
        if hits:
            return 0.98
    except Exception:
        pass
    # Family / series heuristics (same OEM line, adjacent numbers)
    if "challenger" in ac_l and "challenger" in low:
        nums = re.findall(r"\b(\d{3,4})\b", ac_l)
        nums_b = re.findall(r"\b(\d{3,4})\b", low)
        if nums and nums_b and nums[0] != nums_b[0]:
            return 0.78
        return 0.62
    if "citation" in ac_l and "citation" in low:
        return 0.72
    if "falcon" in ac_l and "falcon" in low:
        return 0.72
    if "gulfstream" in ac_l or re.search(r"\bg\d{3}", ac_l):
        if "gulfstream" in low or re.search(r"\bg[-\s]?\d{3}", low, re.I):
            return 0.72
    return 0.4


def _section_match_score(section: str, blob: str) -> float:
    sec = (section or "interior").strip().lower()
    if sec in ("interior",):
        sec = "cabin"
    if sec == "cabin":
        if _COCKPIT_STRONG.search(blob) and not _CABIN_STRONG.search(blob):
            return 0.0
        if _CABIN_STRONG.search(blob):
            return 1.0
        if _EXTERIOR_STRONG.search(blob):
            return 0.15
        return 0.45
    if sec == "cockpit":
        if _COCKPIT_STRONG.search(blob):
            return 1.0
        if _CABIN_STRONG.search(blob) and not _COCKPIT_STRONG.search(blob):
            return 0.0
        return 0.45
    if sec == "exterior":
        if _EXTERIOR_STRONG.search(blob) or re.search(r"\b(parked|ramp|sky|cloud)\b", blob, re.I):
            return 1.0
        if _CABIN_STRONG.search(blob) and not _EXTERIOR_STRONG.search(blob):
            return 0.1
        return 0.55
    if sec in ("bedroom", "lavatory"):
        if sec in blob:
            return 1.0
        if _CABIN_STRONG.search(blob):
            return 0.55
        return 0.2
    return 0.5


def _semantic_text_proxy(aircraft: str, section: str, blob: str) -> float:
    """Keyword overlap vs a synthetic intent line (CLIP-free)."""
    ref = f"{aircraft} {section} cabin interior cockpit".lower()
    ref_toks = {t for t in re.split(r"[^\w]+", ref) if len(t) >= 3}
    blob_toks = set(re.split(r"[^\w]+", blob))
    if not ref_toks:
        return 0.5
    hit = len(ref_toks & blob_toks) / max(6, len(ref_toks) * 0.35)
    return max(0.0, min(1.0, 0.35 + hit * 0.45))


def _quality_score(blob: str) -> float:
    q = 0.68
    if _WATERMARK_STOCK.search(blob):
        q -= 0.22
    if _BLURRY.search(blob):
        q -= 0.18
    if re.search(r"\b(full\s*cabin|wide\s*angle|layout|seating)\b", blob, re.I):
        q += 0.12
    if re.search(r"\b(modern|refurbish|new\s*interior)\b", blob, re.I):
        q += 0.06
    return max(0.15, min(1.0, q))


def _final_score(air: float, sec: float, sem: float, qual: float) -> float:
    return max(
        0.0,
        min(1.0, 0.4 * air + 0.3 * sec + 0.2 * sem + 0.1 * qual),
    )


def rank_and_filter_aviation_images(
    *,
    query_intent: Dict[str, Any],
    images: List[Dict[str, Any]],
    min_selected: int = 2,
    max_selected: int = 5,
    min_final_score: float = 0.52,
) -> Dict[str, Any]:
    """
    Rank/filter image dicts (url/title/source/alt). Returns JSON-shaped dict per product spec.

    ``images`` may include ``_gallery_item`` to preserve SearchAPI gallery row shape on output.
    """
    aircraft = str(query_intent.get("aircraft") or "").strip()
    section = str(query_intent.get("section") or query_intent.get("type") or "interior").strip()

    scored_rows: List[Tuple[float, str, Dict[str, Any]]] = []
    rejected = 0

    for im in images:
        b = _blob(im)
        hr = _hard_reject_reason(b, section)
        if hr:
            rejected += 1
            continue
        air = _aircraft_match_score(aircraft, b)
        if air < 0.5:
            rejected += 1
            continue
        sec = _section_match_score(section, b)
        if sec <= 0.0:
            rejected += 1
            continue
        sem = _semantic_text_proxy(aircraft, section, b)
        if sem < 0.42:
            rejected += 1
            continue
        qual = _quality_score(b)
        fs = _final_score(air, sec, sem, qual)
        if fs < min_final_score:
            rejected += 1
            continue
        reason_parts = [
            f"aircraft={air:.2f}",
            f"section={sec:.2f}",
            f"text={sem:.2f}",
            f"quality={qual:.2f}",
        ]
        scored_rows.append((fs, "; ".join(reason_parts), im))

    scored_rows.sort(key=lambda t: t[0], reverse=True)
    top = scored_rows[:max_selected]

    if len(top) < min_selected:
        conf = 0.35 if top else 0.2
        return {
            "selected_images": [],
            "rejected_count": len(images),
            "confidence": conf,
        }

    selected: List[Dict[str, Any]] = []
    for fs, reason, im in top:
        entry: Dict[str, Any] = {
            "url": str(im.get("url") or "").strip(),
            "score": round(fs, 4),
            "reason": reason[:220],
        }
        gi = im.get("_gallery_item")
        if isinstance(gi, dict):
            entry["_gallery_item"] = gi
        selected.append(entry)

    conf_out = round(min(s[0] for s in top) * 0.92 + 0.05, 3)
    conf_out = max(0.55, min(0.98, conf_out))

    return {
        "selected_images": selected,
        "rejected_count": rejected,
        "confidence": conf_out,
    }


def apply_rank_filter_to_gallery_items(
    *,
    gallery_items: List[Dict[str, Any]],
    query_intent: Dict[str, Any],
    max_out: int,
    gallery_meta: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Run :func:`rank_and_filter_aviation_images` on consultant gallery rows; restore ``_gallery_item`` shape.
    """
    if not searchapi_image_rank_filter_engine_enabled() or not gallery_items:
        return gallery_items

    images_in = []
    for it in gallery_items:
        images_in.append(
            {
                "url": it.get("url"),
                "title": it.get("description") or "",
                "source": it.get("source") or "",
                "alt": "",
                "page_url": it.get("page_url"),
                "_gallery_item": it,
            }
        )

    out = rank_and_filter_aviation_images(
        query_intent=query_intent,
        images=images_in,
        min_selected=2,
        max_selected=max(2, min(max_out, 5)),
        min_final_score=float((os.getenv("SEARCHAPI_IMAGE_RANK_MIN_SCORE") or "0.52").strip() or 0.52),
    )

    if gallery_meta is not None:
        gallery_meta["image_rank_filter_engine"] = {
            "rejected_count": out.get("rejected_count"),
            "confidence": out.get("confidence"),
            "selected_n": len(out.get("selected_images") or []),
        }

    sel = out.get("selected_images") or []
    if not sel:
        logger.info("IMAGE_RANK_FILTER: returning empty gallery (precision > quantity)")
        return []

    rebuilt: List[Dict[str, Any]] = []
    for s in sel:
        gi = s.get("_gallery_item")
        if isinstance(gi, dict) and (gi.get("url") or "").strip():
            rebuilt.append(gi)
    return rebuilt[:max_out]
