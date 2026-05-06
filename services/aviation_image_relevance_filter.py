"""
Aviation Image Relevance Filter — **metadata-only** (title / URL / snippet text).

True fuselage or oval-window verification needs **vision**; this module applies the same
**strict** policy using observable text: **if uncertain → reject** (conservative for brokers).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

# --- Hard reject (non-aviation or hostile sources) ---
_HOUSE_HOTEL_BUILDING = re.compile(
    r"\b("
    r"house\b|villa\b|mansion\b|townhouse\b|condo\b|apartment\b|loft\b|"
    r"hotel\b|resort\b|motel\b|airbnb|vrbo|zillow|realtor|"
    r"building\s+interior|office\s+lobby|conference\s+room|ballroom\b|"
    r"showroom\b|furniture\s+showroom|kitchen\s+remodel|bathroom\s+remodel|"
    r"living\s+room\s+design|interior\s+design\b|home\s+decor"
    r")\b",
    re.I,
)
_POOL_SPA_LIFESTYLE = re.compile(
    r"\b("
    r"swimming\s+pool|pool\s+deck|infinity\s+pool|spa\s+day|hot\s+tub\b|jacuzzi\b|"
    r"wedding\s+venue|bridal\b|honeymoon\s+suite"
    r")\b",
    re.I,
)
_SOCIAL_MEME = re.compile(
    r"\b("
    r"reddit\b|redd\.it|imgflip|9gag|me\.me|ifunny|meme\b|dank\b|"
    r"blogspot|tumblr\.com|wordpress\.com|medium\.com/p/"
    r")\b",
    re.I,
)
_LOW_SIGNAL_MEDIA = re.compile(
    r"\b("
    r"low\s*resolution|low\s*res|pixelated|thumbnail\s*only|tiny\s*icon|"
    r"16x16|32x32|favicon|sprite\s*sheet|emoji"
    r")\b",
    re.I,
)

# --- Strong aviation accept cues (text proxy) ---
_US_N = re.compile(r"\bn[1-9][a-z0-9]{1,5}\b", re.I)
_OEM_OR_FAMILY = re.compile(
    r"\b("
    r"gulfstream|bombardier|dassault|embraer|cessna|beechcraft|learjet|"
    r"challenger|falcon|citation|global\s*\d|phenom|praetor|longitude|latitude|"
    r"hawker|pilatus|pc-12|boeing\s*business|airbus\s*acj|"
    r"bizjet|business\s*jet|private\s*jet|corporate\s*jet"
    r")\b",
    re.I,
)
_AIRCRAFT_LAYOUT = re.compile(
    r"\b("
    r"fuselage|flight\s*deck|cockpit|galley|lavatory|divan|club\s*seating|"
    r"vip\s*cabin|narrow\s*cabin|oval\s*windows?|cabin\s*windows?|"
    r"jet\s*cabin|aircraft\s*cabin|inflight\s*cabin|cabin\s+interior|interior\s+cabin|main\s+cabin|"
    r"planespotting|jetphotos|airliners\.net|abpic|flightaware"
    r")\b",
    re.I,
)
_EXTERIOR_OPS = re.compile(
    r"\b("
    r"ramp\b|taxiing|taxi\b|takeoff|landing\b|touchdown|rotate\b|"
    r"parked\s+at|airside|apron\b|hangar\s+(door|open)|gear\s+down"
    r")\b",
    re.I,
)

_ACCEPT_THRESHOLD = 0.68


def _blob(image: Dict[str, Any]) -> str:
    parts = []
    for k in ("url", "title", "source", "alt", "description", "_source_page", "page_url", "link"):
        parts.append(str(image.get(k) or ""))
    return " ".join(parts).strip().lower()


def _hard_reject_reason(blob: str) -> Optional[str]:
    if not blob or len(blob) < 6:
        return "empty_or_trivial_metadata"
    if _HOUSE_HOTEL_BUILDING.search(blob):
        return "house_hotel_or_building_interior_context"
    if _POOL_SPA_LIFESTYLE.search(blob):
        return "pool_spa_or_non_aviation_venue"
    if _SOCIAL_MEME.search(blob):
        return "reddit_meme_or_social_host"
    if _LOW_SIGNAL_MEDIA.search(blob):
        return "low_resolution_or_thumbnail_language"
    return None


def _relevance_score(blob: str) -> float:
    """0–1 score for aviation relevance from text cues only."""
    s = 0.0
    if _US_N.search(blob):
        s += 0.42
    if _OEM_OR_FAMILY.search(blob):
        s += 0.28
    if _AIRCRAFT_LAYOUT.search(blob):
        s += 0.34
    if _EXTERIOR_OPS.search(blob):
        s += 0.22
    # Known aviation photo hosts (still need some OEM/airframe hint)
    if re.search(r"\b(jetphotos\.net|planespotters\.net|airliners\.net)\b", blob, re.I):
        s += 0.18
    # Penalize generic lone "interior" / "room" without jet/aircraft
    if re.search(r"\b(master\s+bedroom|living\s+room|dining\s+room)\b", blob, re.I) and not _OEM_OR_FAMILY.search(
        blob
    ):
        s -= 0.45
    return max(0.0, min(1.0, s))


def evaluate_aviation_image_relevance(
    image: Dict[str, Any],
    *,
    accept_threshold: float = _ACCEPT_THRESHOLD,
) -> Dict[str, Any]:
    """
    Strict broker-style gate for one image row (SearchAPI / Tavily shape).

    Returns ``{"accepted": bool, "confidence": float, "reason": str}``.
    **If uncertain → reject** (``accepted`` False, moderate ``confidence`` as relevance score).
    """
    blob = _blob(image)
    rej = _hard_reject_reason(blob)
    if rej:
        return {
            "accepted": False,
            "confidence": round(0.08, 3),
            "reason": rej,
        }

    score = _relevance_score(blob)
    # Require at least two independent signal families unless tail number present
    has_tail = bool(_US_N.search(blob))
    has_oem = bool(_OEM_OR_FAMILY.search(blob))
    has_layout = bool(_AIRCRAFT_LAYOUT.search(blob))
    has_ext = bool(_EXTERIOR_OPS.search(blob))
    strong_host = bool(re.search(r"\b(jetphotos\.net|planespotters\.net)\b", blob, re.I))

    families = sum(1 for x in (has_tail, has_oem, has_layout, has_ext, strong_host) if x)
    if not has_tail and families < 2:
        score = min(score, 0.52)
    if not has_tail and not has_oem and not has_layout:
        score = min(score, 0.38)

    # OEM + airframe-interior vocabulary is a strong text proxy for bizjet photography.
    if has_oem and has_layout:
        score = max(score, 0.74)

    # U.S. tail on known planespotter-style host is strong proxy for real airframe photography.
    if has_tail and (strong_host or has_ext or has_layout):
        score = max(score, 0.78)

    accepted = score >= float(accept_threshold)
    if accepted:
        reason = "aviation_text_cues_met_strict_gate"
        if has_tail:
            reason = "registration_visible_in_metadata"
        elif has_layout and has_oem:
            reason = "oem_plus_cabin_or_airframe_cues"
    else:
        reason = "uncertain_or_insufficient_aviation_evidence_reject"

    return {
        "accepted": bool(accepted),
        "confidence": round(float(score), 3),
        "reason": reason,
    }


def filter_aviation_images_by_relevance(
    images: list,
    *,
    accept_threshold: float = _ACCEPT_THRESHOLD,
) -> list:
    """Keep only rows where :func:`evaluate_aviation_image_relevance` accepts."""
    out = []
    for im in images or []:
        if not isinstance(im, dict):
            continue
        ev = evaluate_aviation_image_relevance(im, accept_threshold=accept_threshold)
        if ev.get("accepted"):
            row = dict(im)
            row["relevance_filter"] = ev
            out.append(row)
    return out
