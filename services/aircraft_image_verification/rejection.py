"""
Hard rejection rules — images never enter the verified gallery when rejected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from services.aircraft_image_verification.source_ranking import SourceTier, classify_source_tier


@dataclass(frozen=True)
class ImageVerificationContext:
    """Target aircraft identity and requested visual facet."""

    tail: Optional[str] = None
    model: Optional[str] = None
    section: str = "exterior"  # exterior | cabin | cockpit | interior


_GENERIC_INTERIOR = re.compile(
    r"\b("
    r"interior\s+design|furniture\b|living\s+room|kitchen\s+remodel|"
    r"home\s+decor|bedroom\s+design|bathroom\s+remodel|sofa\b|sectional\b"
    r")\b",
    re.I,
)
_RESIDENTIAL_CABIN = re.compile(
    r"\b("
    r"log\s+cabin|wood\s+cabin|cabin\s+rental|vacation\s+rental|"
    r"airbnb|vrbo|gatlinburg|smoky\s+mountain|broken\s+bow|"
    r"house\b|hotel\b|resort\b|zillow|realtor"
    r")\b",
    re.I,
)
_CGI_RENDER = re.compile(
    r"\b(cg[ii]\b|3d\s*render|unreal\s*engine|blender\s*render|"
    r"concept\s*art|digital\s*art|minecraft|anime)\b",
    re.I,
)
# Spotter / ramp cues (exclude bare host names like jetphotos.com from exterior-only triggers)
_EXTERIOR_STRONG = re.compile(
    r"\b("
    r"ramp\b|airside|takeoff|landing\b|touchdown|air[- ]to[- ]air|"
    r"planespotting|spotter|rotate\b|departure|arrival\b|"
    r"gear\s*down|wheels?\s*down|taxiing|taxi\b|parked|walkaround"
    r")\b",
    re.I,
)
_COCKPIT_STRONG = re.compile(
    r"\b(cockpit|flight\s*deck|flightdeck|pfd\b|mfd\b|glass\s*cockpit)\b",
    re.I,
)
_CABIN_STRONG = re.compile(
    r"\b(aircraft\s+cabin|bizjet\s+cabin|jet\s+cabin|vip\s+cabin|"
    r"galley|divan|berth|lavatory|club\s+seating)\b",
    re.I,
)
_GENERIC_CABIN_ONLY = re.compile(
    r"\b(cabin\s+interior|luxury\s+cabin|private\s+cabin|jet\s+interior)\b",
    re.I,
)
_AIRLINER_MARKERS = re.compile(
    r"\b(md-?\s*8\d|boeing\s*7\d|airbus\s*a\d|american\s+airlines|mcdonnell\s+douglas)\b",
    re.I,
)
_PISTON_MARKERS = re.compile(
    r"\b(pa-28|cherokee\s+(?:140|six)|piper\s+pa-|cessna\s+172|beechcraft\s+bonanza)\b",
    re.I,
)
_BIZJET_MODEL_HINT = re.compile(
    r"\b(gulfstream|global\s*\d|challenger|citation|falcon|phenom|learjet|praetor|legacy|eclipse\s*500|g\d{3})\b",
    re.I,
)


def _blob(row: Dict[str, Any]) -> str:
    return " ".join(
        str(row.get(k) or "")
        for k in (
            "url",
            "title",
            "source",
            "description",
            "alt",
            "_source_page",
            "page_url",
            "snippet",
            "link",
        )
    ).lower()


def _section_mismatch(blob: str, section: str) -> Optional[str]:
    sec = (section or "exterior").strip().lower()
    if sec in ("interior",):
        sec = "cabin"
    spotter_host = bool(re.search(r"jetphotos|planespotters|airliners\.net", blob, re.I))
    if sec in ("cabin", "bedroom", "lavatory"):
        if _COCKPIT_STRONG.search(blob) and not _CABIN_STRONG.search(blob):
            return "cockpit_when_cabin_requested"
        if _EXTERIOR_STRONG.search(blob) and not _CABIN_STRONG.search(blob):
            # Spotter URLs often say "jetphotos" without cabin keywords — allow if title is not exterior-only.
            if spotter_host and not re.search(
                r"\b(ramp|takeoff|landing|taxi|airborne|walkaround|exterior|parked)\b",
                blob,
                re.I,
            ):
                return None
            if not spotter_host:
                return "exterior_when_cabin_requested"
    if sec == "cockpit":
        if _CABIN_STRONG.search(blob) and not _COCKPIT_STRONG.search(blob):
            return "cabin_when_cockpit_requested"
        if _EXTERIOR_STRONG.search(blob) and not _COCKPIT_STRONG.search(blob):
            return "exterior_when_cockpit_requested"
    if sec == "exterior":
        if _CABIN_STRONG.search(blob) and not _EXTERIOR_STRONG.search(blob):
            if not re.search(r"\b(exterior|ramp|parked|taxi|flight|airport)\b", blob, re.I):
                return "interior_when_exterior_requested"
    return None


def evaluate_rejection(
    row: Dict[str, Any],
    ctx: ImageVerificationContext,
) -> Optional[str]:
    """
    Return a rejection reason code, or ``None`` if the row may proceed to scoring.
    """
    blob = _blob(row)
    url = str(row.get("url") or row.get("image") or row.get("imageUrl") or "")
    page = str(row.get("_source_page") or row.get("page_url") or row.get("link") or "")
    title = str(row.get("title") or row.get("description") or "")

    tier, _ = classify_source_tier(url=url, page_url=page, source_label=str(row.get("source") or ""), title=title)
    if tier == SourceTier.STOCK_UNVERIFIED:
        return "stock_imagery_unverified"

    if _GENERIC_INTERIOR.search(blob) or _RESIDENTIAL_CABIN.search(blob):
        return "generic_or_residential_interior"

    if _CGI_RENDER.search(blob):
        return "cgi_or_render_not_verified"

    # Generic cabin marketing with no aircraft identity in metadata
    if _GENERIC_CABIN_ONLY.search(blob) and not ctx.tail and not ctx.model:
        return "generic_cabin_no_aircraft_anchor"

    if ctx.tail or ctx.model:
        identity = _identity_rejection(blob, ctx)
        if identity:
            return identity
        class_rej = _airframe_class_mismatch(blob, ctx)
        if class_rej:
            return class_rej
    else:
        return "missing_aircraft_identity"

    sec_rej = _section_mismatch(blob, ctx.section)
    if sec_rej:
        return sec_rej

    # Unrelated cabin: cabin keywords but wrong OEM family when model specified
    if ctx.model and _CABIN_STRONG.search(blob) and not _aircraft_context_for_model(blob, ctx.model):
        if not ctx.tail:
            return "unrelated_cabin_wrong_aircraft"

    return None


def _aircraft_context_for_model(blob: str, model: str) -> bool:
    try:
        from services.consultant_aircraft_images import (
            _derive_model_positive_tokens,
            _model_positive_token_matches_blob,
        )

        pos = _derive_model_positive_tokens(model)
        return any(_model_positive_token_matches_blob(blob, p) for p in pos if len(p) >= 3)
    except Exception:
        ml = model.lower()
        return ml in blob or any(p.lower() in blob for p in model.split() if len(p) >= 4)


def _airframe_class_mismatch(blob: str, ctx: ImageVerificationContext) -> Optional[str]:
    """Reject airliner/piston spotter hits when the anchored aircraft is a business jet."""
    anchor = (ctx.model or "").strip()
    if not anchor and ctx.tail:
        return None
    if anchor and not _BIZJET_MODEL_HINT.search(anchor):
        return None
    if _AIRLINER_MARKERS.search(blob) or _PISTON_MARKERS.search(blob):
        return "airframe_class_mismatch"
    return None


def _identity_rejection(blob: str, ctx: ImageVerificationContext) -> Optional[str]:
    """Require exact tail OR exact model match — no unanchored generics."""
    from rag.aviation_tail import normalize_tail_token
    from services.searchapi_aircraft_images import strip_domains

    if ctx.tail:
        tail = normalize_tail_token(ctx.tail)
        bag_u = strip_domains(blob).upper()
        if tail and tail not in bag_u and tail.lower() not in blob:
            return "tail_number_not_verified"
        # Conflicting registration on same result
        import re

        for m in re.finditer(r"\bN[1-9A-Z][A-Z0-9]{1,5}\b", bag_u):
            if normalize_tail_token(m.group(0)) != tail:
                return "conflicting_tail_on_image"
        return None

    model = (ctx.model or "").strip()
    if not model:
        return "missing_aircraft_identity"

    try:
        from services.consultant_aircraft_images import (
            _derive_model_negative_tokens,
            _model_tokens_match_strict,
        )

        neg = _derive_model_negative_tokens(model)
        if neg and any(n.lower() in blob for n in neg):
            return "wrong_aircraft_variant"
        if not _model_tokens_match_strict(blob, model):
            return "exact_model_match_required"
    except Exception:
        parts = [p for p in re.split(r"\s+", model) if len(p) >= 3]
        if parts and sum(1 for p in parts if p.lower() in blob) < max(1, len(parts) // 2):
            return "exact_model_match_required"

    return None
