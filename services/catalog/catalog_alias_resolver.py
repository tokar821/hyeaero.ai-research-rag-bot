"""
Catalog alias resolver — normalize common broker tokens before comparison, capability, and ranking.

Maps marketing names and spoken aliases to canonical display strings and, when needed,
to the nearest in-catalog profile key for physics evaluation.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

from services.mission.aircraft_profiles import AIRCRAFT_PROFILES

# User-facing canonical display names (stable broker vocabulary).
_DISPLAY_ALIASES: Dict[str, str] = {
    "challenger 3500": "Challenger 350",
    "cl3500": "Challenger 350",
    "g500": "Gulfstream G500",
    "gulfstream g500": "Gulfstream G500",
    "g600": "Gulfstream G600",
    "gulfstream g600": "Gulfstream G600",
    "global 5500": "Global 5500",
    "bombardier global 5500": "Global 5500",
    "global 6000": "Bombardier Global 6000",
    "bombardier global 6000": "Bombardier Global 6000",
    "global 6500": "Global 6500",
    "bombardier global 6500": "Global 6500",
    "falcon eight x": "Falcon 8X",
    "falcon 7x": "Falcon 7X",
    "dassault falcon 7x": "Falcon 7X",
    "global seven five zero zero": "Global 7500",
    "g650er": "Gulfstream G650ER",
    "g650": "Gulfstream G650",
    "g550": "Gulfstream G550",
    "gulfstream g550": "Gulfstream G550",
    "citation longitude": "Citation Longitude",
    "cessna citation longitude": "Citation Longitude",
    "legacy 600": "Legacy 600",
    "embraer legacy 600": "Legacy 600",
    "legacy 650": "Legacy 600",
}

def _normalize_key(raw: str) -> str:
    spoken = re.sub(r"[^\w\s]", " ", (raw or "").lower())
    return re.sub(r"\s+", " ", spoken).strip()


def resolve_canonical_display_name(raw: str) -> str:
    """Return broker-canonical display name for a raw token."""
    token = (raw or "").strip()
    if not token:
        return ""
    key = _normalize_key(token)
    if key in _DISPLAY_ALIASES:
        return _DISPLAY_ALIASES[key]
    if token in AIRCRAFT_PROFILES:
        return token
    # Manufacturer prefix strip
    key2 = re.sub(r"^(?:gulfstream|bombardier|embraer|dassault|cessna|textron)\s+", "", key)
    if key2 in _DISPLAY_ALIASES:
        return _DISPLAY_ALIASES[key2]
    return token


def resolve_catalog_profile_key(raw: str) -> Optional[str]:
    """
    Resolve a model token to a verified catalog key for physics evaluation.

    No cross-model substitution (e.g. G500 is never mapped to G650).
    Returns None when specifications cannot be verified.
    """
    display = resolve_canonical_display_name(raw)
    if not display:
        return None
    try:
        from services.data_authority.aircraft_spec_repository import get_verified_spec

        spec = get_verified_spec(display)
        if spec is not None:
            return spec.canonical_name
    except Exception:
        pass
    if display in AIRCRAFT_PROFILES:
        return display
    low = display.lower()
    for name in AIRCRAFT_PROFILES:
        if name.lower() == low:
            return name
    return None


__all__ = [
    "resolve_canonical_display_name",
    "resolve_catalog_profile_key",
]
