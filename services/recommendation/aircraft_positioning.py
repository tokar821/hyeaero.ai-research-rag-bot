"""
Aircraft positioning hierarchy — broker-grade cabin / range tier understanding.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, Optional

from services.catalog.catalog_alias_resolver import resolve_canonical_display_name


class PositionTier(IntEnum):
    ENTRY_LIGHT = 1
    LIGHT_MID = 2
    SUPER_MID = 3
    LARGE_CABIN = 4
    UPPER_LARGE = 5
    FLAGSHIP_ULR = 6


_TIER_BY_MODEL: Dict[str, PositionTier] = {
    "Citation CJ2": PositionTier.ENTRY_LIGHT,
    "Citation CJ4": PositionTier.ENTRY_LIGHT,
    "Learjet 75": PositionTier.ENTRY_LIGHT,
    "Pilatus PC-24": PositionTier.ENTRY_LIGHT,
    "Pilatus PC-12": PositionTier.ENTRY_LIGHT,
    "Citation Latitude": PositionTier.LIGHT_MID,
    "Citation Longitude": PositionTier.SUPER_MID,
    "Praetor 600": PositionTier.SUPER_MID,
    "Gulfstream G280": PositionTier.SUPER_MID,
    "Challenger 350": PositionTier.SUPER_MID,
    "Challenger 650": PositionTier.LARGE_CABIN,
    "Falcon 2000": PositionTier.LARGE_CABIN,
    "Falcon 7X": PositionTier.UPPER_LARGE,
    "Falcon 8X": PositionTier.UPPER_LARGE,
    "Gulfstream G650": PositionTier.FLAGSHIP_ULR,
    "Gulfstream G650ER": PositionTier.FLAGSHIP_ULR,
    "Global 6500": PositionTier.FLAGSHIP_ULR,
    "Global 7500": PositionTier.FLAGSHIP_ULR,
}

_CATEGORY_FALLBACK: Dict[str, PositionTier] = {
    "light": PositionTier.ENTRY_LIGHT,
    "light jet": PositionTier.ENTRY_LIGHT,
    "midsize": PositionTier.LIGHT_MID,
    "super-midsize": PositionTier.SUPER_MID,
    "super midsize": PositionTier.SUPER_MID,
    "large-cabin": PositionTier.LARGE_CABIN,
    "large cabin": PositionTier.LARGE_CABIN,
    "ulr": PositionTier.FLAGSHIP_ULR,
    "ultra-long-range": PositionTier.FLAGSHIP_ULR,
}


def aircraft_position_tier(model: str, *, category: str = "") -> PositionTier:
    """Resolve broker positioning tier for a catalog model."""
    canonical = resolve_canonical_display_name(model) or model
    if canonical in _TIER_BY_MODEL:
        return _TIER_BY_MODEL[canonical]
    low = canonical.lower()
    for key, tier in _TIER_BY_MODEL.items():
        if key.lower() in low or low in key.lower():
            return tier
    cat = (category or "").strip().lower()
    return _CATEGORY_FALLBACK.get(cat, PositionTier.SUPER_MID)


def tier_distance(from_tier: PositionTier, to_tier: PositionTier) -> int:
    """Absolute tier steps — used for replacement realism."""
    return abs(int(from_tier) - int(to_tier))


def is_prestige_collapse(from_model: str, to_model: str, *, category: str = "") -> bool:
    """True when replacement drops more than one major broker tier inappropriately."""
    src = aircraft_position_tier(from_model)
    dst = aircraft_position_tier(to_model, category=category)
    return tier_distance(src, dst) >= 3


__all__ = [
    "PositionTier",
    "aircraft_position_tier",
    "tier_distance",
    "is_prestige_collapse",
]
