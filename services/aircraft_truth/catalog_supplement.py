"""
Verified operational supplements for catalog aircraft.

Brochure NM and scored indices are not used for truth validation — only fields
in this supplement plus core catalog keys (practical_nm, pax_max, category).
"""

from __future__ import annotations

from typing import Any, Dict

# runway_class: short_field | regional | super_mid | large_cabin | ultra_long
# baggage_volume_cu_ft: typical usable baggage volume (internal catalog baseline)
CATALOG_TRUTH_SUPPLEMENT: Dict[str, Dict[str, Any]] = {
    "Citation CJ2": {
        "runway_class": "short_field",
        "baggage_volume_cu_ft": 53,
        "truth_verified": True,
    },
    "Citation CJ4": {
        "runway_class": "short_field",
        "baggage_volume_cu_ft": 58,
        "truth_verified": True,
    },
    "Citation Latitude": {
        "runway_class": "super_mid",
        "baggage_volume_cu_ft": 100,
        "truth_verified": True,
    },
    "Praetor 600": {
        "runway_class": "super_mid",
        "baggage_volume_cu_ft": 95,
        "truth_verified": True,
    },
    "Challenger 350": {
        "runway_class": "super_mid",
        "baggage_volume_cu_ft": 106,
        "truth_verified": True,
    },
    "Challenger 650": {
        "runway_class": "large_cabin",
        "baggage_volume_cu_ft": 125,
        "truth_verified": True,
    },
    "Challenger Longitude": {
        "runway_class": "large_cabin",
        "baggage_volume_cu_ft": 130,
        "truth_verified": True,
    },
    "Gulfstream G280": {
        "runway_class": "super_mid",
        "baggage_volume_cu_ft": 120,
        "truth_verified": True,
    },
    "Falcon 2000": {
        "runway_class": "large_cabin",
        "baggage_volume_cu_ft": 115,
        "truth_verified": True,
    },
    "Falcon 7X": {
        "runway_class": "large_cabin",
        "baggage_volume_cu_ft": 140,
        "truth_verified": True,
    },
    "Falcon 8X": {
        "runway_class": "ultra_long",
        "baggage_volume_cu_ft": 150,
        "truth_verified": True,
    },
    "Gulfstream G650ER": {
        "runway_class": "ultra_long",
        "baggage_volume_cu_ft": 195,
        "truth_verified": True,
    },
    "Gulfstream G650": {
        "runway_class": "ultra_long",
        "baggage_volume_cu_ft": 195,
        "truth_verified": True,
    },
    "Global 6500": {
        "runway_class": "ultra_long",
        "baggage_volume_cu_ft": 195,
        "truth_verified": True,
    },
    "Global 7500": {
        "runway_class": "ultra_long",
        "baggage_volume_cu_ft": 210,
        "truth_verified": True,
    },
    "Learjet 75": {
        "runway_class": "regional",
        "baggage_volume_cu_ft": 65,
        "truth_verified": True,
    },
    "Pilatus PC-24": {
        "runway_class": "short_field",
        "baggage_volume_cu_ft": 90,
        "truth_verified": True,
    },
    "Pilatus PC-12": {
        "runway_class": "short_field",
        "baggage_volume_cu_ft": 40,
        "truth_verified": True,
    },
}
