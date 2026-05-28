"""
Ontology normalization for aircraft model identifiers.

Purpose:
- Normalize aliases to stable identity keys for comparisons.
- Prevent duplicate rows (e.g., "G280" vs "Gulfstream G280").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class NormalizedAircraft:
    identity_key: str
    display_name: str
    family: str


_MODEL_NORMALIZATION: Dict[str, Tuple[str, str, str]] = {
    # Challenger family
    "challenger 3500": ("challenger_350_family", "Challenger 3500", "Challenger 350 family"),
    "challenger 350": ("challenger_350_family", "Challenger 350", "Challenger 350 family"),
    "cl3500": ("challenger_350_family", "Challenger 3500", "Challenger 350 family"),
    "cl350": ("challenger_350_family", "Challenger 350", "Challenger 350 family"),
    "challenger 650": ("challenger_600_family", "Challenger 650", "Challenger 600/650 family"),
    "legacy 600": ("legacy_600_family", "Legacy 600", "Legacy 600/650 family"),
    "legacy 650": ("legacy_600_family", "Legacy 650", "Legacy 600/650 family"),
    # Global family
    "global 7500": ("global_ulr_family", "Global 7500", "Global ultra-long-range family"),
    "global 6500": ("global_ulr_family", "Global 6500", "Global ultra-long-range family"),
    "global 8000": ("global_ulr_family", "Global 8000", "Global ultra-long-range family"),
    # Falcon family
    "falcon 8x": ("falcon_trijet_heavy", "Falcon 8X", "Falcon tri-jet heavy class"),
    "falcon 7x": ("falcon_trijet_heavy", "Falcon 7X", "Falcon tri-jet heavy class"),
    "falcon 10x": ("falcon_trijet_heavy", "Falcon 10X", "Falcon tri-jet heavy class"),
    # Gulfstream family
    "gulfstream g650er": ("g650_family", "Gulfstream G650ER", "G650 family"),
    "g650er": ("g650_family", "Gulfstream G650ER", "G650 family"),
    "gulfstream g650": ("g650_family", "Gulfstream G650", "G650 family"),
    "g650": ("g650_family", "Gulfstream G650", "G650 family"),
    "gulfstream g500": ("g500_family", "Gulfstream G500", "G500 family"),
    "g500": ("g500_family", "Gulfstream G500", "G500 family"),
    "gulfstream g280": ("g280_family", "Gulfstream G280", "G280 family"),
    "g280": ("g280_family", "Gulfstream G280", "G280 family"),
    # Embraer
    "praetor 600": ("praetor_600", "Praetor 600", "Praetor family"),
}


def normalize_aircraft_model(model: str) -> Optional[NormalizedAircraft]:
    raw = (model or "").strip()
    if not raw:
        return None
    key = re.sub(r"\s+", " ", raw.lower())
    key = key.replace("—", "-").replace("–", "-")
    key = re.sub(r"\binc\.?\b", "", key).strip()
    if key in _MODEL_NORMALIZATION:
        ident, display, fam = _MODEL_NORMALIZATION[key]
        return NormalizedAircraft(identity_key=ident, display_name=display, family=fam)

    # Manufacturer prefix stripping (keeps unknowns stable)
    key2 = re.sub(r"^(?:gulfstream|bombardier|embraer|dassault|cessna|textron)\s+", "", key)
    key2 = key2.strip()
    if key2 in _MODEL_NORMALIZATION:
        ident, display, fam = _MODEL_NORMALIZATION[key2]
        return NormalizedAircraft(identity_key=ident, display_name=display, family=fam)

    # Default: stable identity is normalized key itself (family unknown)
    return NormalizedAircraft(identity_key=key2.replace(" ", "_"), display_name=raw, family="")


__all__ = ["NormalizedAircraft", "normalize_aircraft_model"]

