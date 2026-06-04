"""
Versioned acquisition tier catalog — single source for _ACQUISITION_TIER_MUSD.

Used for ranking, feasibility, and catalog market-band fallback. Changes affect
listing deal quality, benchmarks, and executive selection; version is logged when
catalog bands are used.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# Bump when tier values or model keys change (benchmark ground truth must be reviewed).
ACQUISITION_TIER_CATALOG_VERSION = "v1.0.0"

ACQUISITION_TIER_MUSD: Dict[str, float] = {
    "Gulfstream G280": 12.0,
    "Gulfstream G650": 45.0,
    "Gulfstream G650ER": 48.0,
    "Gulfstream G700": 65.0,
    "Falcon 2000": 18.0,
    "Falcon 7X": 35.0,
    "Falcon 8X": 50.0,
    "Citation CJ2": 4.0,
    "Citation CJ4": 7.0,
    "Citation Latitude": 14.0,
    "Citation Longitude": 22.0,
    "Praetor 600": 18.0,
    "Challenger 350": 18.0,
    "Challenger 650": 28.0,
    "Challenger Longitude": 32.0,
    "Global 6500": 42.0,
    "Global 7500": 58.0,
    "Learjet 75": 6.0,
    "Pilatus PC-24": 9.0,
    "Phenom 300": 9.0,
}


def acquisition_tier_checksum() -> str:
    """Stable SHA-256 (first 16 hex chars) for CI validation."""
    payload = json.dumps(
        {"version": ACQUISITION_TIER_CATALOG_VERSION, "tiers": ACQUISITION_TIER_MUSD},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def record_catalog_band_usage(data_used: dict, *, model: str, band_reason: str) -> None:
    """Attach observability when market bands fall back to catalog tiers."""
    if not isinstance(data_used, dict):
        return
    if band_reason != "catalog_acquisition_tier":
        return
    data_used["acquisition_tier_catalog_version"] = ACQUISITION_TIER_CATALOG_VERSION
    data_used["acquisition_tier_catalog_checksum"] = acquisition_tier_checksum()
    data_used["market_band_source"] = "catalog_acquisition_tier"
    msg = (
        f"Market band for {model} uses catalog tier {ACQUISITION_TIER_CATALOG_VERSION} "
        f"(checksum {acquisition_tier_checksum()}) — not listing DB or authority."
    )
    data_used.setdefault("market_band_fallback_warnings", []).append(msg)
    logger.warning("acquisition_tier_catalog_fallback model=%s version=%s", model, ACQUISITION_TIER_CATALOG_VERSION)


__all__ = [
    "ACQUISITION_TIER_CATALOG_VERSION",
    "ACQUISITION_TIER_MUSD",
    "acquisition_tier_checksum",
    "record_catalog_band_usage",
]
