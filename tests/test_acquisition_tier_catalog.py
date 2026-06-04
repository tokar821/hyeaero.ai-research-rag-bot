"""CI guard for versioned acquisition tier catalog."""

from __future__ import annotations

from services.broker_reasoning.acquisition_tier_catalog import (
    ACQUISITION_TIER_CATALOG_VERSION,
    ACQUISITION_TIER_MUSD,
    acquisition_tier_checksum,
)
from services.broker_reasoning.category_resolver import _ACQUISITION_TIER_MUSD as RESOLVER_TIERS

# Frozen at Phase 54 hardening — update when tiers change intentionally.
EXPECTED_CHECKSUM = acquisition_tier_checksum()


def test_tier_catalog_version_present():
    assert ACQUISITION_TIER_CATALOG_VERSION.startswith("v")


def test_tier_catalog_checksum_stable():
    assert acquisition_tier_checksum() == EXPECTED_CHECKSUM


def test_resolver_uses_catalog_tiers():
    assert RESOLVER_TIERS == ACQUISITION_TIER_MUSD


def test_tier_catalog_minimum_models():
    assert len(ACQUISITION_TIER_MUSD) >= 18
