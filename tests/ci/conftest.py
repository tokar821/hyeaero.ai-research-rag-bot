"""Phase 31 — CI tier marker application hook."""

from __future__ import annotations

from tests.ci.tier_registry import apply_tier_markers


def pytest_collection_modifyitems(config, items):
    apply_tier_markers(items)
