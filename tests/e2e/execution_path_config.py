"""
Execution path policy for broker measurement suites.

Two paths exist in ``broker_certify``:
  - ``e2e``: consultant retrieval bundle (production-like LLM path)
  - ``layers``: deterministic post-layers pipeline (certification path)

Suites must declare which path they measure. Divergence between paths is
audited in ``test_execution_path_parity.py``.
"""

from __future__ import annotations

import os
from typing import Dict, Literal

ExecutionPath = Literal["e2e", "layers"]

# Certification KPI suites (Phase 53 recertified) — layers only.
CERTIFICATION_PREFER_E2E: bool = False

# Production corpus replay — category-based path policy.
PRODUCTION_REPLAY_PREFER_E2E: bool = os.environ.get("HYEAERO_REPLAY_PREFER_E2E", "1") != "0"

# Default replay path by production query category (when HYEAERO_REPLAY_PREFER_E2E=1).
REPLAY_CATEGORY_PATH: Dict[str, ExecutionPath] = {
    "mission": "layers",
    "buy_decision": "e2e",
    "comparison": "e2e",
    "valuation": "e2e",
    "alternative": "e2e",
    "listing": "e2e",
}

# Categories that always use layers regardless of global replay toggle.
REPLAY_LAYERS_CATEGORIES = frozenset({"mission"})

# Minimum mission primary rate for replay report (session gate).
MISSION_PRIMARY_MIN_RATE_PCT: float = float(os.environ.get("HYEAERO_MISSION_PRIMARY_MIN_PCT", "80"))

# Parity CI strict mode (see test_execution_path_parity.py).
PARITY_STRICT: bool = os.environ.get("HYEAERO_PARITY_STRICT", "1") != "0"


def prefer_e2e_for_replay(category: str) -> bool:
    """Return whether replay should use e2e for this production query category."""
    if category in REPLAY_LAYERS_CATEGORIES:
        return False
    if not PRODUCTION_REPLAY_PREFER_E2E:
        return False
    return REPLAY_CATEGORY_PATH.get(category, "e2e") == "e2e"


def expected_path_for_replay(category: str) -> ExecutionPath:
    """Resolved execution path for a replay category."""
    return "layers" if not prefer_e2e_for_replay(category) else "e2e"


def prefer_e2e_for_suite(suite_name: str) -> bool:
    """Map suite name to default prefer_e2e policy."""
    layers_suites = {
        "real_aircraft_benchmark",
        "listing_validation_suite",
        "market_recommendation_audit",
    }
    if suite_name in layers_suites:
        return CERTIFICATION_PREFER_E2E
    return PRODUCTION_REPLAY_PREFER_E2E


__all__ = [
    "CERTIFICATION_PREFER_E2E",
    "PRODUCTION_REPLAY_PREFER_E2E",
    "MISSION_PRIMARY_MIN_RATE_PCT",
    "PARITY_STRICT",
    "REPLAY_CATEGORY_PATH",
    "REPLAY_LAYERS_CATEGORIES",
    "prefer_e2e_for_replay",
    "expected_path_for_replay",
    "prefer_e2e_for_suite",
]
