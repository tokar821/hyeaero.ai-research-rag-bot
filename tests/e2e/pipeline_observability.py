"""
Phase 54 — observability enrichment on ``data_used`` (measurement only).

Maps execution outputs → observability keys. MUST NOT call inference, ranking,
or deal-quality computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

REQUIRED_OBSERVABILITY_KEYS = (
    "execution_path",
    "broker_certify_path",
    "broker_certify_prefer_e2e",
    "tier_source",
    "market_source",
    "tier_fallback_used",
    "executive_applied",
    "deal_quality_observed",
)

VALID_EXECUTION_PATHS = frozenset({"e2e", "layers"})


@dataclass(frozen=True)
class ExecutionResult:
    """Snapshot of broker_certify execution for observability mapping only."""

    path: str
    prefer_e2e: bool
    tier_source: str
    market_source: str
    executive_applied: bool
    deal_quality_observed: bool
    tier_fallback_used: bool
    acquisition_tier_catalog_version: str = ""
    tier_fallback_checksum: str = ""


def _market_source(du: dict) -> str:
    if du.get("market_band_source"):
        return str(du["market_band_source"])
    mr = du.get("market_reality") or {}
    if isinstance(mr, dict) and mr.get("band_mid_usd") is not None:
        return "market_reality_band"
    band = du.get("market_band") or {}
    if isinstance(band, dict) and band.get("reason"):
        return str(band["reason"])
    if du.get("market_band_fallback_warnings"):
        return "catalog_acquisition_tier"
    return "unknown"


def _tier_source(du: dict) -> str:
    if du.get("acquisition_tier_catalog_version"):
        return f"catalog:{du['acquisition_tier_catalog_version']}"
    if du.get("market_band_source") == "catalog_acquisition_tier":
        return "catalog_acquisition_tier"
    if du.get("market_band_fallback_warnings"):
        return "catalog_fallback"
    feasibility = du.get("acquisition_budget_feasibility") or du.get("budget_feasibility")
    if isinstance(feasibility, dict) and feasibility.get("tier_musd") is not None:
        return "feasibility_tier"
    return "none"


def _tier_fallback_used(du: dict) -> bool:
    return bool(
        du.get("market_band_fallback_warnings")
        or du.get("market_band_source") == "catalog_acquisition_tier"
        or du.get("acquisition_tier_catalog_version")
    )


def _deal_quality_observed(du: dict) -> bool:
    dq = du.get("deal_quality")
    return isinstance(dq, dict) and bool(dq.get("verdict"))


def _executive_applied(du: dict, *, path: str) -> bool:
    if path != "layers":
        return False
    if du.get("executive_broker_layer_applied"):
        return True
    rec = du.get("executive_recommendation") or {}
    return bool(isinstance(rec, dict) and rec.get("primary_recommendation"))


def build_execution_result(du: dict, *, path: str, prefer_e2e: bool) -> ExecutionResult:
    """Derive observability snapshot from execution artifacts already in ``du``."""
    return ExecutionResult(
        path=path,
        prefer_e2e=prefer_e2e,
        tier_source=_tier_source(du),
        market_source=_market_source(du),
        executive_applied=_executive_applied(du, path=path),
        deal_quality_observed=_deal_quality_observed(du),
        tier_fallback_used=_tier_fallback_used(du),
        acquisition_tier_catalog_version=str(du.get("acquisition_tier_catalog_version") or ""),
        tier_fallback_checksum=str(du.get("acquisition_tier_catalog_checksum") or ""),
    )


def attach_observability(du: dict, execution: ExecutionResult) -> None:
    """Map execution → observability contract (no inference)."""
    if not isinstance(du, dict):
        return

    prior_path = du.get("broker_certify_path")
    du["execution_path"] = execution.path
    du["broker_certify_path"] = execution.path
    du["broker_certify_prefer_e2e"] = execution.prefer_e2e
    du["tier_source"] = execution.tier_source
    du["market_source"] = execution.market_source
    du["tier_fallback_used"] = execution.tier_fallback_used
    du["executive_applied"] = execution.executive_applied
    du["deal_quality_observed"] = execution.deal_quality_observed

    if execution.acquisition_tier_catalog_version:
        du.setdefault("acquisition_tier_catalog_version", execution.acquisition_tier_catalog_version)
    if execution.tier_fallback_used and execution.tier_fallback_checksum:
        du["tier_fallback_checksum"] = execution.tier_fallback_checksum

    if prior_path and prior_path != execution.path:
        du["execution_path_mismatch"] = {
            "broker_certify_path": prior_path,
            "returned_path": execution.path,
        }
    elif du.get("execution_path") != du.get("broker_certify_path"):
        du["execution_path_mismatch"] = True

    if execution.prefer_e2e and execution.path == "layers":
        du["e2e_unavailable_fallback"] = True


def attach_pipeline_observability(
    du: dict,
    *,
    path: str,
    prefer_e2e: Optional[bool] = None,
) -> None:
    """Mirror broker_certify execution into canonical observability keys."""
    if not isinstance(du, dict):
        return
    if prefer_e2e is None:
        prefer_e2e = bool(du.get("broker_certify_prefer_e2e", path == "e2e"))
    execution = build_execution_result(du, path=path, prefer_e2e=prefer_e2e)
    attach_observability(du, execution)


def assert_required_observability_keys(du: dict) -> None:
    """CI guard: all contract keys present with valid types."""
    for key in REQUIRED_OBSERVABILITY_KEYS:
        assert key in du, f"missing observability key: {key}"
    assert du["execution_path"] in VALID_EXECUTION_PATHS
    assert du["broker_certify_path"] in VALID_EXECUTION_PATHS
    assert isinstance(du["broker_certify_prefer_e2e"], bool)
    assert isinstance(du["tier_source"], str)
    assert isinstance(du["market_source"], str)
    assert isinstance(du["executive_applied"], bool)
    assert isinstance(du["deal_quality_observed"], bool)
    assert isinstance(du["tier_fallback_used"], bool)


def assert_observability_contract(du: dict, *, path: str, prefer_e2e: bool) -> None:
    """CI: full observability contract after broker_certify."""
    assert_required_observability_keys(du)
    assert du.get("broker_certify_path") == path, (
        f"broker_certify_path={du.get('broker_certify_path')!r} != returned path={path!r}"
    )
    assert du.get("execution_path") == path
    assert du.get("broker_certify_prefer_e2e") is prefer_e2e
    assert not du.get("execution_path_mismatch"), (
        f"execution/observability path mismatch: {du.get('execution_path_mismatch')}"
    )


def assert_listing_observability(du: dict, *, inferred_verdict: str) -> None:
    """Listing CI: verdict label + execution-derived market signals."""
    assert_required_observability_keys(du)
    assert inferred_verdict, "infer_listing_verdict returned empty"
    if du.get("listing_price_infeasible") or du.get("acquisition_budget_infeasible"):
        return
    has_dq = du["deal_quality_observed"]
    has_band = bool(du.get("market_reality")) or bool(du.get("market_band_source"))
    assert has_dq or has_band, (
        "listing missing deal_quality and market band observability "
        f"(market_source={du.get('market_source')!r} deal_quality_observed={has_dq})"
    )


def assert_mission_execution_contract(du: dict, *, path: str, primary: str) -> None:
    """Mission KPI: layers path, executive applied, meaningful primary."""
    from tests.e2e.production_audit_helpers import _primary_is_meaningful

    assert_observability_contract(du, path=path, prefer_e2e=False)
    assert path == "layers", f"mission requires layers path, got {path!r}"
    assert du["executive_applied"] is True, "mission requires executive_applied"
    assert _primary_is_meaningful(primary), f"mission missing primary_recommendation: {primary!r}"


def assert_replay_row_observability(du: dict, *, path: str, prefer_e2e: bool) -> None:
    """Replay suite per-row observability contract."""
    assert_observability_contract(du, path=path, prefer_e2e=prefer_e2e)
    assert "execution_path" in du
    assert "broker_certify_path" in du
    assert du["execution_path"] in VALID_EXECUTION_PATHS


__all__ = [
    "ExecutionResult",
    "REQUIRED_OBSERVABILITY_KEYS",
    "VALID_EXECUTION_PATHS",
    "attach_observability",
    "attach_pipeline_observability",
    "build_execution_result",
    "assert_required_observability_keys",
    "assert_observability_contract",
    "assert_listing_observability",
    "assert_mission_execution_contract",
    "assert_replay_row_observability",
]
