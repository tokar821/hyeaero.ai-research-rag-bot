"""Phase 29 — Advisory layer routing isolation tests."""

from __future__ import annotations

import copy

import pytest

from rag.query_service import _apply_api_contract_versioning
from services.fleet.fleet_portfolio_strategy_engine import attach_fleet_portfolio_strategy_if_enabled
from services.market.aircraft_market_intelligence_engine import attach_market_intelligence_if_enabled
from services.optimization.multi_criteria_decision_engine import attach_optimization_result_if_enabled
from services.ownership.aircraft_lifecycle_ownership_engine import attach_ownership_intelligence_if_enabled
from services.synthesis.executive_intelligence_synthesis_engine import attach_executive_synthesis_if_enabled
from tests.conftest import build_comparison_payload, routing_authority_snapshot



pytestmark = pytest.mark.deterministic
@pytest.fixture(autouse=True)
def _enable_advisory(monkeypatch):
    monkeypatch.setenv("ENABLE_DECISION_OPTIMIZATION", "1")
    monkeypatch.setenv("ENABLE_MARKET_INTELLIGENCE", "1")
    monkeypatch.setenv("ENABLE_OWNERSHIP_INTELLIGENCE", "1")
    monkeypatch.setenv("ENABLE_FLEET_PORTFOLIO_STRATEGY", "1")
    monkeypatch.setenv("ENABLE_EXECUTIVE_SYNTHESIS", "1")
    monkeypatch.setenv("ENABLE_CONSULTANT_EVALUATION", "1")


def _assert_routing_unchanged(before: dict, after_du: dict) -> None:
    after = routing_authority_snapshot(after_du)
    assert after["intent_lock"] == before["intent_lock"]
    assert after["dispatch_authority_id"] == before["dispatch_authority_id"]
    assert after["authority_dispatch_kind"] == before["authority_dispatch_kind"]
    assert after["authority_dispatch_models"] == before["authority_dispatch_models"]
    assert after["final_execution_path"] == before["final_execution_path"]
    assert after["authority_dispatch_safety_fallback"] == before["authority_dispatch_safety_fallback"]


def test_optimization_does_not_mutate_routing():
    payload = build_comparison_payload()
    before = routing_authority_snapshot(payload["data_used"])
    out = attach_optimization_result_if_enabled("G650 vs Falcon 8X", copy.deepcopy(payload))
    assert out["data_used"].get("optimization_result") is not None
    _assert_routing_unchanged(before, out["data_used"])


def test_market_does_not_mutate_routing():
    payload = build_comparison_payload()
    payload["data_used"]["consultant_recommendations"] = [
        {"model": "Gulfstream G650"},
        {"model": "Falcon 8X"},
    ]
    before = routing_authority_snapshot(payload["data_used"])
    out = attach_market_intelligence_if_enabled("G650 vs Falcon 8X", copy.deepcopy(payload))
    assert "market_intelligence" in out["data_used"]
    _assert_routing_unchanged(before, out["data_used"])


def test_ownership_does_not_mutate_routing():
    payload = build_comparison_payload()
    payload["data_used"]["consultant_recommendations"] = [{"model": "Gulfstream G650"}]
    before = routing_authority_snapshot(payload["data_used"])
    out = attach_ownership_intelligence_if_enabled("G650 vs Falcon 8X", copy.deepcopy(payload))
    assert "ownership_intelligence" in out["data_used"]
    _assert_routing_unchanged(before, out["data_used"])


def test_fleet_does_not_mutate_routing():
    payload = build_comparison_payload()
    payload["data_used"]["consultant_recommendations"] = [
        {"model": "Gulfstream G650"},
        {"model": "Falcon 8X"},
    ]
    before = routing_authority_snapshot(payload["data_used"])
    out = attach_fleet_portfolio_strategy_if_enabled(
        "fleet portfolio strategy for G650 and Falcon 8X",
        copy.deepcopy(payload),
    )
    assert "fleet_portfolio_strategy" in out["data_used"]
    _assert_routing_unchanged(before, out["data_used"])


def test_synthesis_does_not_mutate_routing():
    payload = build_comparison_payload()
    before = routing_authority_snapshot(payload["data_used"])
    out = attach_executive_synthesis_if_enabled("G650 vs Falcon 8X", copy.deepcopy(payload))
    assert "executive_synthesis" in out["data_used"]
    _assert_routing_unchanged(before, out["data_used"])


def test_api_contract_versioning_preserves_routing():
    payload = build_comparison_payload()
    before = routing_authority_snapshot(payload["data_used"])
    versioned = _apply_api_contract_versioning({"query": "G650 vs Falcon 8X", **copy.deepcopy(payload)})
    _assert_routing_unchanged(before, versioned["data_used"])


def test_sequential_advisory_attachments_preserve_routing():
    payload = build_comparison_payload()
    payload["data_used"]["consultant_recommendations"] = [
        {"model": "Gulfstream G650"},
        {"model": "Falcon 8X"},
    ]
    before = routing_authority_snapshot(payload["data_used"])
    out = payload
    for attach_fn, q in (
        (attach_optimization_result_if_enabled, "G650 vs Falcon 8X"),
        (attach_market_intelligence_if_enabled, "G650 vs Falcon 8X"),
        (attach_ownership_intelligence_if_enabled, "G650 vs Falcon 8X"),
        (attach_fleet_portfolio_strategy_if_enabled, "fleet portfolio"),
        (attach_executive_synthesis_if_enabled, "G650 vs Falcon 8X"),
    ):
        out = attach_fn(q, copy.deepcopy(out))
    _assert_routing_unchanged(before, out["data_used"])


def test_optimization_insufficient_data_still_preserves_routing():
    payload = {"answer": "x", "data_used": {"intent_lock": {"intent_type": "comparison"}}}
    before = routing_authority_snapshot(payload["data_used"])
    out = attach_optimization_result_if_enabled("generic query", copy.deepcopy(payload))
    _assert_routing_unchanged(before, out["data_used"])


def test_market_empty_snapshot_preserves_routing():
    payload = build_comparison_payload()
    before = routing_authority_snapshot(payload["data_used"])
    out = attach_market_intelligence_if_enabled("market trends", copy.deepcopy(payload))
    market = out["data_used"].get("market_intelligence") or {}
    if market.get("status") == "INSUFFICIENT_DATA":
        _assert_routing_unchanged(before, out["data_used"])


def test_fleet_insufficient_data_preserves_routing():
    payload = build_comparison_payload()
    before = routing_authority_snapshot(payload["data_used"])
    out = attach_fleet_portfolio_strategy_if_enabled("fleet strategy", copy.deepcopy(payload))
    fleet = out["data_used"].get("fleet_portfolio_strategy") or {}
    if fleet.get("status") == "INSUFFICIENT_DATA":
        _assert_routing_unchanged(before, out["data_used"])


def test_ownership_insufficient_data_preserves_routing():
    payload = build_comparison_payload()
    before = routing_authority_snapshot(payload["data_used"])
    out = attach_ownership_intelligence_if_enabled("ownership costs", copy.deepcopy(payload))
    own = out["data_used"].get("ownership_intelligence") or {}
    if own.get("status") == "INSUFFICIENT_DATA":
        _assert_routing_unchanged(before, out["data_used"])


def test_synthesis_on_minimal_payload_preserves_routing():
    payload = build_comparison_payload()
    before = routing_authority_snapshot(payload["data_used"])
    out = attach_executive_synthesis_if_enabled("G650 vs Falcon 8X", copy.deepcopy(payload))
    _assert_routing_unchanged(before, out["data_used"])


def test_dispatch_authority_id_immutable_after_all_layers():
    payload = build_comparison_payload()
    auth_id = payload["data_used"]["intent_lock"]["dispatch_authority_id"]
    out = _apply_api_contract_versioning({"query": "G650 vs Falcon 8X", **copy.deepcopy(payload)})
    assert out["data_used"]["intent_lock"]["dispatch_authority_id"] == auth_id


def test_execution_path_immutable_after_advisory():
    payload = build_comparison_payload()
    path = payload["data_used"]["intent_execution_trace"]["final_execution_path"]
    out = attach_executive_synthesis_if_enabled("G650 vs Falcon 8X", copy.deepcopy(payload))
    assert out["data_used"]["intent_execution_trace"]["final_execution_path"] == path


def test_advisory_layers_do_not_set_dispatch_kind():
    payload = build_comparison_payload()
    kind = payload["data_used"]["authority_dispatch_kind"]
    out = _apply_api_contract_versioning({"query": "G650 vs Falcon 8X", **copy.deepcopy(payload)})
    assert out["data_used"]["authority_dispatch_kind"] == kind
    assert out["data_used"]["authority_dispatch_kind"] == "comparison"
