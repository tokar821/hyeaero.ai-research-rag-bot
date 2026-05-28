"""Orchestration Router V2 tests."""

from __future__ import annotations

from services.orchestration.orchestration_router_v2 import (
    OrchestrationQueryTypeV2,
    OrchestrationRendererV2,
    apply_orchestration_v2_metadata,
    detect_hard_conflict_query,
    route_orchestration_v2,
)


def test_named_aircraft_capability_route():
    r = route_orchestration_v2(
        "Can a Citation Longitude fly SFO to Paris nonstop with eight passengers in winter?"
    )
    assert r.query_type == OrchestrationQueryTypeV2.NAMED_AIRCRAFT_CAPABILITY
    assert r.allow_recommendation_ranking is False
    assert r.allow_tier_fallback is False
    assert r.allow_operational_synthesis is False
    assert "Citation Longitude" in r.named_aircraft_models


def test_explicit_comparison_preserves_models():
    q = "Compare Bombardier Challenger 650, Gulfstream G280, Embraer Praetor 600 for 200 hours per year."
    r = route_orchestration_v2(q)
    assert r.query_type == OrchestrationQueryTypeV2.EXPLICIT_COMPARISON
    assert len(r.preserve_comparison_models) >= 2
    assert r.allow_tier_fallback is False


def test_strategic_blocks_shortlist():
    r = route_orchestration_v2(
        "We want one flagship aircraft but the network structurally breaks — what breaks operationally?"
    )
    assert r.query_type == OrchestrationQueryTypeV2.STRATEGIC_FLEET_ANALYSIS
    assert r.renderer == OrchestrationRendererV2.STRATEGIC_ANALYSIS
    assert r.allow_recommendation_ranking is False
    assert r.allow_operational_synthesis is False


def test_network_allows_synthesis_only():
    r = route_orchestration_v2(
        "How should continuation hubs be represented without breaking New York origin integrity?"
    )
    assert r.query_type == OrchestrationQueryTypeV2.NETWORK_STRUCTURE
    assert r.allow_operational_synthesis is True
    assert r.allow_recommendation_ranking is False


def test_recommendation_request_allows_ranking():
    r = route_orchestration_v2("What aircraft should we consider for northern Canada field support?")
    assert r.query_type == OrchestrationQueryTypeV2.RECOMMENDATION_REQUEST
    assert r.allow_recommendation_ranking is True
    assert r.allow_tier_fallback is True
    assert r.allow_operational_synthesis is False


def test_v2_metadata_blocks_tier_downgrade():
    du: dict = {}
    r = route_orchestration_v2("Compare Challenger 350 vs Gulfstream G280")
    apply_orchestration_v2_metadata(du, r)
    assert du.get("tier_downgrade_blocked")
    assert du.get("kernel_synthesis_blocked") is True


def test_hard_conflict_pre_filter_routes_strategic():
    q = (
        "We must keep operating cost below a Global 7500 but need reliable "
        "LAX to Tokyo nonstop in winter with NBAA IFR reserves."
    )
    assert detect_hard_conflict_query(q)
    r = route_orchestration_v2(q)
    assert r.query_type == OrchestrationQueryTypeV2.STRATEGIC_FLEET_ANALYSIS
    assert r.allow_recommendation_ranking is False
    assert r.routing_debug.get("pre_filter_triggered") is True
    assert not r.named_aircraft_models
    assert r.routing_debug.get("final_route") == "strategic_fleet_analysis"


def test_comparison_absolute_priority_three_models():
    q = "Compare Gulfstream G650ER vs Global 7500 vs Falcon 8X for transpacific executive travel."
    r = route_orchestration_v2(q)
    assert r.query_type == OrchestrationQueryTypeV2.EXPLICIT_COMPARISON
    assert r.routing_debug.get("comparison_override_triggered") is True
    assert len(r.preserve_comparison_models) >= 2


def test_network_hierarchy_overrides_recommendation_default():
    q = "How should this hierarchy actually be structured for Dallas and Dubai?"
    r = route_orchestration_v2(q)
    assert r.query_type == OrchestrationQueryTypeV2.NETWORK_STRUCTURE
    assert r.routing_debug.get("network_override_triggered") is True


def test_fleet_strategy_archetype_not_explicit_comparison():
    q = (
        "Which is better for our company: a super-midsize fleet strategy or a single "
        "ultra-long-range flagship aircraft?"
    )
    r = route_orchestration_v2(q)
    assert r.query_type == OrchestrationQueryTypeV2.STRATEGIC_FLEET_ANALYSIS
    assert r.query_type != OrchestrationQueryTypeV2.EXPLICIT_COMPARISON


def test_routing_debug_on_every_route():
    r = route_orchestration_v2("Can a Citation Longitude fly SFO to Paris nonstop?")
    assert r.routing_debug
    assert "intent_type" in r.routing_debug
    assert "final_route" in r.routing_debug
