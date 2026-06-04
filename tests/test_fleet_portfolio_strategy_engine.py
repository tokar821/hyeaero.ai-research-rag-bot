"""Phase 26 — Fleet Portfolio Strategy Engine tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.deterministic

from services.fleet.fleet_portfolio_strategy_engine import (
    FleetInput,
    analyze_fleet_redundancy,
    attach_fleet_portfolio_strategy_if_enabled,
    build_5_year_upgrade_path,
    build_fleet_portfolio_strategy,
    build_fleet_portfolio_strategy_report,
    build_mission_coverage_map,
    compute_fleet_cost_overlap,
    evaluate_fleet_portfolio_strategy_hooks,
    fleet_portfolio_strategy_enabled,
    identify_fleet_gaps,
    optimize_fleet_structure,
    rank_aircraft_for_replacement,
)


def _mixed_fleet():
    return ["Citation Latitude", "Gulfstream G650", "Citation CJ3+"]


def _redundant_fleet():
    return ["Gulfstream G650", "Gulfstream G550", "Falcon 8X"]


def test_multi_aircraft_fleet_coverage():
    coverage = build_mission_coverage_map(_mixed_fleet())
    assert coverage["transcontinental"] > 0
    assert coverage["regional"] >= coverage["intercontinental"]
    assert all(0 <= v <= 100 for v in coverage.values())


def test_redundancy_detection():
    redundancy = analyze_fleet_redundancy(_redundant_fleet())
    assert redundancy["redundancy_score"] > 0
    assert redundancy["overlapping_range_pairs"] or redundancy["overlapping_cabin_pairs"]


def test_gap_identification():
    gaps = identify_fleet_gaps(["Citation CJ3+"])
    assert gaps
    assert any("long-range" in g or "intercontinental" in g or "missing" in g for g in gaps)


def test_cost_overlap_detection():
    matrix = compute_fleet_cost_overlap(_redundant_fleet())
    assert "Gulfstream G650" in matrix
    assert matrix["Gulfstream G650"]["Gulfstream G550"] > 0
    assert matrix["Gulfstream G650"]["Gulfstream G650"] == 100.0


def test_replacement_ranking():
    order = rank_aircraft_for_replacement(_mixed_fleet())
    assert len(order) == 3
    assert len(set(order)) == 3


def test_upgrade_plan_generation():
    fleet = _mixed_fleet()
    replacement = rank_aircraft_for_replacement(fleet)
    recs = optimize_fleet_structure(FleetInput(aircraft_owned=fleet))
    plan = build_5_year_upgrade_path(fleet, replacement, recs)
    assert len(plan) == 5
    assert plan[0]["year"] == 1
    assert "retire_aircraft" in plan[0]


def test_efficiency_scoring():
    report = build_fleet_portfolio_strategy_report(
        FleetInput(aircraft_owned=_mixed_fleet(), annual_utilization_hours=400),
    )
    assert 0 <= report.total_fleet_efficiency_score <= 100
    assert report.mission_coverage_map


def test_constraint_satisfaction():
    fleet_input = FleetInput(
        aircraft_owned=_mixed_fleet() + ["Challenger 3500"],
        budget_constraints={"max_aircraft": 3},
    )
    recs = optimize_fleet_structure(fleet_input)
    assert any("budget cap" in r for r in recs)


def test_reproducibility():
    fleet_input = FleetInput(aircraft_owned=_mixed_fleet())
    a = build_fleet_portfolio_strategy_report(fleet_input)
    b = build_fleet_portfolio_strategy_report(fleet_input)
    assert a.strategy_id == b.strategy_id
    assert a.to_dict() == b.to_dict()


def test_env_gating(monkeypatch):
    monkeypatch.delenv("ENABLE_FLEET_PORTFOLIO_STRATEGY", raising=False)
    payload = {"answer": "x", "data_used": {}}
    assert not fleet_portfolio_strategy_enabled()
    out = attach_fleet_portfolio_strategy_if_enabled("test", payload)
    assert "fleet_portfolio_strategy" not in (out.get("data_used") or {})

    monkeypatch.setenv("ENABLE_FLEET_PORTFOLIO_STRATEGY", "1")
    out2 = attach_fleet_portfolio_strategy_if_enabled(
        "fleet strategy",
        {
            "data_used": {
                "fleet_input": {
                    "aircraft_owned": _mixed_fleet(),
                    "annual_utilization_hours": 350,
                }
            }
        },
    )
    bundle = (out2.get("data_used") or {}).get("fleet_portfolio_strategy") or {}
    assert bundle
    assert bundle.get("fleet_panel")
    assert bundle.get("replacement_priority_order")


def test_evaluator_hooks():
    bad = {
        "data_used": {
            "fleet_portfolio_strategy": {
                "current_aircraft": ["A", "B"],
                "mission_coverage_map": {"regional": 150.0},
                "redundancy_analysis": {"redundancy_score": 120.0},
                "cost_overlap_matrix": {"A": {"B": 150.0}},
                "replacement_priority_order": ["B", "A"],
                "phased_upgrade_plan": [
                    {"year": 1, "retire_aircraft": ["C"], "acquire_aircraft": []},
                ],
            }
        }
    }
    failures = evaluate_fleet_portfolio_strategy_hooks(bad)
    assert "mission_coverage_consistency" in failures
    assert "redundancy_validity" in failures
    assert "cost_overlap_accuracy" in failures
    assert "upgrade_feasibility" in failures


def test_build_fleet_portfolio_strategy_insufficient_data():
    out = build_fleet_portfolio_strategy("fleet strategy", {"data_used": {}})
    assert out["status"] == "INSUFFICIENT_DATA"
    assert out["confidence"] == 0
    assert out["trends"] == {}
