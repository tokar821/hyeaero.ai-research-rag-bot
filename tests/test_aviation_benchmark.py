"""
Aviation mission benchmark framework tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.aviation_benchmark_runner import load_aviation_mission_suite, run_aviation_mission_benchmark
from evals.aviation_benchmark_scoring import detect_automated_failures, score_benchmark_case
from evals.aviation_benchmark_report import generate_benchmark_report

_SUITE = Path(__file__).resolve().parents[1] / "evals" / "aviation_mission_suite.json"


def test_suite_loads():
    suite = load_aviation_mission_suite(_SUITE)
    assert suite["schema_version"] in (1, 2)
    assert len(suite["cases"]) >= 20
    cats = {c["category"] for c in suite["cases"]}
    assert "runway_flexibility" in cats
    assert "asia_nonstop" in cats


def test_miami_caribbean_golden_forbids_g650():
    case = next(c for c in load_aviation_mission_suite(_SUITE)["cases"] if c["id"] == "runway_001")
    report = run_aviation_mission_benchmark(case_ids=["runway_001"], suite_path=_SUITE)
    row = next(c for c in report["cases"] if c["id"] == "runway_001")
    assert "Gulfstream G650" in case["golden"]["forbidden_any_models"]
    assert not any("impossible_aircraft_recommended:Gulfstream G650" in f for f in row["automated_failures"])


def test_tokyo_prefers_ultra_long():
    report = run_aviation_mission_benchmark(case_ids=["asia_001"], suite_path=_SUITE)
    row = next(c for c in report["cases"] if c["id"] == "asia_001")
    top = row["metadata"]["top_recommendations"]
    assert any(m in top[:3] for m in ("Gulfstream G650", "Falcon 8X", "Global 7500"))


def test_turn_leak_detection():
    failures = detect_automated_failures(
        case={
            "input": "8 passengers Miami to Caribbean",
            "prior_context_must_not_leak": {"routes": ["Los Angeles -> Tokyo"]},
            "golden": {},
        },
        turn_profile={
            "routes": [{"origin": "Los Angeles", "destination": "Tokyo"}],
            "passengers": 8,
        },
        merged_profile={},
        recommendations=[],
        answer="",
        mission_category="regional_utility",
    )
    assert any("previous_turn_leak" in f for f in failures)


def test_invalid_route_scoring():
    result = score_benchmark_case(
        case={
            "id": "x",
            "category": "range_realism",
            "golden": {"routes_must_be_empty": True},
        },
        turn_profile={"routes": [{"origin": "What Would You Like", "destination": "Work"}]},
        merged_profile={},
        mission_state={},
        recommendations=[],
    )
    assert result.scores["route_accuracy"] < 0.5 or result.automated_failures


def test_report_has_required_sections():
    report = run_aviation_mission_benchmark(
        suite_path=_SUITE,
        categories=["runway_flexibility", "asia_nonstop"],
    )
    assert "contamination_report" in report
    assert "realism_score" in report
    assert "aircraft_diversity_score" in report
    assert "recommendation_precision" in report
    assert "dimension_scores" in report
    assert "mission_understanding" in report["dimension_scores"]


def test_benchmark_runner_pass_rate():
    report = run_aviation_mission_benchmark(suite_path=_SUITE)
    assert report["summary"]["total_cases"] >= 20
    assert report["summary"]["pass_rate"] >= 0.5
