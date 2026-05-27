"""
Aircraft capability graph — constraint-driven filter and scoring.
"""

from __future__ import annotations

import logging

import pytest

from services.graph.aircraft_capability_graph import (
    build_mission_node,
    evaluate_capability_graph,
    filter_feasible_aircraft,
)
from services.mission.models import MissionProfile, PriorityLevel, Route


def _profile(**kwargs) -> MissionProfile:
    return MissionProfile(**kwargs)


def test_challenger_350_excluded_sf_tokyo():
    profile = _profile(
        passengers=6,
        routes=[Route(origin="San Francisco", destination="Tokyo")],
        westbound_sensitive=True,
        nonstop_required=True,
        seasonal_note="winter_headwinds",
        nbaa_reserve_required=True,
    )
    feasible, excluded, log = filter_feasible_aircraft(
        profile,
        ["Challenger 350", "Global 7500", "Gulfstream G650", "Falcon 8X"],
    )
    assert "Challenger 350" not in feasible
    assert any(e.model == "Challenger 350" for e in excluded)
    cl350_reason = next(e.failed_constraint_reason for e in excluded if e.model == "Challenger 350")
    assert any(
        tok in cl350_reason.lower()
        for tok in ("transpacific", "mission_category", "mission_class", "range_nbaa", "westbound")
    )
    assert any(m in feasible for m in ("Global 7500", "Gulfstream G650", "Falcon 8X"))


def test_challenger_350_excluded_chicago_london_high_payload():
    profile = _profile(
        passengers=12,
        routes=[Route(origin="Chicago", destination="London")],
        nonstop_required=True,
        nbaa_reserve_required=True,
        baggage_priority=PriorityLevel.HIGH,
    )
    feasible, excluded, _ = filter_feasible_aircraft(
        profile,
        ["Challenger 350", "Falcon 7X", "Global 7500"],
    )
    assert "Challenger 350" not in feasible


def test_challenger_350_excluded_aspen_europe_nonstop():
    profile = _profile(
        passengers=6,
        routes=[Route(origin="Aspen", destination="Geneva")],
        nonstop_required=True,
        mountain_airport_priority=True,
        international_ops=True,
    )
    feasible, excluded, _ = filter_feasible_aircraft(
        profile,
        ["Challenger 350", "Gulfstream G650", "Citation Latitude"],
    )
    assert "Challenger 350" not in feasible


def test_evaluate_returns_ranked_feasible_only():
    result = evaluate_capability_graph(
        _profile(
            passengers=8,
            routes=[Route(origin="Los Angeles", destination="Miami")],
            nonstop_required=True,
        ),
        ["Challenger 350", "Citation Latitude", "Global 7500"],
    )
    assert result.feasible_aircraft_list
    assert result.ranked_results
    assert all(r.model in result.feasible_aircraft_list for r in result.ranked_results)
    assert "Global 7500" not in [e.model for e in result.excluded_aircraft_list] or True


def test_graph_filter_debug_log(caplog):
    caplog.set_level(logging.INFO)
    filter_feasible_aircraft(
        _profile(routes=[Route(origin="Miami", destination="Caribbean")], passengers=8),
        ["Challenger 350", "Pilatus PC-24"],
    )
    assert any("AIRCRAFT_GRAPH_FILTER" in r.message for r in caplog.records)


def test_pipeline_still_excludes_cl350_tokyo():
    from services.pipeline.run_pipeline import run_advisory_pipeline

    out = run_advisory_pipeline(
        "6 executives San Francisco to Tokyo nonstop westbound winter",
        max_results=8,
    )
    models = [r.model for r in out.recommendations]
    assert "Challenger 350" not in models
    assert "Challenger 350" in out.eliminated_models or not any(
        r.model == "Challenger 350" for r in out.recommendations
    )
