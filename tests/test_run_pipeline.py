"""
Pipeline integration — feasibility must filter before scoring and output.
"""

from __future__ import annotations

import logging

from services.mission.models import MissionProfile, Route
from services.pipeline.run_pipeline import (
    filter_candidates_by_feasibility,
    generate_candidate_aircraft_list,
    run_advisory_pipeline,
)


def test_generate_candidate_full_catalog_for_advisory():
    models = generate_candidate_aircraft_list("8 passengers LA to Miami nonstop recommend")
    assert "Challenger 350" in models
    assert "Global 7500" in models
    assert len(models) >= 12


def test_dallas_ny_london_5_pax_eliminates_super_midsize():
    profile = MissionProfile(
        passengers=5,
        routes=[
            Route(origin="Dallas", destination="New York"),
            Route(origin="New York", destination="London"),
        ],
        nonstop_required=True,
        nbaa_reserve_required=True,
        international_ops=True,
    )
    candidates = generate_candidate_aircraft_list("transatlantic mission")
    feasible, _feas, log = filter_candidates_by_feasibility(profile, candidates)
    assert "Challenger 350" not in feasible
    assert "Gulfstream G280" not in feasible
    assert "Praetor 600" not in feasible
    eliminated_names = {e["aircraft"] for e in log if e.get("pass_fail") == "fail"}
    assert "Challenger 350" in eliminated_names
    assert any("Global 7500" in m or "Falcon 8X" in m or "Gulfstream G650" in m for m in feasible)


def test_pipeline_output_excludes_infeasible():
    result = run_advisory_pipeline(
        "8 passengers Miami to Caribbean short runway focus",
        mission_profile=MissionProfile(
            passengers=8,
            routes=[Route(origin="Miami", destination="Caribbean")],
            short_field_priority=__import__(
                "services.mission.models", fromlist=["PriorityLevel"]
            ).PriorityLevel.HIGH,
        ),
        max_results=8,
    )
    models = [r.model for r in result.recommendations]
    assert "Global 7500" not in models
    assert "Falcon 8X" not in models
    assert "Challenger 350" in result.eliminated_models or "Challenger 350" not in models


def test_feasibility_elimination_log_format(caplog):
    caplog.set_level(logging.INFO)
    profile = MissionProfile(
        passengers=15,
        routes=[Route(origin="New York", destination="London")],
        nonstop_required=True,
    )
    _, _, log = filter_candidates_by_feasibility(
        profile,
        ["Challenger 350", "Global 7500"],
    )
    assert log
    assert "aircraft" in log[0]
    assert "pass_fail" in log[0]
    assert any("AIRCRAFT_GRAPH_FILTER" in r.message for r in caplog.records)
