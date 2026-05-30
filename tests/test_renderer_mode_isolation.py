"""Renderer mode isolation and builder tests."""

from __future__ import annotations

import json
import re

from services.rendering.renderer_response_builder import (
    RendererBuildContext,
    build_renderer_envelope,
    normalize_aircraft_id,
)
from services.consultant.mission_state import MissionState


def test_normalize_spoken_global_alias():
    assert normalize_aircraft_id("Global Seven Five Zero Zero") == "Global 7500"
    assert normalize_aircraft_id("G650ER") == "Gulfstream G650ER"


def test_comparison_from_json_no_markdown():
    comparison_json = json.dumps(
        {
            "mode": "explicit_comparison",
            "aircraft": [
                {"name": "Gulfstream G650ER", "category": "ULR", "mission_fit_score": 0.81},
                {"name": "Global 7500", "category": "ULR", "mission_fit_score": 0.79},
            ],
            "comparison_matrix": {"dimensions": ["range", "mission_fit"]},
            "verdict": {"best_overall": "Gulfstream G650ER"},
            "data_quality": {"status": "OK"},
        }
    )
    env = build_renderer_envelope(
        RendererBuildContext(
            mode="explicit_comparison",
            answer=comparison_json,
            mission=MissionState(),
        )
    )
    assert env.mode == "explicit_comparison"
    assert env.component == "comparison_table_v2"
    assert "| Aircraft |" not in json.dumps(env.to_dict())
    assert len(env.payload["comparison_rows"]) == 2


def test_capability_no_shortlist_contamination():
    env = build_renderer_envelope(
        RendererBuildContext(
            mode="named_aircraft_capability",
            mission=MissionState(passenger_count=8, routes=["LAX-LHR"]),
            named_aircraft_models=["Falcon 8X"],
        )
    )
    assert env.mode == "named_aircraft_capability"
    blob = json.dumps(env.payload)
    assert "shortlist" not in blob.lower()
    assert "comparison_rows" not in blob


def test_strategic_no_shortlist():
    env = build_renderer_envelope(
        RendererBuildContext(
            mode="strategic_fleet_analysis",
            mission=MissionState(routes=["NYC-LON"]),
            data_used={"mission_hard_invalid": True},
        )
    )
    assert env.mode == "strategic_fleet_analysis"
    assert not env.payload.get("shortlist")


def test_build_fail_closed_on_empty_capability():
    env = build_renderer_envelope(
        RendererBuildContext(mode="named_aircraft_capability", mission=MissionState())
    )
    assert env.mode == "error"
    assert env.payload["reason"]


def test_spoken_alias_three_way_comparison_rebuild():
    env = build_renderer_envelope(
        RendererBuildContext(
            mode="explicit_comparison",
            query=(
                "Compare: Global Seven Five Zero Zero, G650ER, Falcon Eight X "
                "for nonstop Los Angeles to Tokyo winter missions."
            ),
            answer='{"status":"INSUFFICIENT_DATA"}',
            mission=MissionState(),
        )
    )
    assert env.mode == "explicit_comparison"
    names = [a["name"] for a in env.payload["aircraft"]]
    assert "Global 7500" in names
    assert "Gulfstream G650ER" in names
    assert "Falcon 8X" in names


def test_impossible_mission_uplifts_to_strategic():
    env = build_renderer_envelope(
        RendererBuildContext(
            mode="named_aircraft_capability",
            query=(
                "I want much lower operating costs than a Global 7500, guaranteed nonstop "
                "Sydney westbound in winter, one aircraft only, Aspen capability, 14 passengers."
            ),
            named_aircraft_models=["Global 7500"],
            mission=MissionState(passenger_count=14),
        )
    )
    assert env.mode == "strategic_fleet_analysis"
    assert not env.payload.get("shortlist")
