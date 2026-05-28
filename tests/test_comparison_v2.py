"""Comparison v2 structured output tests."""

from __future__ import annotations

import json

from services.comparison.aircraft_registry_lock import lock_comparison_aircraft
from services.comparison.comparison_pipeline_v2 import run_comparison_v2
from services.comparison.comparison_renderer_v2 import render_insufficient_data
from services.comparison.comparison_validator_v2 import validate_comparison_payload
from services.consultant.comparison_structured_output import format_comparison_response
from services.consultant.mission_state import MissionState


def _mission() -> MissionState:
    return MissionState(
        passenger_count=8,
        nonstop_requirement=True,
        seasonal_constraints="winter",
        westbound=True,
    )


def test_registry_dedupes_duplicate_tokens():
    lock = lock_comparison_aircraft(["Gulfstream G650", "G650", "G650ER", "Global 7500"])
    assert len(lock.canonical) == 3
    assert "Global 7500" in lock.canonical
    assert "Gulfstream G650ER" in lock.canonical or "Gulfstream G650" in lock.canonical


def test_registry_rejects_unverified_and_unknown():
    lock = lock_comparison_aircraft(["Unverified", "Gulfstream G650ER", "fake-jet-xyz"])
    assert "Gulfstream G650ER" in lock.canonical
    assert "Unverified" in lock.rejected


def test_insufficient_when_fewer_than_two_canonical():
    out = render_insufficient_data("missing canonical aircraft set")
    data = json.loads(out)
    assert data["status"] == "INSUFFICIENT_DATA"
    assert data["mode"] == "explicit_comparison"
    assert "reason" in data
    assert validate_comparison_payload(data).ok


def test_valid_comparison_returns_json_only():
    mission = _mission()
    raw = format_comparison_response(
        query="Compare Gulfstream G650ER vs Global 7500",
        mission=mission,
        compare_models=["Gulfstream G650ER", "Global 7500"],
    )
    assert raw.strip().startswith("{")
    assert "| Aircraft |" not in raw
    assert "Unverified" not in raw
    data = json.loads(raw)
    assert data["mode"] == "explicit_comparison"
    assert data["data_quality"]["status"] == "OK"
    assert len(data["aircraft"]) == 2
    names = {a["name"] for a in data["aircraft"]}
    assert "Gulfstream G650ER" in names
    assert "Global 7500" in names
    assert validate_comparison_payload(data).ok


def test_unknown_models_insufficient_no_prose():
    mission = _mission()
    raw = run_comparison_v2(
        query="G500 vs G600 vs Global 5500",
        mission=mission,
        compare_models=["G500", "G600", "Global 5500"],
        mode="explicit_comparison",
    )
    data = json.loads(raw)
    assert data["status"] == "INSUFFICIENT_DATA"
    assert "Comparison Type:" not in raw
    assert "## " not in raw


def test_no_hybrid_narrative_in_output():
    mission = _mission()
    raw = format_comparison_response(
        query="Compare Challenger 650 and Praetor 600",
        mission=mission,
        compare_models=["Challenger 650", "Praetor 600"],
    )
    parsed = json.loads(raw)
    assert "strategic" not in raw.lower() or parsed.get("mode") == "explicit_comparison"
    assert "recommend" not in raw.lower() or "recommendation" not in raw.lower()
