"""Render guard and output discipline tests."""

from __future__ import annotations

import json

from services.rendering.prose_renderer_v2 import (
    is_incomplete_query,
    is_raw_json_leakage,
    render_comparison_prose,
    render_incomplete_query,
)
from services.rendering.render_guard import render_fail_closed, render_from_envelope
from services.sanity.aircraft_class_guard import violates_class_sanity
from services.catalog.catalog_alias_resolver import (
    resolve_canonical_display_name,
    resolve_catalog_profile_key,
)
from services.consultant.mission_state import MissionState
from services.orchestration.pipeline_orchestrator import ConsultantOrchestrationResult


def test_incomplete_query_colon():
    assert is_incomplete_query("Leadership insists:")
    assert render_incomplete_query() in render_fail_closed(
        ConsultantOrchestrationResult(answer="", mission_state=MissionState()),
        query="Leadership insists:",
    )


def test_raw_json_leakage_detected():
    blob = json.dumps({"mode": "explicit_comparison", "component": "x", "payload": {}})
    assert is_raw_json_leakage(blob)


def test_comparison_insufficient_no_partial_table():
    text = render_comparison_prose({"aircraft": [{"name": "A"}]})
    assert text == "INSUFFICIENT DATA FOR STRUCTURED COMPARISON"
    assert "| Aircraft |" not in text


def test_class_guard_blocks_cj_on_ulr():
    mission = MissionState(routes=["LAX-LHR"], passenger_count=10)
    assert violates_class_sanity(mission, "Citation CJ2", query="westbound winter NBAA")


def test_catalog_alias_global_6000():
    assert resolve_canonical_display_name("Global 6000") == "Bombardier Global 6000"
    assert resolve_catalog_profile_key("Global 6000") == "Global 6500"


def test_render_fail_closed_prose_not_json():
    env = {
        "mode": "named_aircraft_capability",
        "component": "capability_verdict_v2",
        "payload": {
            "aircraft": "Falcon 8X",
            "mission": {},
            "verdict": "MARGINAL",
            "constraints": [{"type": "constraint", "detail": "winter margin"}],
        },
        "meta": {},
    }
    text = render_from_envelope(
        env,
        query="Could a Falcon 8X fly LAX-LHR westbound in winter?",
        mission=MissionState(routes=["LAX-LHR"], passenger_count=10),
    )
    assert not text.strip().startswith("{")
    assert "shortlist" not in text.lower()
