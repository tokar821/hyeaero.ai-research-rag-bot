"""Context continuity tests (Priority 8)."""

from __future__ import annotations

from services.conversation_continuity.context_continuity import (
    attach_context_continuity,
    resolve_context_continuity,
)
from services.consultant.mission_state import MissionState
from services.operations.operational_realism_bridge import assess_mission_operational_realism


def test_reference_aircraft_continuity():
    state = resolve_context_continuity(
        "I still need lower operating cost than a Global 7500 for the same network as before."
    )
    assert state.reference_aircraft or state.cost_ceiling_reference
    assert state.apply_to_ranking


def test_attach_context_persists():
    du = {}
    attach_context_continuity(
        du,
        "still lower cost than G650ER",
        broker_memory={"recurring_routes": ["Dallas-Houston"]},
    )
    assert du.get("context_continuity")
    assert du.get("continuity_reference_aircraft") or du["context_continuity"].get(
        "reference_aircraft"
    )


def test_operational_realism_bridge():
    mission = MissionState(routes=["LAX-LHR"], passenger_count=10, westbound=True)
    spec = {"practical_nm": 5600, "range_nm": 6000, "max_pax": 16, "category": "ULR"}
    out = assess_mission_operational_realism(
        mission,
        "Falcon 8X",
        spec,
        query="westbound winter LAX to London",
    )
    assert "reserve" in out
    assert "dispatch" in out
