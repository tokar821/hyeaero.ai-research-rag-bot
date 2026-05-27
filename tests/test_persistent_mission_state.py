"""Persistent internal MissionState — cross-turn update semantics."""

from services.mission import extract_mission
from services.state.mission_state import (
    PERSISTENT_MISSION_STATE_KEY,
    MissionState,
    MissionType,
    advance_persistent_mission_state,
    load_persistent_mission_state,
    persistent_to_mission_profile,
    sync_persistent_mission_state,
)


def test_advance_updates_not_regenerates():
    prior = MissionState(passengers=8, routes=["Los Angeles -> Miami"], turn_count=1)
    turn = extract_mission("What about runway flexibility?")
    updated = advance_persistent_mission_state(prior, turn, "What about runway flexibility?")
    assert updated.passengers == 8
    assert updated.routes == ["Los Angeles -> Miami"]
    assert updated.turn_count == 2
    assert updated.priorities.runway in ("high", "medium", "none")


def test_routes_and_pax_from_turn_applied():
    prior = MissionState()
    turn = extract_mission("10 passengers LA to Miami nonstop $10M")
    updated = advance_persistent_mission_state(prior, turn, "10 passengers LA to Miami nonstop $10M")
    assert updated.passengers == 10
    assert "Miami" in " ".join(updated.routes)


def test_mission_type_comparison():
    prior = MissionState(mission_type=MissionType.BUSINESS_TRAVEL, routes=["NYC -> London"])
    turn = extract_mission("Compare Gulfstream G650 vs Falcon 8X")
    updated = advance_persistent_mission_state(prior, turn, "Compare G650 vs Falcon 8X")
    assert updated.mission_type == MissionType.COMPARISON


def test_implicit_range_from_routes():
    prior = MissionState()
    turn = extract_mission("6 executives San Francisco to Tokyo nonstop")
    updated = advance_persistent_mission_state(
        prior, turn, "6 executives San Francisco to Tokyo nonstop"
    )
    assert updated.range_requirement_nm and updated.range_requirement_nm > 2000


def test_never_stringifies_internal():
    ms = MissionState(passengers=6)
    assert "passengers" not in str(ms).lower()
    assert repr(ms).startswith("<MissionState internal>")


def test_load_and_sync_round_trip():
    du: dict = {}
    p1, prof1, _ = sync_persistent_mission_state(
        "8 passengers Miami to Caribbean",
        data_used=du,
    )
    assert du.get(PERSISTENT_MISSION_STATE_KEY)
    assert prof1.passengers == 8

    loaded = load_persistent_mission_state(data_used=du)
    p2, prof2, _ = sync_persistent_mission_state(
        "short runway focus",
        data_used=du,
    )
    assert p2.passengers == 8
    assert "Caribbean" in " ".join(p2.routes)
    assert prof2.passengers == 8


def test_persistent_to_profile_carries_priorities():
    state = MissionState(
        passengers=6,
        routes=["Los Angeles -> Miami"],
        nonstop_required=True,
    )
    state.priorities.cost = "high"
    profile = persistent_to_mission_profile(state)
    assert profile.passengers == 6
    assert profile.operating_cost_priority.value == "high"
    assert profile.nonstop_required is True
