"""Mission state persistence and validate_mission_state_consistency."""

from services.mission import extract_mission
from services.state.mission_state import (
    MissionState,
    advance_persistent_mission_state,
    load_persistent_mission_state,
    sync_persistent_mission_state,
)
from services.state.mission_validation import (
    validateMissionStateConsistency,
    validate_mission_state_consistency,
)


def test_passengers_persist_when_not_mentioned():
    prior = MissionState(passengers=8, routes=["Los Angeles -> Miami"], turn_count=1)
    turn = extract_mission("What about runway flexibility and operating cost?")
    updated = advance_persistent_mission_state(prior, turn, "What about runway flexibility?")
    report = validate_mission_state_consistency(prior, updated, turn, "What about runway flexibility?")
    assert updated.passengers == 8
    assert updated.routes == ["Los Angeles -> Miami"]
    assert report.is_consistent
    assert "passengers" in report.inherited_fields


def test_routes_persist_follow_up_turn():
    du: dict = {}
    sync_persistent_mission_state("8 passengers Miami to Caribbean", data_used=du)
    state, _, report = sync_persistent_mission_state("short runway focus", data_used=du)
    assert state.passengers == 8
    assert "Caribbean" in " ".join(state.routes)
    assert report.is_consistent
    assert not report.needs_route_clarification


def test_multi_city_itinerary_skips_route_clarification():
    prior = MissionState(turn_count=0)
    turn = extract_mission("Dallas, New York, London, 15 passengers recommend")
    updated = advance_persistent_mission_state(
        prior, turn, "Dallas, New York, London, 15 passengers recommend"
    )
    report = validate_mission_state_consistency(
        prior,
        updated,
        turn,
        "Dallas, New York, London, 15 passengers recommend",
    )
    assert updated.routes
    assert updated.passengers == 15
    assert not report.needs_route_clarification


def test_missing_route_triggers_clarification_not_guess():
    prior = MissionState(passengers=6, turn_count=1)
    turn = extract_mission("What aircraft do you recommend?")
    updated = advance_persistent_mission_state(prior, turn, "What aircraft do you recommend?")
    report = validate_mission_state_consistency(
        prior, updated, turn, "What aircraft do you recommend?"
    )
    assert not updated.routes
    assert report.needs_route_clarification
    assert "origin and destination" in report.clarifying_question.lower()
    assert updated.range_requirement_nm is None


def test_validate_mission_state_consistency_alias():
    assert validateMissionStateConsistency is validate_mission_state_consistency


def test_passenger_update_only_when_explicit():
    prior = MissionState(passengers=8, routes=["Los Angeles -> Miami"], turn_count=1)
    turn = extract_mission("12 passengers same route nonstop")
    updated = advance_persistent_mission_state(prior, turn, "12 passengers same route nonstop")
    assert updated.passengers == 12
    report = validate_mission_state_consistency(
        prior, updated, turn, "12 passengers same route nonstop"
    )
    assert "passengers" in report.updated_fields


def test_constraints_inherit_nonstop_westbound():
    prior = MissionState(
        routes=["San Francisco -> Tokyo"],
        nonstop_required=True,
        westbound=True,
        turn_count=2,
    )
    turn = extract_mission("compare cabin comfort")
    updated = advance_persistent_mission_state(prior, turn, "compare cabin comfort")
    assert updated.nonstop_required is True
    assert updated.westbound is True
    assert updated.routes == prior.routes
