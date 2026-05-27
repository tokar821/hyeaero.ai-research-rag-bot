"""Session mission memory — cross-turn persistence and override rules."""

from services.mission import extract_mission
from services.mission.memory_bridge import extract_mission_with_memory
from services.mission.models import OwnershipMode
from services.state.mission_state import (
    MissionState,
    advance_persistent_mission_state,
    load_persistent_mission_state,
    sync_persistent_mission_state,
)
from services.state.session_mission_memory import (
    detect_session_field_overrides,
    merge_turn_with_session,
)


def test_session_inherits_passengers_budget_on_follow_up():
    du: dict = {}
    sync_persistent_mission_state(
        "10 passengers LA to Miami nonstop $10M",
        data_used=du,
    )
    prior = load_persistent_mission_state(data_used=du)
    turn = extract_mission("recommend again for this trip")
    merged = merge_turn_with_session(turn, prior, "recommend again for this trip")
    assert merged.profile.passengers == 10
    assert merged.profile.budget_usd_mid == 10_000_000
    assert "passengers" in merged.inherited_fields
    assert "budget_usd" in merged.inherited_fields


def test_explicit_pax_override_blocks_session_carryover():
    prior = MissionState(passengers=10, budget_usd=10_000_000, routes=["Los Angeles -> Miami"])
    turn = extract_mission("4 passengers same route")
    overrides = detect_session_field_overrides("4 passengers same route", turn)
    assert "passengers" in overrides
    merged = merge_turn_with_session(turn, prior, "4 passengers same route")
    assert merged.profile.passengers == 4
    updated = advance_persistent_mission_state(prior, turn, "4 passengers same route")
    assert updated.passengers == 4


def test_ownership_and_home_persist_via_memory_bridge():
    _, merged1, mem1 = extract_mission_with_memory(
        "Based in Teterboro, leaning fractional ownership",
        memory=None,
    )
    assert merged1.home_base == "Teterboro"
    assert merged1.ownership_interest == OwnershipMode.FRACTIONAL

    du = {"mission_memory": mem1.to_dict()}
    sync_persistent_mission_state(
        "8 passengers Miami to Aspen nonstop",
        conversation_state=du,
        data_used=du,
    )
    _, merged2, _ = extract_mission_with_memory(
        "6 passengers Miami to Aspen nonstop",
        conversation_state=du,
        data_used=du,
    )
    assert merged2.passengers == 6
    assert merged2.home_base == "Teterboro"
    assert merged2.ownership_interest == OwnershipMode.FRACTIONAL


def test_runway_priority_inherits_when_turn_silent():
    prior = MissionState(passengers=6, routes=["Dallas -> Aspen"])
    prior.priorities.runway = "high"
    turn = extract_mission("recommend aircraft for this trip")
    merged = merge_turn_with_session(turn, prior, "recommend aircraft for this trip")
    assert "runway" in merged.inherited_fields
    assert merged.profile.runway_priority.value == "high"


def test_two_turn_sync_carries_session_fields():
    du: dict = {}
    _, _, _ = sync_persistent_mission_state(
        "8 passengers based in Dallas budget $12M LA to Miami nonstop",
        data_used=du,
    )
    state, profile, report = sync_persistent_mission_state(
        "what about runway flexibility?",
        data_used=du,
    )
    assert state.passengers == 8
    assert state.budget_usd == 12_000_000
    assert "passengers" in report.inherited_fields
    assert profile.passengers == 8
