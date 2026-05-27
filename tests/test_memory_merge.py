"""
Safe mission memory merge — no route/passenger contamination; stable prefs persist.
"""

from __future__ import annotations

import os

import pytest

from services.memory.mission_memory import (
    MemoryField,
    MissionMemory,
    advance_memory,
    expire_stale_fields,
    merge_memory,
    mission_memory_enabled,
)
from services.mission.memory_bridge import extract_mission_with_memory
from services.mission.models import MissionProfile, OwnershipMode, Route


@pytest.fixture(autouse=True)
def _memory_enabled(monkeypatch):
    monkeypatch.setenv("CONSULTANT_MISSION_MEMORY", "1")


def _profile(**kwargs) -> MissionProfile:
    return MissionProfile(**kwargs)


def _memory(
    *,
    home: str | None = None,
    ownership: str | None = None,
    fleet: list[str] | None = None,
    routes: list | None = None,
    passengers: int | None = None,
) -> MissionMemory:
    mem = MissionMemory(turn_index=1)
    if home:
        mem.home_base = MemoryField(value=home, confidence=0.9, ttl_turns=5, field_key="home_base")
    if ownership:
        mem.ownership_posture = MemoryField(
            value=ownership, confidence=0.91, ttl_turns=5, field_key="ownership_posture"
        )
    if fleet:
        mem.fleet_preferences = [
            MemoryField(value=m, confidence=0.85, ttl_turns=5, field_key="fleet_preferences")
            for m in fleet
        ]
    # Legacy pollution — must never affect merge
    if routes is not None or passengers is not None:
        d = mem.to_dict()
        if routes is not None:
            d["routes"] = routes
        if passengers is not None:
            d["passengers"] = passengers
        return MissionMemory.from_dict(d)
    return mem


def test_no_stale_route_carryover():
    mem = _memory(home="Teterboro", ownership="fractional", routes=["Paris -> Dubai"])
    turn = _profile(
        routes=[Route(origin="Miami", destination="Aspen")],
        passengers=6,
    )
    merged = merge_memory(turn, mem)
    assert len(merged.routes) == 1
    assert merged.routes[0].label() == "Miami -> Aspen"
    assert merged.passengers == 6


def test_no_passenger_contamination_from_memory():
    mem = _memory(passengers=12, ownership="charter")
    turn = _profile(passengers=4, routes=[Route(origin="Boston", destination="Chicago")])
    merged = merge_memory(turn, mem)
    assert merged.passengers == 4


def test_ownership_preference_persists_when_turn_silent():
    mem = _memory(ownership="fractional")
    turn = _profile(routes=[Route(origin="New York", destination="London")])
    merged = merge_memory(turn, mem)
    assert merged.ownership_interest == OwnershipMode.FRACTIONAL


def test_current_turn_ownership_wins():
    mem = _memory(ownership="fractional")
    turn = _profile(ownership_interest=OwnershipMode.CHARTER)
    merged = merge_memory(turn, mem)
    assert merged.ownership_interest == OwnershipMode.CHARTER


def test_home_base_persists_when_turn_omits():
    mem = _memory(home="Teterboro")
    turn = _profile(routes=[Route(origin="Boston", destination="Miami")])
    merged = merge_memory(turn, mem)
    assert merged.home_base == "Teterboro"


def test_current_turn_home_base_wins():
    mem = _memory(home="Teterboro")
    turn = _profile(home_base="Van Nuys")
    merged = merge_memory(turn, mem)
    assert merged.home_base == "Van Nuys"


def test_fleet_preferences_merge_without_override():
    mem = _memory(fleet=["Gulfstream G650"])
    turn = _profile(fleet_preferences=["Citation X"])
    merged = merge_memory(turn, mem)
    assert "Citation X" in merged.fleet_preferences
    assert "Gulfstream G650" in merged.fleet_preferences


def test_memory_optional_when_disabled(monkeypatch):
    monkeypatch.setenv("CONSULTANT_MISSION_MEMORY", "0")
    mem = _memory(ownership="fractional", home="Teterboro")
    turn = _profile()
    merged = merge_memory(turn, mem)
    assert merged.ownership_interest is None
    assert not merged.home_base


def test_advance_memory_never_stores_routes_or_passengers():
    turn = _profile(
        routes=[Route(origin="Dallas", destination="Denver")],
        passengers=8,
        ownership_interest=OwnershipMode.FRACTIONAL,
        home_base="Dallas",
        fleet_preferences=["Challenger 350"],
    )
    next_mem = advance_memory(None, turn, user_message="8 passengers Dallas to Denver fractional")
    blob = next_mem.to_dict()
    assert "routes" not in blob or blob.get("routes") is None
    assert "passengers" not in blob or blob.get("passengers") is None
    assert blob["ownership_posture"]["value"] == "fractional"
    assert blob["home_base"]["value"] == "Dallas"
    assert len(blob["fleet_preferences"]) >= 1


def test_ttl_expires_ownership():
    mem = MissionMemory(
        turn_index=2,
            ownership_posture=MemoryField(
                value="fractional", confidence=0.91, ttl_turns=0, field_key="ownership_posture"
            ),
    )
    expired = expire_stale_fields(mem)
    assert expired.ownership_posture is None


def test_memory_field_serialization():
    fld = MemoryField(value="fractional", confidence=0.91, ttl_turns=5, field_key="ownership_posture")
    restored = MemoryField.from_dict(fld.to_dict())
    assert restored is not None
    assert restored.value == "fractional"
    assert restored.confidence == pytest.approx(0.91)
    assert restored.ttl_turns == 5


def test_extract_mission_with_memory_two_turn_simulation():
    # Turn 1: establish fractional + home
    _, merged1, mem1 = extract_mission_with_memory(
        "We are based in Teterboro and prefer fractional ownership for US trips",
        memory=MissionMemory(),
    )
    assert merged1.ownership_interest == OwnershipMode.FRACTIONAL
    assert merged1.home_base == "Teterboro"

    # Turn 2: new route/pax only — must not inherit prior route/pax from turn 1 text
    turn2, merged2, _ = extract_mission_with_memory(
        "6 passengers from Miami to Aspen nonstop",
        conversation_state={"mission_memory": mem1.to_dict()},
    )
    assert turn2.passengers == 6
    assert any("Miami" in r.label() for r in turn2.routes)
    assert merged2.passengers == 6
    assert merged2.ownership_interest == OwnershipMode.FRACTIONAL
    assert merged2.home_base == "Teterboro"
    assert not any("Teterboro" in r.label() for r in merged2.routes)


def test_temporary_priorities_not_in_memory():
    from services.mission.models import PriorityLevel

    turn = _profile(cabin_priority=PriorityLevel.HIGH, nonstop_required=True)
    next_mem = advance_memory(None, turn)
    assert next_mem.to_dict().get("cabin_priority") is None
    assert next_mem.to_dict().get("nonstop_required") is None


def test_mission_memory_enabled_default():
    assert mission_memory_enabled()
