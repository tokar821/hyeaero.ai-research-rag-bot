"""Place index — accent-insensitive capture."""

from __future__ import annotations

from services.mission.mission_extractor import extract_mission
from services.mission.mission_place_index import city_captured, places_captured_from_mission
from services.mission.mission_understanding_engine import (
    MissionUnderstandingPacket,
    _collect_regional_environment,
)


def test_sao_paulo_accent_captured():
    q = (
        "We transport between 4 and 16 people between Houston, São Paulo, Lagos, and Frankfurt."
    )
    profile = extract_mission(q)
    captured = places_captured_from_mission(profile, q)
    assert any("Sao Paulo" in p for p in captured)
    assert city_captured("são paulo", profile, q)
    assert city_captured("sao paulo", profile, q)


def test_multi_continent_not_caribbean_band():
    q = (
        "We transport between 4 and 16 people depending on mission between Houston, "
        "São Paulo, Lagos, and Frankfurt. Cargo space matters more than cabin."
    )
    profile = extract_mission(q)
    pkt = MissionUnderstandingPacket()
    _collect_regional_environment(q, profile, pkt)
    assert "Caribbean executive regional jet band" not in pkt.fallback_operational_band
    assert pkt.inferred_constraints.get("island_ops") is not True
