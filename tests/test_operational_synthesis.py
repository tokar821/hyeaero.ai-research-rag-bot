"""Operational synthesis enrichment tests."""

from __future__ import annotations

from services.consultant.mission_state import MissionState
from services.mission.mission_extractor import extract_mission
from services.mission.mission_understanding_engine import build_mission_understanding
from services.mission.adapters import mission_profile_to_state
from services.mission.operational_synthesis import enrich_operational_synthesis


def test_ulr_westbound_synthesis_includes_dispatch_realism():
    q = "SFO to Tokyo and London nonstop westbound winter 8 passengers"
    profile = extract_mission(q)
    mission = mission_profile_to_state(profile)
    pkt = build_mission_understanding(q, profile, mission, use_llm=False)
    text = enrich_operational_synthesis(pkt, mission, profile, query=q)
    low = text.lower()
    assert "ultra-long-range" in low or "ulr" in low
    assert "westbound" in low or "winter" in low
    assert "g650" in low or "7500" in low or "dispatch" in low


def test_executive_caribbean_not_utility():
    q = "Miami Caribbean South America 8 passengers runway flexibility over luxury"
    profile = extract_mission(q)
    mission = mission_profile_to_state(profile)
    pkt = build_mission_understanding(q, profile, mission, use_llm=False)
    assert pkt.inferred_constraints.get("executive_travel_profile")
    assert pkt.inferred_constraints.get("minimum_jet_cabin_floor")
    assert not mission.mountain_airport_requirement
    text = enrich_operational_synthesis(pkt, mission, profile, query=q)
    assert "executive" in text.lower() or "turboprop" in text.lower()
