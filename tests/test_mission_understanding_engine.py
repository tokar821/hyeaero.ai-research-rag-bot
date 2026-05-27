"""Mission Understanding Engine v2 — latent inference before ranking."""

from __future__ import annotations

from services.consultant.mission_state import MissionState
from services.mission.mission_extractor import extract_mission
from services.mission.mission_understanding_engine import (
    build_mission_understanding,
    format_understanding_first_advisory,
)
from services.mission.adapters import mission_profile_to_state


def test_enterprise_europe_infer_corporate_shuttle():
    q = (
        "We have 1000 employees, 100+ million revenue, about 50 Europe trips a year "
        "with 12 people — what aircraft program makes sense?"
    )
    profile = extract_mission("12 executives Europe trips")
    mission = mission_profile_to_state(profile)
    pkt = build_mission_understanding(q, profile, mission)
    assert pkt.inferred_constraints.get("enterprise_employees") == 1000
    assert pkt.travel_pattern in ("transatlantic_executive", "executive_shuttle")
    assert pkt.dispatch_priority == "high"
    assert pkt.nonstop_priority == "high"
    assert pkt.ownership_profile == "corporate_shuttle_candidate"
    # Recommend only after route evidence + confidence; this scenario has no city pair.
    assert not pkt.recommend_aircraft
    assert pkt.fallback_operational_band


def test_caribbean_runway_over_luxury_environment():
    q = "Miami Caribbean South America 8 passengers runway flexibility over luxury"
    profile = extract_mission(q)
    mission = mission_profile_to_state(profile)
    pkt = build_mission_understanding(q, profile, mission)
    assert pkt.inferred_constraints.get("runway_over_cabin") or pkt.runway_complexity == "high"
    assert any("island" in e.lower() or "tropical" in e.lower() for e in pkt.operational_environment)
    assert pkt.corridor_type == "caribbean_regional"
    assert pkt.fallback_operational_band


def test_multi_leg_pacific_europe_fallback_band():
    q = "SFO to Tokyo and London nonstop westbound winter 8 passengers"
    profile = extract_mission(q)
    mission = mission_profile_to_state(profile)
    pkt = build_mission_understanding(q, profile, mission)
    assert pkt.inferred_constraints.get("dual_use_or_multi_leg") or len(profile.routes) >= 1
    assert pkt.fallback_operational_band
    text = format_understanding_first_advisory(mission, pkt, recommendations=[])
    # With low confidence (no catalog sizing), we should degrade to class-band guidance.
    assert "Aircraft Class Band" in text
    assert "empty" not in text.lower() or "class band" in text.lower()


def test_understanding_first_advisory_never_blank_options_header_only():
    mission = MissionState(routes=["TEB → London"], passenger_count=8)
    pkt = build_mission_understanding(
        "8 pax TEB London winter westbound",
        extract_mission("8 pax TEB London winter westbound"),
        mission,
    )
    body = format_understanding_first_advisory(mission, pkt, recommendations=[], query="8 pax TEB London winter westbound")
    assert "Mission Fit" in body
    assert ("Aircraft Options" in body) or ("Aircraft Class Band" in body)
    assert len(body) > 120


def test_london_aspen_portfolio_synthesis():
    q = "We need London nonstop capability, but we also spend a lot of time in Aspen."
    profile = extract_mission(q)
    mission = mission_profile_to_state(profile)
    pkt = build_mission_understanding(q, profile, mission)
    assert pkt.inferred_constraints.get("incompatible_mission_bands")
    body = format_understanding_first_advisory(mission, pkt, recommendations=[], query=q)
    assert "Fleet Structure" in body
    assert "multi-aircraft" in body.lower() or "incompatible" in body.lower()
    assert "Transatlantic" in body
    assert "Mountain" in body


def test_pe_synthesis_ownership_and_portfolio():
    q = (
        "We're a private equity group. Teams move between New York, Dallas, London, and occasionally Dubai. "
        "Senior partners hate fuel stops, but we also visit smaller industrial airports domestically. "
        "We currently charter around 300 hours annually and are debating ownership."
    )
    profile = extract_mission(q)
    mission = mission_profile_to_state(profile)
    pkt = build_mission_understanding(q, profile, mission)
    assert pkt.inferred_constraints.get("ownership_economics_relevant")
    assert pkt.inferred_constraints.get("annual_charter_hours") == 300
    body = format_understanding_first_advisory(mission, pkt, recommendations=[], query=q)
    assert "Ownership Economics" in body
    assert "300" in body
    assert "Fleet Structure" in body or "multi-aircraft" in body.lower()


def test_single_aircraft_resistance():
    q = "We want one aircraft for London, Aspen, Caribbean islands, and Tokyo."
    profile = extract_mission(q)
    mission = mission_profile_to_state(profile)
    pkt = build_mission_understanding(q, profile, mission)
    assert pkt.inferred_constraints.get("single_aircraft_request")
    body = format_understanding_first_advisory(mission, pkt, recommendations=[], query=q)
    assert "single platform" in body.lower() or "one aircraft" in body.lower()
    assert "incompatible" in body.lower() or "multi-aircraft" in body.lower()
