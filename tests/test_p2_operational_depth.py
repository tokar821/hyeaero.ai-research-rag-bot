"""P2 — airport DB expansion, wind realism, ownership simulator, visual memory scope."""

from __future__ import annotations

from services.orchestration.ownership_simulator import simulate_ownership_economics
from services.airport.airport_database import (
    mission_airport_profiles,
    profile_from_icao,
    resolve_airports_in_text,
)
from services.consultant.mission_state import MissionState
from services.memory.visual_scope import clear_visual_memory_patch, mission_route_changed
from services.operational.mission_operational_assessment import build_mission_operational_context
from services.operational.wind_realism import compute_wind_adjustment
from services.mission.models import MissionProfile


def test_airport_database_teb_london():
    aps = resolve_airports_in_text("Teterboro to London")
    icaos = {a.icao for a in aps}
    assert "KTEB" in icaos
    assert "EGLF" in icaos or "EGLL" in icaos


def test_airport_database_icao_count():
    assert len(mission_airport_profiles(["Aspen to Telluride"])) >= 2
    assert profile_from_icao("KASE") is not None


def test_wind_winter_westbound_penalty():
    mission = MissionState(
        routes=["London to New York"],
        westbound=True,
        seasonal_constraints="winter",
    )
    wind = compute_wind_adjustment(mission, stage_distance_nm=3100, route_label="London to New York")
    assert wind.total_penalty_nm > 200
    assert wind.westbound_penalty_nm > 0


def test_operational_context_includes_wind():
    mission = MissionState(
        passenger_count=8,
        routes=["TEB → London"],
        westbound=True,
        seasonal_constraints="winter",
        nonstop_requirement=True,
    )
    profile = MissionProfile(nonstop_required=True, westbound_sensitive=True)
    ctx = build_mission_operational_context(mission, profile, query="8 pax winter westbound nonstop")
    assert ctx.wind is not None
    assert ctx.wind.total_penalty_nm > 0
    assert ctx.to_dict().get("wind")


def test_ownership_simulator_burdened_hourly():
    sim = simulate_ownership_economics(
        "I fly 250 hours a year — fractional vs full ownership for Challenger 350",
        anchor_model="Challenger 350",
    )
    assert sim.annual_hours == 250
    assert sim.all_in_hour_usd > sim.variable_cost_per_hour_usd
    assert "Ownership Economics" in "\n".join(sim.lines)


def test_visual_memory_cleared_on_route_change():
    state = {
        "conversation_memory": {
            "last_mission_routes": ["TEB → Miami"],
            "active_aircraft": "Citation Latitude",
            "active_tail": "N807JS",
        }
    }
    assert mission_route_changed(["TEB → Miami"], ["Aspen → London"])
    patch = clear_visual_memory_patch(state, new_routes=["Aspen → London"])
    assert patch.get("visual_memory_cleared")
    assert patch.get("active_tail") is None
    assert patch.get("active_aircraft") is None
