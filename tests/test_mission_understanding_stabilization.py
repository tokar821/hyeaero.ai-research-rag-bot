"""Mission understanding stabilization — passenger realism, Europe routes, priorities."""

from __future__ import annotations

from services.consultant.mission_state import MissionState
from services.mission.mission_extractor import extract_mission
from services.mission.mission_understanding_engine import (
    apply_understanding_to_profile,
    build_mission_understanding,
)
from services.mission.models import MissionProfile
from services.mission.route_extractor import extract_routes, resolve_place
from services.recommendation.aircraft_category_gating import (
    GatedMissionCategory,
    determine_gated_mission_category,
)


def test_moscaw_typo_resolves_to_moscow():
    place, conf = resolve_place("Moscaw")
    assert place is not None
    assert place.canonical == "Moscow"
    assert conf > 0.7


def test_nyc_berlin_moscow_routes_extracted():
    text = "NYC to Berlin and Moscow, 6 passengers, runway flexibility and operating cost"
    routes = extract_routes(text)
    labels = {r.route.label() for r in routes}
    assert "New York -> Berlin" in labels or any("Berlin" in l for l in labels)
    assert any("Moscow" in l for l in labels)


def test_four_pax_transatlantic_caps_planning_band():
    profile = extract_mission(
        "Dallas and New York and London, 4 passengers, twice monthly — what structure?"
    )
    mission = MissionState(passenger_count=4, routes=profile.route_labels())
    packet = build_mission_understanding(
        "Dallas and New York and London, 4 passengers, twice monthly",
        profile,
        mission,
    )
    assert packet.inferred_constraints.get("cabin_utilization_modest") or packet.inferred_constraints.get(
        "planning_band_ceiling"
    )
    assert packet.corridor_type != "transatlantic_ulr" or packet.inferred_constraints.get(
        "planning_band_ceiling"
    ) == "super_midsize"
    prof = apply_understanding_to_profile(MissionProfile(), packet)
    assert prof.planning_band_ceiling == "super_midsize"


def test_europe_cost_runway_sets_international_jet_floor():
    profile = extract_mission(
        "NYC to Berlin, 6 passengers, runway flexibility more than luxury, operating cost priority"
    )
    mission = MissionState(passenger_count=6, routes=profile.route_labels())
    packet = build_mission_understanding(
        "NYC to Berlin, runway flexibility and operating cost more than luxury, 6 pax",
        profile,
        mission,
    )
    assert packet.inferred_constraints.get("international_jet_floor") or packet.inferred_constraints.get(
        "balanced_cost_dispatch"
    )
    prof = apply_understanding_to_profile(profile, packet)
    gate = determine_gated_mission_category(mission, mission_profile=prof)
    assert gate.category != GatedMissionCategory.LIGHT_JET
