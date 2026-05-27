"""Route feasibility engine tests."""

from services.consultant.mission_state import MissionState, build_mission_from_current_turn
from services.consultant.route_feasibility import assess_route_for_aircraft, assess_mission_routes


def test_la_miami_reliably_nonstop_super_midsize():
    mission = build_mission_from_current_turn("LA to Miami 8 pax nonstop")
    a = assess_route_for_aircraft(
        route_label="Los Angeles → Miami",
        aircraft_practical_nm=2700,
        aircraft_brochure_nm=3200,
        mission=mission,
        passenger_count=8,
    )
    assert a.classification in ("reliably_nonstop", "practical_restricted")
    assert a.distance_nm > 1500


def test_westbound_winter_penalty_reduces_confidence():
    mission = build_mission_from_current_turn("West coast to Europe westbound winter")
    a = assess_route_for_aircraft(
        route_label="West Coast → Europe",
        aircraft_practical_nm=5600,
        aircraft_brochure_nm=6450,
        mission=mission,
    )
    assert a.westbound_penalty_nm > 0
    assert a.seasonal_note or a.caveats


def test_light_jet_not_feasible_transatlantic():
    mission = build_mission_from_current_turn("SFO to Paris nonstop")
    a = assess_route_for_aircraft(
        route_label="San Francisco → Paris",
        aircraft_practical_nm=1650,
        aircraft_brochure_nm=2000,
        mission=mission,
    )
    assert a.classification in ("not_feasible", "brochure_capable")
    assert not a.reliably_nonstop


def test_mountain_airport_adds_payload_penalty():
    mission = build_mission_from_current_turn("Dallas to Aspen hot and high")
    mission.mountain_airport_requirement = True
    assessments = assess_mission_routes(
        mission,
        aircraft_practical_nm=2700,
        aircraft_brochure_nm=3200,
    )
    assert assessments
    assert assessments[0].payload_penalty_note or assessments[0].caveats
