"""Pre-ranking representation — distribution, route graph, industrial, governance."""

from __future__ import annotations

from services.consultant.mission_state import MissionState
from services.mission.industrial_airport_classifier import classify_industrial_airports
from services.mission.mission_extractor import extract_mission
from services.mission.mission_governance import resolve_mission_governance
from services.mission.mission_understanding_engine import MissionUnderstandingPacket
from services.mission.passenger_distribution import extract_passenger_distribution
from services.mission.pre_ranking_representation import apply_pre_ranking_representation
from services.mission.route_graph_representation import build_route_graph, infer_continuation_legs


def test_passenger_distribution_3_to_14():
    q = (
        "We move teams ranging from 3 executives to 14-person deal groups between "
        "Chicago, New York, and Europe with cargo for equipment."
    )
    dist = extract_passenger_distribution(q)
    assert dist.min_pax == 3
    assert dist.max_pax == 14
    assert dist.planning_load == 14
    assert dist.cargo_required


def test_abu_dhabi_continuation_in_route_graph():
    q = (
        "We fly Boston, Chicago, and San Francisco frequently with small teams, but twice "
        "a month the founder flies nonstop to Abu Dhabi."
    )
    profile = extract_mission(q)
    graph = build_route_graph(q, profile)
    labels = [r.label() for r in graph.all_legs()]
    assert any("Abu Dhabi" in lbl for lbl in labels)
    assert any(n.canonical == "Abu Dhabi" for n in graph.nodes)


def test_industrial_gravel_classifier():
    q = (
        "Aircraft often land on short gravel strips near Calgary oil fields, with quarterly "
        "nonstop London flights."
    )
    industrial = classify_industrial_airports(q)
    assert industrial.active
    assert any(c.value in ("gravel", "oil_field") for c in industrial.classes)


def test_governance_ceo_defer_global_ranking():
    q = (
        "Our CEO requires nonstop New York–Dubai capability. However, 80% of flights are "
        "2–3 hour domestic hops with 4 executives."
    )
    profile = extract_mission(q)
    pkt = MissionUnderstandingPacket()
    gov = resolve_mission_governance(q, profile, pkt)
    assert gov.ceo_ulr_mandate
    assert gov.domestic_utilization_dominant
    assert gov.defer_global_aircraft_ranking


def test_pre_ranking_pipeline_merge():
    q = (
        "We move teams ranging from 3 executives to 14-person deal groups between "
        "Chicago and New York."
    )
    profile = extract_mission(q)
    mission = MissionState()
    pkt = MissionUnderstandingPacket()
    profile, mission, pkt = apply_pre_ranking_representation(q, profile, mission, pkt, {})
    assert profile.passenger_distribution
    assert profile.passenger_distribution.max_pax == 14
    assert profile.passengers == 14
