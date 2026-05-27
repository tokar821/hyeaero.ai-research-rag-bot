"""Mission Understanding Phase 2 — semantic model stabilization."""

from __future__ import annotations

from services.consultant.mission_state import MissionState
from services.mission.mission_extractor import extract_mission
from services.mission.mission_semantic_model import (
    DOMAIN_ARCTIC,
    DOMAIN_EXECUTIVE,
    DOMAIN_INDUSTRIAL,
    DOMAIN_MOUNTAIN,
    build_mission_semantic_model,
    stabilize_mission_semantics,
)
from services.mission.mission_understanding_engine import build_mission_understanding
from services.mission.models import MissionProfile


def test_arctic_is_hard_domain_with_extreme_weight():
    q = (
        "We operate oil extraction sites in Nunavut and Northern Canada gravel strips, but also "
        "fly executives Houston to London monthly. Winter dispatch failures keep happening."
    )
    profile = extract_mission(q)
    mission = MissionState(routes=profile.route_labels())
    model = build_mission_semantic_model(q, profile, mission)

    assert DOMAIN_ARCTIC in model.mission_domains
    assert DOMAIN_ARCTIC in model.hard_domains
    assert "arctic_hard_domain" in model.constraint_flags
    assert "arctic_extreme_constraint_multiplier" in model.constraint_flags
    assert "winter_dispatch_binding" in model.constraint_flags
    assert model.domain_weights[model.mission_domains.index(DOMAIN_ARCTIC)] >= 0.95


def test_industrial_is_hard_domain():
    q = "We move equipment from Permian Basin drilling fields and also send executives to Paris."
    profile = extract_mission(q)
    mission = MissionState(routes=profile.route_labels())
    model = build_mission_semantic_model(q, profile, mission)

    assert DOMAIN_INDUSTRIAL in model.hard_domains
    assert "industrial_hard_domain" in model.constraint_flags
    assert "field_access_over_cabin" in model.constraint_flags


def test_domain_weights_are_not_uniform():
    q = (
        "We operate between Los Angeles, Tokyo, and Singapore, but also move teams into "
        "Aspen ski regions in winter."
    )
    profile = extract_mission(q)
    mission = MissionState(routes=profile.route_labels())
    model = build_mission_semantic_model(q, profile, mission)

    assert len(set(model.domain_weights)) > 1
    if DOMAIN_MOUNTAIN in model.mission_domains and DOMAIN_EXECUTIVE in model.mission_domains:
        mtn_w = model.domain_weights[model.mission_domains.index(DOMAIN_MOUNTAIN)]
        exec_w = model.domain_weights[model.mission_domains.index(DOMAIN_EXECUTIVE)]
        assert exec_w > mtn_w


def test_single_aircraft_invalid_with_hard_domains():
    q = (
        "We operate across Houston oil fields, Northern Canada Arctic gravel strips, and London "
        "executive HQ. We previously tried a single aircraft and had winter failures."
    )
    profile = extract_mission(q)
    mission = MissionState(routes=profile.route_labels())
    model = build_mission_semantic_model(q, profile, mission)

    assert len(model.hard_domains) >= 2
    assert "single_aircraft_universal_hard_domain_coverage" in model.invalid_interpretations
    assert "single_aircraft_preference_over_hard_domain_conflict" in model.invalid_interpretations


def test_semantic_model_attached_to_understanding_packet():
    q = "We fly between Arctic drilling sites in Northern Canada and Calgary to Frankfurt."
    profile = extract_mission(q)
    mission = MissionState(routes=profile.route_labels())
    data_used: dict = {}
    packet = build_mission_understanding(q, profile, mission, data_used=data_used)

    assert "mission_semantic_model" in data_used
    assert packet.inferred_constraints.get("arctic_hard_domain")
    assert packet.inferred_constraints.get("mission_semantic_domains")
    assert packet.inferred_constraints.get("semantic_invalid_interpretations") is not None
    assert packet.understanding_notes


def test_stabilize_does_not_add_routes():
    q = "Houston to London, 8 passengers"
    before = extract_mission(q)
    routes_before = before.route_labels()
    mission = MissionState(routes=routes_before)
    profile = MissionProfile(routes=list(before.routes))
    stabilize_mission_semantics(q, profile, mission)
    assert profile.route_labels() == routes_before
