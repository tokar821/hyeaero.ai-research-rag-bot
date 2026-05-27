"""P1 operational depth — payload realism, dispatch reliability, reserves, telemetry."""

from __future__ import annotations

from services.consultant.mission_state import MissionState
from services.mission.models import MissionProfile, PriorityLevel
from services.mission.route_distance_authority import resolve_route_distance
from services.operational.dispatch_reliability import (
    assess_aircraft_dispatch,
    assess_mission_dispatch_factors,
)
from services.operational.mission_operational_assessment import (
    apply_verdict_cap,
    assess_aircraft_operational,
    build_mission_operational_context,
)
from services.operational.payload_realism import build_mission_payload_profile
from services.operational.reserve_profiles import PlanningMode, compute_reserve_breakdown
from services.telemetry.reasoning_packet import (
    IMMUTABLE_PACKET_KEY,
    attach_reasoning_packet,
    build_reasoning_packet_from_pipeline,
)


def test_payload_ski_modifier_increases_penalty():
    mission = MissionState(passenger_count=8, routes=["TEB → London"])
    light = build_mission_payload_profile(
        mission, query="8 passengers with skis", stage_distance_nm=3100
    )
    plain = build_mission_payload_profile(mission, query="8 passengers", stage_distance_nm=3100)
    assert "ski" in light.modifiers
    assert light.total_payload_lb > plain.total_payload_lb
    assert light.fuel_trade_nm_penalty > plain.fuel_trade_nm_penalty


def test_conservative_reserve_exceeds_aggressive():
    mission = MissionState(passenger_count=6, routes=["TEB → London"])
    payload = build_mission_payload_profile(mission, stage_distance_nm=3100)
    conservative = compute_reserve_breakdown(
        stage_distance_nm=3100,
        payload=payload,
        planning_mode=PlanningMode.CONSERVATIVE,
    )
    aggressive = compute_reserve_breakdown(
        stage_distance_nm=3100,
        payload=payload,
        planning_mode=PlanningMode.AGGRESSIVE,
    )
    assert conservative.total_required_nm > aggressive.total_required_nm


def test_winter_westbound_dispatch_not_reliable_with_tight_margin():
    from services.operational.payload_realism import MissionPayloadProfile
    from services.operational.reserve_profiles import ReserveBreakdown

    factors = assess_mission_dispatch_factors(
        MissionState(
            westbound=True,
            seasonal_constraints="winter",
            nonstop_requirement=True,
        )
    )
    payload = MissionPayloadProfile(
        passengers=8,
        passenger_weight_lb=1600,
        baggage_weight_lb=600,
        modifier_weight_lb=0,
        total_payload_lb=2200,
    )
    reserve = ReserveBreakdown(
        planning_mode="conservative",
        stage_distance_nm=3100,
        base_reserve_nm=200,
        alternate_nm=100,
        holding_nm=30,
        westbound_nm=250,
        payload_required_nm=3100,
        geodesic_extra_nm=0,
        total_required_nm=3680,
    )
    assessment = assess_aircraft_dispatch(
        "Citation Latitude",
        {"practical_nm": 3800, "category": "super-midsize", "dispatch_score": 0.75},
        margin_nm=120,
        reserve=reserve,
        payload=payload,
        factors=factors,
    )
    assert assessment.technically_possible
    assert not assessment.works_reliably
    assert assessment.tech_stop_probability >= 0.25


def test_apply_verdict_cap_never_upgrades():
    assert (
        apply_verdict_cap("PRIMARY RECOMMENDATION", "MISSION-RISKY")
        == "MISSION-RISKY"
    )
    assert (
        apply_verdict_cap("VIABLE WITH COMPROMISES", "MISSION-RISKY")
        == "MISSION-RISKY"
    )
    assert apply_verdict_cap("MISSION-RISKY", "PRIMARY RECOMMENDATION") == "MISSION-RISKY"


def test_mission_operational_context_teb_lon_catalog():
    mission = MissionState(
        passenger_count=8,
        routes=["TEB → London"],
        nonstop_requirement=True,
        westbound=True,
        seasonal_constraints="winter",
    )
    profile = MissionProfile(nonstop_required=True, passengers=8)
    ctx = build_mission_operational_context(
        mission, profile, query="winter westbound TEB to London with skis"
    )
    assert ctx.catalog_peak_nm > 3000
    assert ctx.payload.modifiers
    assert ctx.reserve.total_required_nm > ctx.catalog_peak_nm


def test_citation_latitude_capped_on_winter_westbound_teb_lon():
    mission = MissionState(
        passenger_count=10,
        routes=["TEB → London"],
        westbound=True,
        seasonal_constraints="winter",
        nonstop_requirement=True,
        baggage_priority="high",
    )
    profile = MissionProfile(nonstop_required=True, passengers=10, baggage_priority=PriorityLevel.HIGH)
    ctx = build_mission_operational_context(
        mission, profile, query="10 pax heavy baggage skis winter westbound"
    )
    spec = {
        "practical_nm": 3200,
        "brochure_nm": 3500,
        "category": "super-midsize",
        "dispatch_score": 0.72,
    }
    op = assess_aircraft_operational("Citation Latitude", spec, ctx)
    assert op.recommended_verdict_cap in (
        "MISSION-RISKY",
        "NOT OPERATIONALLY CREDIBLE",
        "VIABLE WITH COMPROMISES",
    )


def test_reasoning_packet_immutable_structure():
    route = resolve_route_distance("TEB → London")
    data_used = {
        "route_distance_authority": [route.to_dict()],
        "mission_operational_context": {
            "corridor_id": "transatlantic_executive",
            "payload": {"passengers": 8, "modifiers": ["ski"]},
            "reserve": {"planning_mode": "conservative", "total_required_nm": 3600},
        },
        "corridor_hard_elimination": {
            "elimination_reasons": {"g280": "ULR band required"},
            "eliminated": ["g280"],
        },
    }
    packet = build_reasoning_packet_from_pipeline(
        data_used=data_used,
        recommendations=[],
        operational_context=data_used["mission_operational_context"],
        aircraft_operational=[
            {
                "model": "G650",
                "dispatch": {
                    "technically_possible": True,
                    "works_reliably": True,
                    "reliability_score": 0.82,
                },
            }
        ],
        elimination_log=[{"aircraft_name": "G280", "stage": "corridor", "reason": "band"}],
    )
    d = packet.to_dict()
    assert d["immutable"] is True
    assert d["route_sources"]
    assert d["corridor_classification"] == "transatlantic_executive"
    assert d["payload_assumptions"]["modifiers"] == ["ski"]
    assert any(e["model"] == "g280" for e in d["eliminations"])
    assert d["confidence"]["route_confidence"] > 0.9

    attach_reasoning_packet(data_used, packet)
    assert IMMUTABLE_PACKET_KEY in data_used
    assert data_used[IMMUTABLE_PACKET_KEY]["immutable"] is True
