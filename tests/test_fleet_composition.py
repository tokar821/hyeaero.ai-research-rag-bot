"""P2 — multi-domain operational composition (elimination-driven)."""

from __future__ import annotations

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation, RecommendationExplanation
from services.elimination.elimination_invariant import collect_eliminated_models
from services.fleet.fleet_composition import (
    MissionSegmentRole,
    build_fleet_composition_plan,
    detect_multi_aircraft_mission,
    format_fleet_composition_block,
    merge_fleet_into_recommendations,
)
from services.fleet.fleet_domain_analysis import (
    SegmentationTrigger,
    analyze_multi_domain_operational_problem,
)
from services.fleet.fleet_invariant import assert_fleet_invariants
from services.mission.models import MissionProfile


def _rec(model: str, verdict: str = "VIABLE WITH COMPROMISES") -> AircraftRecommendation:
    return AircraftRecommendation(
        model=model,
        category="ultra-long",
        total_score=0.7,
        confidence=0.8,
        rank=1,
        avoid=False,
        fit="Good Fit",
        fit_verdict=verdict,
        explanation=RecommendationExplanation(summary=""),
    )


def test_structural_invalid_no_universal_survivor():
    profile = MissionProfile(
        mountain_airports=True,
        international_ops=True,
        nonstop_required=True,
    )
    mission = MissionState(
        routes=["TEB → London", "KASE → KTEX"],
        mountain_airport_requirement=True,
        passenger_count=6,
    )
    pool = list(
        {
            "Gulfstream G650",
            "Pilatus PC-24",
            "Citation Latitude",
            "Global 7500",
            "Learjet 75",
        }
    )
    analysis = analyze_multi_domain_operational_problem(profile, mission, pool)
    assert len(analysis.domains) >= 2
    assert analysis.single_aircraft_structurally_invalid
    assert analysis.multi_domain_required
    assert analysis.trigger == SegmentationTrigger.ELIMINATION_FAILURE
    assert not analysis.universal_survivors


def test_preference_alone_does_not_trigger_multi_domain():
    profile = MissionProfile(fleet_preferences=["Gulfstream G650", "Pilatus PC-24"])
    mission = MissionState(routes=["TEB → Miami"], passenger_count=6)
    assert not detect_multi_aircraft_mission(profile, mission, query="")


def test_fleet_plan_assigns_ulr_and_mountain_domains():
    profile = MissionProfile(
        mountain_airports=True,
        international_ops=True,
        nonstop_required=True,
    )
    mission = MissionState(
        routes=["TEB → London", "KASE → KTEX"],
        mountain_airport_requirement=True,
        passenger_count=6,
    )
    recs = [
        _rec("Gulfstream G650", "PRIMARY RECOMMENDATION"),
        _rec("Pilatus PC-24"),
        _rec("Citation Latitude"),
    ]
    plan = build_fleet_composition_plan(
        profile, mission, recs, query="transatlantic and Aspen"
    )
    assert plan.multi_aircraft_required
    assert plan.trigger == SegmentationTrigger.ELIMINATION_FAILURE.value
    assert plan.single_aircraft_structurally_invalid
    roles = {a.role for a in plan.assignments}
    assert MissionSegmentRole.ULR_INTERNATIONAL in roles
    assert MissionSegmentRole.MOUNTAIN_FIELD in roles
    ulr = next(a for a in plan.assignments if a.role == MissionSegmentRole.ULR_INTERNATIONAL)
    mtn = next(a for a in plan.assignments if a.role == MissionSegmentRole.MOUNTAIN_FIELD)
    assert ulr.primary_model in ("Gulfstream G650", "Gulfstream G650ER", "Global 7500", "Global 6500")
    assert mtn.primary_model in ("Pilatus PC-24", "Pilatus PC-12", "Learjet 75", "Citation CJ2")
    assert ulr.primary_model != mtn.primary_model
    assert ulr.domain_feasible and mtn.domain_feasible


def test_domain_traces_include_lineage():
    profile = MissionProfile(mountain_airports=True, international_ops=True)
    mission = MissionState(
        routes=["TEB → London", "Aspen → Telluride"],
        mountain_airport_requirement=True,
    )
    plan = build_fleet_composition_plan(
        profile, mission, [_rec("Gulfstream G650")], query=""
    )
    assert plan.domain_traces
    for tr in plan.domain_traces:
        assert "domain" in tr
        assert "constraint_triggers" in tr
        assert "elimination_lineage" in tr


def test_fleet_invariant_no_globally_eliminated_assignees():
    profile = MissionProfile(mountain_airports=True, international_ops=True)
    mission = MissionState(
        routes=["TEB → London", "KASE → KTEX"],
        mountain_airport_requirement=True,
    )
    data_used = {
        "corridor_hard_elimination": {
            "eliminated": ["Citation Latitude"],
            "elimination_reasons": {"Citation Latitude": "corridor"},
        }
    }
    plan = build_fleet_composition_plan(
        profile,
        mission,
        [_rec("Gulfstream G650"), _rec("Pilatus PC-24")],
        data_used=data_used,
    )
    assert_fleet_invariants(plan)
    for a in plan.assignments:
        if a.primary_model:
            assert a.primary_model.lower().find("latitude") < 0


def test_one_aircraft_only_impossible_mission():
    profile = MissionProfile(mountain_airports=True, international_ops=True)
    mission = MissionState(
        routes=["TEB → London", "KASE → KTEX"],
        mountain_airport_requirement=True,
    )
    analysis = analyze_multi_domain_operational_problem(
        profile,
        mission,
        list({"Gulfstream G650", "Pilatus PC-24"}),
        query="one aircraft only for everything",
    )
    assert analysis.multi_domain_required
    assert analysis.trigger == SegmentationTrigger.IMPOSSIBLE_SINGLE_AIRCRAFT_CONSTRAINT


def test_format_block_states_structural_invalidity():
    profile = MissionProfile(mountain_airports=True, international_ops=True)
    mission = MissionState(routes=["TEB → London", "KASE → KTEX"], mountain_airport_requirement=True)
    plan = build_fleet_composition_plan(profile, mission, [_rec("G650")])
    text = format_fleet_composition_block(plan)
    assert "structurally invalid" in text.lower() or "Multi-domain" in text


def test_merge_fleet_orders_domain_feasible_primaries():
    profile = MissionProfile(mountain_airports=True, international_ops=True)
    mission = MissionState(
        routes=["TEB → London", "Aspen → Telluride"],
        mountain_airport_requirement=True,
    )
    recs = [_rec("Citation Latitude"), _rec("Gulfstream G650"), _rec("Pilatus PC-24")]
    plan = build_fleet_composition_plan(profile, mission, recs)
    merged = merge_fleet_into_recommendations(recs, plan)
    if plan.multi_aircraft_required and plan.presented_models():
        assert merged[0].model in plan.presented_models()
