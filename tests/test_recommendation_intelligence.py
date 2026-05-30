"""Recommendation intelligence layer tests."""

from __future__ import annotations

from services.consultant.mission_state import MissionState
from services.recommendation.aircraft_positioning import (
    PositionTier,
    aircraft_position_tier,
    is_prestige_collapse,
)
from services.recommendation.replacement_hierarchy import (
    is_credible_replacement,
    realistic_replacement_candidates,
)
from services.recommendation.procurement_realism import procurement_credibility_score
from services.recommendation.operator_profile_model import infer_operator_profile, OperatorType
from services.mission.center_of_gravity import detect_center_of_gravity
from services.mission.procurement_driver import analyze_procurement_drivers
from services.rendering.prose_renderer_v2 import is_raw_json_leakage, is_incomplete_query


def test_g650_not_replaced_by_pc24():
    mission = MissionState(routes=["LAX-LHR"], passenger_count=10)
    assert is_prestige_collapse("Gulfstream G650ER", "Pilatus PC-24")
    assert not is_credible_replacement("Gulfstream G650ER", "Pilatus PC-24", mission)


def test_realistic_g650_replacements():
    mission = MissionState(routes=["LAX-LHR"], passenger_count=8)
    cands = realistic_replacement_candidates("Gulfstream G650ER", mission)
    assert "Falcon 8X" in cands
    assert "Citation CJ2" not in cands


def test_procurement_score_penalizes_light_on_ulr():
    mission = MissionState(routes=["LAX-TYO"], passenger_count=12)
    score, notes = procurement_credibility_score(
        "Citation CJ2",
        mission,
        query="nonstop Tokyo winter westbound",
    )
    assert score < 0.5
    assert notes


def test_center_of_gravity_domestic():
    cog = detect_center_of_gravity(
        "Most annual utilization is Dallas, Houston, Chicago. Executives occasionally fly to Singapore.",
        MissionState(),
    )
    assert cog.domestic_dominant
    assert cog.episodic_distortion_risk


def test_incomplete_query_blocks():
    assert is_incomplete_query("Leadership insists:")
    assert not is_raw_json_leakage("## Strategic Fleet Analysis")


def test_operator_profile_enterprise():
    op = infer_operator_profile(MissionState(), query="We have 6000 employees and fly to London monthly")
    assert op.operator_type in (OperatorType.GLOBAL_ENTERPRISE, OperatorType.MIDSIZE_PUBLIC)
    assert op.employee_count == 6000
