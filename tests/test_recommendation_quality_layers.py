"""Tests for tier recovery, multi-factor ranking, and broker response renderer."""

from __future__ import annotations

import re

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.consultant.broker_response_renderer import format_broker_recommendation_response
from services.consultant.dispatch_conflict_renderer import format_dispatch_conflict_block
from services.consultant.comparative_analysis_renderer import format_comparative_analysis_table
from services.mission.mission_understanding_engine import MissionUnderstandingPacket
from services.orchestration.recommendation_gate import finalize_recommendations
from services.recommendation.tier_downgrade_recovery import tier_downgrade_recovery


def test_tier_downgrade_never_empty_for_economics_shortlist():
    mission = MissionState(routes=["San Francisco -> Honolulu"], passenger_count=8)
    q = (
        "I usually fly 8 passengers between San Francisco and Hawaii, occasionally Tokyo, "
        "and care far more about operating economics than prestige. What should I realistically shortlist?"
    )
    ulr_recs = [
        AircraftRecommendation(model="Global 7500", category="ultra-long", total_score=0.9, confidence=0.8, rank=1),
        AircraftRecommendation(model="Gulfstream G650ER", category="ultra-long", total_score=0.85, confidence=0.75, rank=2),
    ]
    recovered, tier = tier_downgrade_recovery(
        mission, q, prior_recommendations=[], data_used={}
    )
    assert recovered
    assert tier in ("super-midsize", "midsize", "light", "profile_fallback")
    assert not all(re.search(r"global|g650", (r.model or ""), re.I) for r in recovered[:3])


def test_finalize_recommendations_applies_recovery():
    mission = MissionState(routes=["Dallas -> New York"], passenger_count=6)
    q = "Recommend aircraft for Dallas to New York with operating economics priority"
    du: dict = {}
    gate = finalize_recommendations(
        q,
        [],
        mission,
        data_used=du,
        packet=MissionUnderstandingPacket(recommend_aircraft=True),
        max_results=3,
    )
    assert not gate.suppress_aircraft
    assert gate.filtered_recommendations


def test_dispatch_conflict_block_uses_required_phrases():
    pkt = MissionUnderstandingPacket(
        inferred_constraints={"incompatible_mission_bands": True, "industrial_hard_domain": True}
    )
    block = format_dispatch_conflict_block(pkt, query="dispatch reliability has suffered")
    assert "dispatch mismatch" in block.lower()
    assert "utilization conflict" in block.lower()
    assert "fleet segmentation required" in block.lower()


def test_comparative_table_has_structured_rows():
    table = format_comparative_analysis_table(
        MissionState(routes=["Chicago -> London"], passenger_count=16),
        query="At what point does a converted airliner become more rational than a large business jet?",
    )
    assert "Capacity economics" in table
    assert "Airport flexibility" in table
    assert "Utilization threshold" in table


def test_broker_response_structure():
    mission = MissionState(routes=["Dallas -> Chicago"], passenger_count=6)
    recs = [
        AircraftRecommendation(
            model="Citation Latitude",
            category="super-midsize",
            total_score=0.78,
            confidence=0.7,
            rank=1,
            fit="Good Fit",
            suitability_score=0.8,
            economics_score=0.75,
            operational_flexibility_score=0.7,
        )
    ]
    out = format_broker_recommendation_response(mission, recs, query="recommend a jet")
    assert "Mission Interpretation" in out
    assert "Constraint Summary" in out
    assert "Ranked Aircraft Shortlist" in out
    assert "Final Verdict" in out
    assert "Citation Latitude" in out
    assert "suitability=" in out
