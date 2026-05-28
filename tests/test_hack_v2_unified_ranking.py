"""HACK v2 — ranking + verdict unification tests."""

from __future__ import annotations

import pytest

from services.consultant.mission_state import MissionState
from services.mission.models import MissionProfile, Route
from services.recommendation.hack_v1_constraint_kernel import (
    HACK_V1_METADATA_KEY,
    HackV1Result,
    attach_hack_v1_metadata,
    apply_hack_v1_gate,
)
from services.recommendation.hack_v2_unified_ranking import (
    RankingIntegrityError,
    hack_v2_unify_rank_and_verdict,
)
from services.consultant.recommendation_engine import (
    AircraftRecommendation,
    RecommendationScore,
)


def _profile(routes: list[str], *, pax: int = 6) -> MissionProfile:
    return MissionProfile(
        routes=[Route.from_label(r) for r in routes if Route.from_label(r)],
        passengers=pax,
        nonstop_required=True,
        nbaa_reserve_required=True,
    )


def _rec(model: str, *, suit: float, flex: float, econ: float) -> AircraftRecommendation:
    rec = AircraftRecommendation(
        model=model,
        category="test",
        total_score=0.0,
        confidence=0.5,
        rank=1,
    )
    rec.suitability_score = suit
    rec.operational_flexibility_score = flex
    rec.economics_score = econ
    rec.scores = [
        RecommendationScore(dimension="range_realism", score=suit, weight=0.0, weighted=0.0),
        RecommendationScore(dimension="passenger_count_fit", score=suit, weight=0.0, weighted=0.0),
        RecommendationScore(dimension="runway_performance", score=flex, weight=0.0, weighted=0.0),
        RecommendationScore(dimension="runway_flexibility", score=flex, weight=0.0, weighted=0.0),
    ]
    return rec


def test_apply_hack_v1_gate_empty_filtered_is_constraint_empty():
    profile = _profile(["Yellowknife -> Remote Gravel Strips"])
    filtered, result = apply_hack_v1_gate(
        profile,
        [],
        all_candidates=["Global 7500"],
        query="Northern Canada Arctic gravel strips",
    )
    assert filtered == []
    assert result.constraint_empty is True


def test_hack_v2_sorts_by_composite_desc_only():
    mission = MissionState(routes=["A -> B"], passenger_count=6)
    du: dict = {}
    attach_hack_v1_metadata(
        du,
        HackV1Result(
            feasible_aircraft_list=["Alpha Jet", "Bravo Jet"],
            rejection_log=[],
            constraint_empty=False,
        ),
    )
    recs = [
        _rec("Bravo Jet", suit=0.9, flex=0.7, econ=0.6),
        _rec("Alpha Jet", suit=0.5, flex=0.5, econ=0.5),
    ]
    rows = hack_v2_unify_rank_and_verdict(
        mission=mission,
        recommendations=recs,
        packet=None,
        data_used=du,
    )
    assert [r["aircraft_name"] for r in rows] == ["Bravo Jet", "Alpha Jet"]
    assert rows[0]["composite_score"] >= rows[1]["composite_score"]
    assert recs[0].fit_verdict in ("GOOD FIT", "CONDITIONAL FIT")


def test_hack_v2_not_a_fit_cannot_rank_first():
    mission = MissionState(routes=["A -> B"], passenger_count=6)
    du: dict = {}
    attach_hack_v1_metadata(
        du,
        HackV1Result(
            feasible_aircraft_list=["Alpha Jet"],
            rejection_log=[],
            constraint_empty=False,
        ),
    )
    rec = _rec("Alpha Jet", suit=0.1, flex=0.1, econ=0.1)
    rec.scores = [
        RecommendationScore(dimension="range_realism", score=0.1, weight=0.0, weighted=0.0),
        RecommendationScore(dimension="passenger_count_fit", score=0.1, weight=0.0, weighted=0.0),
        RecommendationScore(dimension="runway_performance", score=0.1, weight=0.0, weighted=0.0),
        RecommendationScore(dimension="runway_flexibility", score=0.1, weight=0.0, weighted=0.0),
    ]
    rows = hack_v2_unify_rank_and_verdict(
        mission=mission,
        recommendations=[rec],
        packet=None,
        data_used=du,
    )
    assert len(rows) == 1
    assert rows[0]["verdict"] != "NOT A FIT"


def test_hack_v2_rejects_aircraft_outside_hack_v1_set():
    mission = MissionState(routes=["A -> B"], passenger_count=6)
    du: dict = {HACK_V1_METADATA_KEY: {"feasible_aircraft_list": ["Alpha Jet"], "rejection_log": [], "constraint_empty": False}}
    recs = [_rec("Alpha Jet", suit=0.8, flex=0.7, econ=0.6), _rec("CJ4", suit=0.95, flex=0.9, econ=0.9)]
    rows = hack_v2_unify_rank_and_verdict(
        mission=mission,
        recommendations=recs,
        packet=None,
        data_used=du,
    )
    assert [r["aircraft_name"] for r in rows] == ["Alpha Jet"]


def test_hack_v2_missing_hack_v1_raises():
    mission = MissionState(routes=["A -> B"], passenger_count=6)
    with pytest.raises(RankingIntegrityError):
        hack_v2_unify_rank_and_verdict(
            mission=mission,
            recommendations=[_rec("Alpha Jet", suit=0.5, flex=0.5, econ=0.5)],
            packet=None,
            data_used={},
        )
