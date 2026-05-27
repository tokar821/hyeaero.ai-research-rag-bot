"""Recommendation diversity guard — triad genuine-lead, repetition, transparency."""

from __future__ import annotations

from services.consultant.mission_state import build_mission_from_current_turn
from services.consultant.recommendation_engine import AircraftRecommendation, rank_aircraft_recommendations
from services.recommendation.diversity_controls import (
    load_recommendation_history,
    mission_fingerprint,
    record_recommendation_history,
)
from services.recommendation.mission_ranker import MissionCategory, classify_mission_category, rank_missions
from services.recommendation.recommendation_diversity_guard import (
    DEFAULT_TRIAD_MODELS,
    genuinely_scores_highest,
)


def test_genuinely_scores_highest_blocks_triad_when_non_triad_leads():
    leader = AircraftRecommendation(
        model="Citation Latitude",
        category="super-midsize",
        total_score=0.82,
        confidence=0.7,
        rank=0,
    )
    triad = AircraftRecommendation(
        model="Challenger 350",
        category="super-midsize",
        total_score=0.74,
        confidence=0.7,
        rank=0,
    )
    ok, reason = genuinely_scores_highest("Challenger 350", [leader, triad])
    assert not ok
    assert "Citation Latitude" in reason or "outscored" in reason


def test_genuinely_scores_highest_allows_triad_when_leader():
    triad = AircraftRecommendation(
        model="Praetor 600",
        category="super-midsize",
        total_score=0.88,
        confidence=0.7,
        rank=0,
    )
    other = AircraftRecommendation(
        model="Citation CJ4",
        category="light",
        total_score=0.71,
        confidence=0.7,
        rank=0,
    )
    ok, reason = genuinely_scores_highest("Praetor 600", [triad, other])
    assert ok
    assert reason == "genuine_leader"


def test_boston_miami_shortlist_not_all_default_triad():
    models = [
        r.model
        for r in rank_aircraft_recommendations(
            build_mission_from_current_turn("6 pax Boston to Miami nonstop recommend"),
            max_results=5,
        )
        if not r.avoid
    ]
    assert models
    assert set(models[:3]) != DEFAULT_TRIAD_MODELS


def test_ranking_transparency_in_audit():
    mission = build_mission_from_current_turn("6 pax Boston to Miami nonstop")
    _cat, _recs, _feas, audit = rank_missions(mission, max_results=3)
    assert audit is not None
    assert audit.ranking_transparency
    assert len(audit.ranking_transparency) >= 3
    row = audit.ranking_transparency[0]
    assert "model" in row
    assert "adjusted_score" in row
    assert "triad_guard" in row
    assert "vs_leader_delta" in row


def test_repetition_justification_in_audit():
    mission_a = build_mission_from_current_turn("6 pax Boston to Miami nonstop")
    mission_b = build_mission_from_current_turn("8 executives New York to London nonstop")
    cat_b = classify_mission_category(mission_b)
    fp_b = mission_fingerprint(mission_b, mission_category=cat_b)
    data_used: dict = {}
    record_recommendation_history(
        fingerprint=mission_fingerprint(mission_a, mission_category=classify_mission_category(mission_a)),
        ranked_models=["Challenger 350", "Gulfstream G280"],
        data_used=data_used,
    )
    _cat, recs, _feas, audit = rank_missions(
        mission_b,
        max_results=5,
        data_used=data_used,
    )
    hist = load_recommendation_history(None, data_used)
    assert hist
    if audit.repetition_justifications:
        assert any("alternative" in v.lower() or "unrelated" in v.lower() for v in audit.repetition_justifications.values())
    transparency = audit.ranking_transparency or []
    repeated_rows = [r for r in transparency if r.get("repetition_hits", 0) > 0]
    if repeated_rows:
        assert any(r.get("mission_justification_required") for r in repeated_rows)


def test_lax_london_excludes_triad_unless_feasible():
    from services.pipeline.run_pipeline import run_advisory_pipeline

    result = run_advisory_pipeline(
        "8 passengers Los Angeles to London nonstop",
        max_results=5,
    )
    models = [r.model for r in result.recommendations]
    assert "Challenger 350" not in models
    assert "Praetor 600" not in models
    audit = result.recommendation_audit or {}
    assert audit.get("ranking_transparency")
