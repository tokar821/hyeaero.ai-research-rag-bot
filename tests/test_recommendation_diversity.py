"""Recommendation diversity controls — overuse, ULR class fit, repetition, audit."""

from __future__ import annotations

from services.consultant.mission_state import build_mission_from_current_turn
from services.consultant.recommendation_engine import rank_aircraft_recommendations
from services.pipeline.run_pipeline import run_advisory_pipeline
from services.recommendation.diversity_controls import (
    load_recommendation_history,
    mission_fingerprint,
    overuse_penalty_for_model,
    record_recommendation_history,
    repetition_penalty_for_model,
    undersized_platform_hard_reject,
)
from services.recommendation.mission_ranker import MissionCategory, classify_mission_category, rank_missions
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES


def test_ulr_hard_rejects_super_midsize_triad():
    spec = AIRCRAFT_PROFILES["Challenger 350"]
    reason = undersized_platform_hard_reject(
        "Challenger 350",
        spec,
        MissionCategory.ULTRA_LONG_RANGE,
        max_leg_nm=5200,
    )
    assert reason is not None
    assert "super-midsize" in reason.lower() or "ULR" in reason


def test_tokyo_mission_top_three_not_all_super_midsize():
    mission = build_mission_from_current_turn(
        "6 executives San Francisco to Tokyo nonstop westbound winter"
    )
    cat, recs, _, audit = rank_missions(mission, max_results=5)
    assert cat == MissionCategory.ULTRA_LONG_RANGE
    top3 = [r.model for r in recs if not r.avoid][:3]
    assert top3
    assert any(
        m in top3
        for m in ("Global 7500", "Gulfstream G650", "Gulfstream G650ER", "Falcon 8X")
    )
    sm_in_top = sum(
        1
        for r in recs[:3]
        if (r.category or "").replace("_", "-") in ("super-midsize", "midsize")
    )
    assert sm_in_top < 3
    assert audit is not None
    assert audit.ranked_models


def test_la_miami_not_only_default_triad():
    models = [
        r.model
        for r in rank_aircraft_recommendations(
            build_mission_from_current_turn("6 pax Boston to Miami nonstop $10M recommend"),
            max_results=5,
        )
        if not r.avoid
    ]
    assert models
    assert not set(models[:3]) == {"Challenger 350", "Gulfstream G280", "Praetor 600"}


def test_repetition_penalty_on_unrelated_mission():
    mission_a = build_mission_from_current_turn("8 pax LA to Miami nonstop")
    mission_b = build_mission_from_current_turn("6 executives New York to London nonstop winter")
    cat_a = classify_mission_category(mission_a)
    fp_a = mission_fingerprint(mission_a, mission_category=cat_a)
    fp_b = mission_fingerprint(mission_b, mission_category=classify_mission_category(mission_b))

    data_used: dict = {}
    record_recommendation_history(
        fingerprint=fp_a,
        ranked_models=["Challenger 350", "Gulfstream G280"],
        data_used=data_used,
    )
    hist = load_recommendation_history(None, data_used)
    pen, note = repetition_penalty_for_model("Challenger 350", fp_b, hist)
    assert pen > 0
    assert note


def test_overuse_penalty_higher_on_ulr_for_praetor():
    pen, _ = overuse_penalty_for_model(
        "Praetor 600",
        MissionCategory.ULTRA_LONG_RANGE,
        max_leg_nm=5200,
    )
    assert pen >= 0.28


def test_pipeline_emits_recommendation_audit():
    result = run_advisory_pipeline(
        "8 passengers LA to Miami nonstop recommend",
        max_results=3,
    )
    assert result.recommendations
    assert isinstance(result.recommendation_audit, dict)
    assert result.recommendation_audit.get("ranked_models")
    assert "scoring_notes" in result.recommendation_audit


def test_aspen_runway_mission_prefers_field_performance_types():
    mission = build_mission_from_current_turn("Dallas to Aspen hot and high 6 pax")
    cat, recs, _, _ = rank_missions(mission, max_results=4)
    assert cat == MissionCategory.MOUNTAIN_AIRPORT
    top = [r.model for r in recs if not r.avoid][:3]
    assert any(
        m in top
        for m in (
            "Citation Latitude",
            "Pilatus PC-24",
            "Challenger 350",
            "Praetor 600",
            "Gulfstream G280",
            "Challenger Longitude",
            "Challenger 650",
        )
    )
    assert "Falcon 8X" not in top
    assert "Global 7500" not in top


def test_impossible_aircraft_not_in_shortlist_lax_london():
    result = run_advisory_pipeline(
        "8 passengers Los Angeles to London nonstop",
        max_results=5,
    )
    models = [r.model for r in result.recommendations]
    assert "Challenger 350" not in models
    assert "Praetor 600" not in models
    log_models = {
        e.get("aircraft_name") or e.get("aircraft")
        for e in result.elimination_log
        if isinstance(e, dict)
    }
    assert "Challenger 350" in log_models or any(
        "Challenger 350" in str(e) for e in result.elimination_log
    )
