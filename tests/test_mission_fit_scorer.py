"""Tests for Phase 52 mission fit scoring."""

from services.broker_decision.mission_fit_scorer import rank_models_for_recommendation, score_model_fit


def test_coast_to_coast_prefers_longitude_over_g280_at_20m():
    models = ["Gulfstream G280", "Citation Longitude", "Challenger 350"]
    q = "Coast-to-coast nonstop, 6 passengers, $20M budget — what should I buy?"
    ranked = rank_models_for_recommendation(models, query=q, data_used={})
    assert ranked[0] == "Citation Longitude"


def test_named_g650_scores_highest():
    q = "G650 for $18M — is that plausible?"
    assert score_model_fit("Gulfstream G650", query=q, data_used={}) > score_model_fit(
        "Gulfstream G280", query=q, data_used={}
    )
