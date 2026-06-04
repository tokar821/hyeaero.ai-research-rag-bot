"""Phase 50 — broker quality scoring unit tests."""

from __future__ import annotations

from services.broker_scoring.broker_quality_score import score_broker_answer
from services.broker_scoring.recommendation_consistency_audit_v2 import (
    audit_recommendation_consistency_v2,
)


def test_score_infeasible_budget_high_on_budget_dimension():
    du = {"acquisition_budget_infeasible": True}
    result = score_broker_answer(
        "No.\n\nThat budget is far below the current market.\n\nA Gulfstream G700 does not trade near $5M.",
        query="Can I buy a G700 for $5M?",
        data_used=du,
    )
    assert result["total"] >= 75
    assert result["breakdown"]["budget_realism"] >= 15


def test_score_attaches_forbidden_phrase_penalty():
    result = score_broker_answer(
        "Right now, buyer leverage is strong on this deal.",
        query="What would you buy?",
        data_used={},
    )
    assert result["breakdown"]["natural_broker_language"] < 15


def test_recommendation_drift_detected():
    du: dict = {}
    audit_recommendation_consistency_v2(
        "If I were buying today, I'd focus on the Gulfstream G280.",
        query="I have $12M. What should I buy?",
        data_used=du,
    )
    audit_recommendation_consistency_v2(
        "If I were buying today, I'd focus on the Citation Longitude.",
        query="What about something else?",
        data_used=du,
    )
    audit = du["recommendation_consistency_audit_v2"]
    assert audit["recommendation_drift"] is True
    assert audit["drift_events"][0]["type"] == "RECOMMENDATION_DRIFT"


def test_no_drift_when_budget_changes():
    du: dict = {}
    audit_recommendation_consistency_v2(
        "I'd focus on the Gulfstream G280.",
        query="I have $12M.",
        data_used=du,
    )
    audit_recommendation_consistency_v2(
        "I'd focus on the Gulfstream G650.",
        query="What if I stretch to $20M?",
        data_used=du,
    )
    audit = du["recommendation_consistency_audit_v2"]
    assert audit["recommendation_drift"] is False
