"""Phase 51 — unit tests for broker audit diagnostics."""

from __future__ import annotations

import pytest

from services.broker_audit.broker_trace import attach_broker_trace, build_broker_trace
from services.broker_audit.broker_trust_score import attach_broker_trust_score
from services.broker_audit.root_cause_analyzer import FailureCause, analyze_root_cause
from services.broker_scoring.recommendation_consistency_audit_v2 import (
    audit_recommendation_consistency_v2,
)


def test_broker_trace_attaches_to_data_used():
    du = {
        "intent_lock": {"dispatch_authority_id": "comparison"},
        "executive_recommendation": {"primary_recommendation": "Citation Longitude"},
        "broker_quality_score": {"total": 88.0},
    }
    answer = "If I were buying today, I'd focus on the Citation Longitude."
    trace = attach_broker_trace(answer, query="longitude vs challenger", data_used=du)
    assert du["broker_trace"]["authority_selected"] == "comparison"
    assert trace.executive_primary
    assert du["broker_trace"]["broker_quality_score"] == 88.0


def test_broker_trust_score_range():
    du = {"broker_quality_score": {"total": 90.0}, "intent_lock": {"dispatch_authority_id": "alternative"}}
    result = attach_broker_trust_score("I'd focus on the G280.", query="gulfstream under 12m", data_used=du)
    assert 0 <= result["total"] <= 100
    assert du["broker_trust_score"]["total"] == result["total"]


def test_root_cause_infeasible_recommendation():
    du = {}
    result = analyze_root_cause(
        query="Can I buy a G700 for $5M?",
        answer="I'd focus on the Gulfstream G700.",
        data_used=du,
        expect_infeasible=True,
        forbidden_primary="G700",
    )
    assert result.cause == FailureCause.RECOMMENDATION_ERROR


def test_unjustified_drift_flagged():
    du: dict = {}
    audit_recommendation_consistency_v2(
        "I'd focus on the Citation Longitude.",
        query="I have $20M for coast-to-coast.",
        data_used=du,
    )
    audit = audit_recommendation_consistency_v2(
        "I'd focus on the Challenger 350.",
        query="What about maintenance costs?",
        data_used=du,
    )
    assert audit["first_primary"]
    assert audit["latest_primary"]
    assert audit.get("unjustified_recommendation_drift") or audit.get("recommendation_drift")


def test_justified_drift_on_budget_change():
    du: dict = {}
    audit_recommendation_consistency_v2(
        "I'd focus on the G280.",
        query="Budget is $12M.",
        data_used=du,
    )
    audit = audit_recommendation_consistency_v2(
        "I'd focus on the G650.",
        query="Actually I can stretch to $25M now.",
        data_used=du,
    )
    assert audit["budget_changes"]
    unjustified = audit.get("unjustified_recommendation_drift")
    events = audit.get("drift_events") or []
    unjustified_types = [e.get("type") for e in events]
    assert "UNJUSTIFIED_RECOMMENDATION_DRIFT" not in unjustified_types or not unjustified
