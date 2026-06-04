"""Phase 44 — executive broker authority tests."""

from __future__ import annotations

import re

import pytest

from services.client_context.client_context_layer import (
    apply_client_context_turn,
    personalize_client_answer,
)
from services.broker_decision.broker_decision_layer import apply_broker_decision_synthesis
from services.broker_reasoning.broker_reasoning_layer import apply_broker_reasoning_layer
from services.executive_broker.broker_consistency_audit import audit_broker_consistency
from services.executive_broker.executive_answer_rewriter import (
    has_equal_weight_recommendations,
    rewrite_executive_answer,
)
from services.executive_broker.executive_broker_layer import apply_executive_broker_layer
from services.executive_broker.executive_recommendation import ExecutiveRecommendation
from services.executive_broker.recommendation_selector import select_executive_recommendation


def _pipeline(query: str, state: dict | None = None, history: list | None = None) -> tuple[dict, str]:
    du: dict = {}
    apply_client_context_turn(
        query,
        data_used=du,
        history=history,
        client_conversation_state=state or {},
    )
    apply_broker_reasoning_layer(query, data_used=du)
    raw = apply_broker_decision_synthesis(
        "Citation Latitude is a super-midsize jet with strong range.",
        query=query,
        data_used=du,
    )
    raw = personalize_client_answer(raw, query=query, data_used=du)
    out = apply_executive_broker_layer(raw, query=query, data_used=du)
    return du, out


def test_primary_recommendation_lead():
    from services.client_context.client_context_layer import finalize_client_context

    state: dict = {}
    history: list = []
    du1, _ = _pipeline("I have about 12M", state=state, history=history)
    finalize_client_context(du1, state, query="I have about 12M", history=history)
    history.extend(
        [
            {"role": "user", "content": "I have about 12M"},
            {"role": "assistant", "content": "Noted."},
        ]
    )
    du2, _ = _pipeline("I like Gulfstreams", state=state, history=history)
    finalize_client_context(du2, state, query="I like Gulfstreams", history=history)
    history.extend(
        [
            {"role": "user", "content": "I like Gulfstreams"},
            {"role": "assistant", "content": "Understood."},
        ]
    )
    du, out = _pipeline("What should I buy?", state=state, history=history)
    assert du.get("executive_broker_layer_applied") == 1
    assert re.search(r"(?i)\b(?:my primary recommendation would be|i'd focus on|i would buy)\b", out)
    assert "G280" in out
    assert not re.search(
        r"(?i)focus on .+latitude.+praetor",
        out,
    )


def test_not_equal_weight_bullets():
    state = {"remembered_budget_musd": 12.0, "preferred_manufacturers": ["Gulfstream"]}
    du, out = _pipeline("what can I buy for 12M", state=state)
    bullets = re.findall(r"(?m)^\s*[•\-]\s+", out)
    assert len(bullets) <= 4
    assert re.search(r"(?i)\b(?:primary recommendation|i'd focus on)\b", out)


def test_select_g280_not_g650_at_12m():
    du: dict = {
        "client_context": {
            "remembered_budget_musd": 12.0,
            "preferred_manufacturers": ["Gulfstream"],
        },
        "broker_decision": {
            "answer_type": "opportunities",
            "direct_answer": "At $12M, I would focus on Gulfstream G280, Citation Latitude, Praetor 600.",
            "alternatives": [
                {"model": "Gulfstream G280", "rationale": "Entry Gulfstream in band."},
                {"model": "Citation Latitude", "rationale": "Super-mid value."},
                {"model": "Embraer Praetor 600", "rationale": "Range for dollar."},
                {"model": "Bombardier Challenger 350", "rationale": "Cabin."},
            ],
        },
    }
    rec = select_executive_recommendation("What should I buy?", data_used=du)
    assert rec is not None
    assert rec.primary_recommendation == "Gulfstream G280"
    assert all(a["model"] != rec.primary_recommendation for a in rec.alternatives)


def test_rewrite_structure():
    rec = ExecutiveRecommendation(
        primary_recommendation="Gulfstream G280",
        confidence="HIGH",
        rationale="Fits a $12M Gulfstream-focused search.",
        alternatives=[{"model": "Citation Latitude", "rationale": "If you widen OEM preference."}],
        rejected_options=[{"model": "Gulfstream G650", "reason": "Above budget cap."}],
    )
    out = rewrite_executive_answer(
        "At $12M, I would focus on G280, Latitude, and Praetor equally.",
        rec,
    )
    assert "My primary recommendation would be" in out
    assert out.index("My primary recommendation") < 80
    assert "G280" in out


def test_equal_weight_detection():
    assert has_equal_weight_recommendations(
        "At $12M, I would focus on Gulfstream G280, Citation Latitude, and Praetor 600."
    )
    assert not has_equal_weight_recommendations(
        "My primary recommendation would be the Gulfstream G280 — fits your cap."
    )


def test_consistency_audit_flags_budget_drift():
    score = audit_broker_consistency(
        primary="Gulfstream G700",
        alternatives=[],
        data_used={"client_context": {"remembered_budget_musd": 12.0}},
    )
    assert score.budget_drift is True
    assert score.overall < 0.8


def test_skip_pure_comparison():
    du: dict = {}
    out = apply_executive_broker_layer(
        "G650 leads on range; G700 leads on cabin.",
        query="compare G650 vs G700",
        data_used=du,
    )
    assert not du.get("executive_broker_layer_applied")
    assert du.get("executive_layer_suppressed") or du.get("executive_layer_allowed") is False
    assert "primary recommendation" not in out.lower()
