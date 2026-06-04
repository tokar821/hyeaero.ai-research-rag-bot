"""Phase 36 — unit tests for cross-pipeline consistency."""

from __future__ import annotations

import pytest

from services.consistency.consistency_injection_layer import (
    prepare_buy_decision_state,
    prepare_comparison_consistency,
    prepare_valuation_state,
    render_buy_decision_answer,
    render_valuation_answer,
)
from services.consistency.cross_model_identity import resolve_canonical_identity
from services.consistency.pipeline_agreement_checker import AgreementFlag, check_pipeline_agreement
from services.consistency.unified_broker_state import UnifiedBrokerState

pytestmark = pytest.mark.deterministic


def test_resolve_canonical_identity_latitude() -> None:
    ident = resolve_canonical_identity(
        query="Is a 2015 Citation Latitude for $5M a good deal?",
        explicit_model="Citation Latitude",
        source_layer="dispatch",
    )
    assert ident.canonical_model == "Citation Latitude"
    assert ident.confidence_score >= 80


def test_unified_buy_state_single_band() -> None:
    parsed = {"model": "Citation Latitude", "year": 2015, "ask_usd": 5_000_000.0}
    du: dict = {}
    state = prepare_buy_decision_state(
        query="Is a 2015 Citation Latitude for $5M a good deal?",
        parsed=parsed,
        db=None,
        data_used=du,
    )
    assert state.market_band is not None
    assert state.market_bundle is not None
    body = render_buy_decision_answer(state)
    assert "Market Band" in body or "authority catalog" in body.lower()
    assert "Liquidity:" in body
    assert du.get("unified_broker_state", {}).get("canonical_model") == "Citation Latitude"


def test_pipeline_agreement_no_verdict_drift_after_inject() -> None:
    parsed = {"model": "Citation Latitude", "year": 2015, "ask_usd": 5_000_000.0}
    du: dict = {}
    state = prepare_buy_decision_state(
        query="Is a 2015 Citation Latitude for $5M a good deal?",
        parsed=parsed,
        db=None,
        data_used=du,
    )
    report = check_pipeline_agreement(data_used=du, state=state)
    assert AgreementFlag.VERDICT_INCONSISTENCY not in report.flags


def test_valuation_uses_unified_state() -> None:
    du: dict = {}
    state = prepare_valuation_state(
        query="What is a 2019 Citation Latitude worth?",
        model="Citation Latitude",
        year="2019",
        db=None,
        data_used=du,
    )
    body = render_valuation_answer(state, year_label="2019")
    assert "Citation Latitude" in body
    assert "Verdict:" in body
    assert du.get("unified_broker_state")


def test_comparison_identity_lock() -> None:
    du: dict = {}
    report = prepare_comparison_consistency(
        query="G650 vs Falcon 8X",
        compare_models=["Gulfstream G650", "Falcon 8X"],
        data_used=du,
    )
    models = (du.get("comparison_v2") or {}).get("models")
    assert isinstance(models, list)
    assert len(models) == 2
    assert report.aligned or AgreementFlag.MODEL_MISMATCH not in report.flags
