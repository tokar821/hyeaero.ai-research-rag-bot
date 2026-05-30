"""Phase 7 controlled production rollout framework tests."""

import os
from unittest import mock

import pytest

from services.fact.aircraft_fact_responder import respond_aircraft_fact
from services.routing.unified_authority_comparator import compare_authority
from services.routing.unified_emergency_rollback import (
    AUTHORITY_DIVERGENCE_RATE_THRESHOLD,
    HARDENING_FAILURE_RATE_THRESHOLD,
    evaluate_emergency_rollback,
)
from services.routing.unified_intent_production_metrics import reset_production_metrics
from services.routing.unified_intent_router import UnifiedExecutionPath, classify_unified_intent
from services.routing.unified_pipeline_gate import evaluate_pipeline_gate
from services.routing.unified_rollout_controller import (
    RolloutDecision,
    evaluate_rollout,
    extract_rollout_session_keys,
)
from services.telemetry.unified_rollout_telemetry import (
    get_rollout_telemetry_snapshot,
    record_rollout_event,
    reset_rollout_telemetry,
)


@pytest.fixture(autouse=True)
def _reset_metrics():
    reset_rollout_telemetry()
    reset_production_metrics()
    yield
    reset_rollout_telemetry()
    reset_production_metrics()


def test_deterministic_rollout_same_session():
    with mock.patch.dict(os.environ, {"UNIFIED_INTENT_ROLLOUT_PERCENT": "25"}, clear=False):
        d1 = evaluate_rollout(user_id="user-42", conversation_id=None)
        d2 = evaluate_rollout(user_id="user-42", conversation_id=None)
    assert d1 == d2
    assert d1.source == "percentage_rollout"
    assert d1.session_bucket is not None


def test_rollout_percent_zero_defaults_legacy():
    with mock.patch.dict(os.environ, {"UNIFIED_INTENT_ROLLOUT_PERCENT": "0"}, clear=False):
        decision = evaluate_rollout(user_id="user-1", conversation_id="conv-1")
    assert decision.enabled is False
    assert decision.source == "default"


def test_rollout_percent_100_enables_all():
    with mock.patch.dict(os.environ, {"UNIFIED_INTENT_ROLLOUT_PERCENT": "100"}, clear=False):
        decision = evaluate_rollout(user_id="user-1", conversation_id="conv-1")
    assert decision.enabled is True
    assert decision.source == "percentage_rollout"


def test_rollout_conversation_id_fallback():
    with mock.patch.dict(os.environ, {"UNIFIED_INTENT_ROLLOUT_PERCENT": "50"}, clear=False):
        d_uid = evaluate_rollout(user_id="session-alpha", conversation_id="conv-beta")
        d_cid = evaluate_rollout(user_id=None, conversation_id="session-alpha")
    assert d_uid.session_bucket == d_cid.session_bucket


def test_rollout_no_session_key_stays_default():
    with mock.patch.dict(os.environ, {"UNIFIED_INTENT_ROLLOUT_PERCENT": "50"}, clear=False):
        decision = evaluate_rollout(user_id=None, conversation_id=None)
    assert decision.enabled is False
    assert decision.source == "default"
    assert "No user_id or conversation_id" in decision.reason


def test_force_legacy_overrides_percent():
    env = {
        "UNIFIED_INTENT_ROLLOUT_PERCENT": "100",
        "UNIFIED_INTENT_FORCE_LEGACY": "1",
    }
    with mock.patch.dict(os.environ, env, clear=False):
        decision = evaluate_rollout(user_id="user-1", conversation_id="conv-1")
    assert decision.enabled is False
    assert decision.source == "force_legacy"


def test_force_unified_overrides_percent():
    env = {
        "UNIFIED_INTENT_ROLLOUT_PERCENT": "0",
        "UNIFIED_INTENT_FORCE_UNIFIED": "1",
    }
    with mock.patch.dict(os.environ, env, clear=False):
        decision = evaluate_rollout(user_id="user-1", conversation_id="conv-1")
    assert decision.enabled is True
    assert decision.source == "force_unified"


def test_extract_rollout_session_keys():
    uid, cid = extract_rollout_session_keys(
        {"user_id": 99, "conversation_id": "abc-123"}
    )
    assert uid == "99"
    assert cid == "abc-123"


def test_gate_respects_rollout_disabled():
    route = classify_unified_intent("How many seats does a Falcon 8X have?")
    gate_off = evaluate_pipeline_gate(route, enforce_fact=True)
    gate_on = evaluate_pipeline_gate(route, enforce_fact=False)
    assert gate_off.enforce is True
    assert gate_on.enforce is False


def test_authority_comparator_aligned_fact():
    route = classify_unified_intent("How many seats does a Falcon 8X have?")
    gate = evaluate_pipeline_gate(route, enforce_fact=True)
    comparison = compare_authority(
        route,
        gate,
        qri_intent="payload_range_analysis",
        unified_selected=True,
        unified_output_length=80,
    )
    assert comparison.aligned is True
    assert comparison.divergence_reason is None
    assert comparison.unified_execution_path == UnifiedExecutionPath.AIRCRAFT_FACT.value


def test_authority_comparator_structural_divergence():
    route = classify_unified_intent("How many seats does a Falcon 8X have?")
    gate = evaluate_pipeline_gate(route, enforce_fact=True)
    comparison = compare_authority(
        route,
        gate,
        qri_intent="shortlist_ranking",
        unified_selected=True,
        unified_output_length=80,
    )
    assert comparison.aligned is False
    assert comparison.divergence_reason is not None


def test_rollout_telemetry_records_selection_and_divergence():
    decision = RolloutDecision(
        enabled=True,
        source="force_unified",
        reason="test",
        rollout_percent=100,
    )
    route = classify_unified_intent("How many seats does a Falcon 8X have?")
    gate = evaluate_pipeline_gate(route, enforce_fact=True)
    comparison = compare_authority(
        route,
        gate,
        qri_intent="shortlist_ranking",
        unified_selected=True,
        unified_output_length=50,
    )
    record_rollout_event(decision, comparison=comparison)
    snapshot = get_rollout_telemetry_snapshot()
    assert snapshot["unified_selected_count"] == 1
    assert snapshot["authority_divergence_count"] == 1


def test_emergency_rollback_triggers_on_hardening_spike():
    prod = {
        "hardening_failure_count": 10,
        "execution_path_none_count": 2,
        "legacy_fallback_rate": 0,
    }
    rollout = {"total_rollout_events": 20, "authority_divergence_rate": 0.0}
    status = evaluate_emergency_rollback(
        production_metrics=prod,
        rollout_telemetry=rollout,
    )
    assert status.active is True
    assert status.would_force_legacy is True
    assert any("hardening_failure_rate" in s for s in status.signals)


def test_emergency_rollback_triggers_on_divergence_spike():
    prod = {"hardening_failure_count": 0, "execution_path_none_count": 0}
    rollout = {
        "total_rollout_events": 20,
        "authority_divergence_rate": AUTHORITY_DIVERGENCE_RATE_THRESHOLD + 0.1,
    }
    status = evaluate_emergency_rollback(
        production_metrics=prod,
        rollout_telemetry=rollout,
    )
    assert status.active is True
    assert "authority_divergence_rate" in status.reason


def test_emergency_rollback_inactive_within_limits():
    prod = {"hardening_failure_count": 1, "execution_path_none_count": 1}
    rollout = {"total_rollout_events": 100, "authority_divergence_rate": 0.01}
    status = evaluate_emergency_rollback(
        production_metrics=prod,
        rollout_telemetry=rollout,
    )
    assert status.active is False


def test_rollback_observe_does_not_auto_force_legacy():
    from services.routing.unified_emergency_rollback import RollbackStatus

    status = RollbackStatus(
        active=True,
        reason="test rollback",
        would_force_legacy=True,
    )
    with mock.patch.dict(os.environ, {"UNIFIED_INTENT_ROLLOUT_PERCENT": "100"}, clear=False):
        decision = evaluate_rollout(
            user_id="user-1",
            conversation_id="conv-1",
            rollback_status=status,
        )
    assert decision.enabled is True
    assert decision.source == "percentage_rollout"


def test_responder_output_unchanged_by_rollout_layer():
    answer = respond_aircraft_fact("Falcon 8X", "seats")
    with mock.patch.dict(os.environ, {"UNIFIED_INTENT_ROLLOUT_PERCENT": "100"}, clear=False):
        evaluate_rollout(user_id="u1", conversation_id="c1")
    assert respond_aircraft_fact("Falcon 8X", "seats") == answer


def test_percentage_rollout_distribution_stable():
    with mock.patch.dict(os.environ, {"UNIFIED_INTENT_ROLLOUT_PERCENT": "25"}, clear=False):
        enabled = sum(
            1
            for i in range(200)
            if evaluate_rollout(user_id=f"stable-user-{i}", conversation_id=None).enabled
        )
    assert 30 <= enabled <= 80
