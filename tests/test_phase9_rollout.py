"""Phase 9 production rollout monitoring tests."""

import json

from monitoring.drift_capture import DriftEvent, capture_drift_event, get_drift_capture, reset_drift_capture
from monitoring.live_benchmark import compare_live_metadata, get_live_benchmark, reset_live_benchmark
from monitoring.live_path_analytics import get_live_path_analytics, infer_path_category, reset_live_path_analytics
from monitoring.production_event_bus import ingest_consultant_turn
from monitoring.production_health_score import compute_production_health
from monitoring.unified_rollout_dashboard import build_rollout_dashboard_snapshot
from rollout.rollout_plan import ROLLOUT_STAGES, recommend_next_stage
from services.routing.unified_intent_production_metrics import reset_production_metrics
from services.telemetry.unified_rollout_telemetry import record_rollout_event, reset_rollout_telemetry
from services.routing.unified_rollout_controller import RolloutDecision


def _reset_all():
    reset_drift_capture()
    reset_live_path_analytics()
    reset_live_benchmark()
    reset_rollout_telemetry()
    reset_production_metrics()


def test_drift_capture_ring_buffer():
    _reset_all()
    for i in range(5):
        capture_drift_event(
            DriftEvent(
                query=f"query {i}",
                execution_path="aircraft_fact",
                model="Falcon 8X",
                hardening_flags={"routing_failure": False, "ambiguity_detected": False, "fallback_triggered": False},
                rollback_status={"active": False},
                path_category="FACT",
            )
        )
    assert get_drift_capture().count() == 5
    exported = get_drift_capture().export(limit=2)
    assert len(exported) == 2


def test_live_path_analytics_aggregation():
    _reset_all()
    analytics = get_live_path_analytics()
    analytics.record("FACT", unified_enforced=True, fallback=False, latency_ms=50.0)
    analytics.record("FACT", unified_enforced=False, fallback=True, latency_ms=30.0)
    snap = analytics.snapshot()
    assert snap["by_category"]["FACT"]["total_requests"] == 2
    assert snap["by_category"]["FACT"]["successful_executions"] == 1
    assert snap["by_category"]["FACT"]["fallback_executions"] == 1


def test_infer_path_category():
    assert infer_path_category("aircraft_fact") == "FACT"
    assert infer_path_category("none", qri_intent="acquisition_recommendation") == "BUY_DECISION"
    assert infer_path_category("none", qri_intent="mission_feasibility") == "MISSION"


def test_live_benchmark_metadata():
    _reset_all()
    compare_live_metadata(
        unified_execution_path="aircraft_fact",
        legacy_qri_intent="payload_range_analysis",
        authority_aligned=True,
        unified_latency_ms=40.0,
        legacy_latency_ms=120.0,
        unified_output_length=80,
        legacy_output_length=200,
    )
    snap = get_live_benchmark().snapshot()
    assert snap["total_events"] == 1
    assert snap["path_agreement_rate"] >= 0.0


def test_ingest_consultant_turn():
    _reset_all()
    data_used = {
        "unified_intent_shadow": {
            "qri_intent": "payload_range_analysis",
            "model_resolved": "Falcon 8X",
            "hardening_flags": {
                "routing_failure": False,
                "ambiguity_detected": False,
                "fallback_triggered": False,
            },
        },
        "unified_emergency_rollback": {"active": False},
        "unified_authority_comparison": {"aligned": True},
        "unified_intent_telemetry": {"execution_path": "aircraft_fact"},
    }
    ingest_consultant_turn(
        data_used,
        query="How many seats does a Falcon 8X have?",
        qri_intent="payload_range_analysis",
        latency_ms=25.0,
        unified_selected=True,
        unified_enforced=True,
        execution_path="aircraft_fact",
        unified_output_length=90,
    )
    assert get_drift_capture().count() == 1
    assert get_live_path_analytics().snapshot()["totals"]["total_requests"] >= 1


def test_dashboard_snapshot_json():
    _reset_all()
    snap = build_rollout_dashboard_snapshot()
    assert "rollout" in snap
    assert "hardening" in snap
    assert "rollback" in snap
    assert "authority" in snap
    assert "execution_path_distribution" in snap
    json.dumps(snap)


def test_production_health_score():
    dashboard = {
        "rollout": {"total_rollout_events": 100, "unified_traffic_percent": 25.0},
        "hardening": {"hardening_failure_count": 2, "execution_path_none_count": 1},
        "authority": {
            "divergence_rate": 0.03,
            "live_benchmark": {"path_agreement_rate": 0.95},
        },
        "rollback": {"active": False},
    }
    health = compute_production_health(dashboard)
    assert health.status in ("HEALTHY", "WATCH", "DEGRADED", "CRITICAL")
    assert 0.0 <= health.score <= 1.0


def test_rollout_stage_recommendation():
    _reset_all()
    record_rollout_event(
        RolloutDecision(enabled=True, source="percentage_rollout", reason="test", rollout_percent=5),
    )
    rec = recommend_next_stage()
    assert rec.current_stage.percent in ROLLOUT_STAGES
    assert rec.recommended_stage.percent in ROLLOUT_STAGES
    assert rec.action in ("ADVANCE", "HOLD", "HOLD_OR_REDUCE", "ROLLBACK")


def test_failure_analysis_from_drift():
    _reset_all()
    capture_drift_event(
        DriftEvent(
            query="Can Longitude fly?",
            execution_path="none",
            model=None,
            hardening_flags={
                "routing_failure": True,
                "ambiguity_detected": True,
                "fallback_triggered": True,
            },
            rollback_status={"active": False},
            path_category="CAPABILITY",
        )
    )
    from monitoring.failure_analysis import build_live_failure_reports

    report = build_live_failure_reports()
    assert report["top_failing_categories"][0]["category"] == "CAPABILITY"
