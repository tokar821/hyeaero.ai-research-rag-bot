"""
Unified intent telemetry — normalized observability schema.

Unifies shadow, execution, gate, and legacy QRI signals. Observability only.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DRIFT_ALERT_THRESHOLDS = {
    "critical_mission_leakage": 0.0,
    "capability_misroute_rate": 0.5,
    "comparison_misroute_rate": 0.5,
    "router_gate_divergence_rate": 0.2,
}

_ALERT_BUFFER: List[Dict[str, Any]] = []
_DRIFT_COUNTERS: Dict[str, int] = {
    "total_events": 0,
    "drift_events": 0,
    "router_gate_divergence": 0,
    "fact_conflict": 0,
    "capability_conflict": 0,
    "comparison_conflict": 0,
    "mission_leakage_signals": 0,
}


def build_shadow_normalized(
    *,
    router_execution_path: str,
    gate_execution_path: str,
    qri_intent: str,
    alignment_status: str,
    divergence_reason: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "router_execution_path": router_execution_path,
        "gate_execution_path": gate_execution_path,
        "qri_intent": qri_intent,
        "alignment_status": alignment_status,
        "divergence_reason": divergence_reason,
    }


def build_unified_telemetry_event(
    *,
    query: str,
    route_model: Optional[str],
    execution_path: str,
    gate_path: str,
    qri_intent: str,
    router_intent: str,
    shadow_mode: bool,
    drift_detected: bool,
    latency_ms: float,
    trace_id: Optional[str] = None,
    drift_event: Optional[Dict[str, Any]] = None,
    flag_validation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "trace_id": trace_id or str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": (query or "")[:500],
        "model_resolved": route_model or "",
        "execution_path": execution_path,
        "gate_path": gate_path,
        "qri_intent": qri_intent,
        "router_intent": router_intent,
        "shadow_mode": bool(shadow_mode),
        "drift_detected": bool(drift_detected),
        "latency_ms": round(float(latency_ms), 2),
        "drift_event": drift_event or {},
        "flag_validation": flag_validation or {},
    }


def _update_counters(drift_event: Dict[str, Any]) -> None:
    _DRIFT_COUNTERS["total_events"] += 1
    if drift_event.get("is_mismatch"):
        _DRIFT_COUNTERS["drift_events"] += 1
    for key in drift_event.get("mismatch_type") or []:
        if key == "router_vs_gate":
            _DRIFT_COUNTERS["router_gate_divergence"] += 1
        elif key == "fact_conflict":
            _DRIFT_COUNTERS["fact_conflict"] += 1
        elif key == "capability_conflict":
            _DRIFT_COUNTERS["capability_conflict"] += 1
        elif key == "comparison_conflict":
            _DRIFT_COUNTERS["comparison_conflict"] += 1
        elif key == "qri_vs_unified":
            _DRIFT_COUNTERS["mission_leakage_signals"] += 1


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def evaluate_drift_alerts() -> List[Dict[str, Any]]:
    """Emit CRITICAL_ROUTING_ALERT when buffered rates exceed thresholds. Observability only."""
    alerts: List[Dict[str, Any]] = []
    total = _DRIFT_COUNTERS["total_events"]
    if total == 0:
        return alerts

    router_gate_rate = _rate(_DRIFT_COUNTERS["router_gate_divergence"], total)
    if router_gate_rate > DRIFT_ALERT_THRESHOLDS["router_gate_divergence_rate"]:
        alerts.append(
            {
                "level": "CRITICAL_ROUTING_ALERT",
                "type": "router_gate_divergence_rate",
                "rate": round(router_gate_rate, 4),
                "threshold": DRIFT_ALERT_THRESHOLDS["router_gate_divergence_rate"],
            }
        )

    cap_rate = _rate(_DRIFT_COUNTERS["capability_conflict"], total)
    if cap_rate > DRIFT_ALERT_THRESHOLDS["capability_misroute_rate"]:
        alerts.append(
            {
                "level": "CRITICAL_ROUTING_ALERT",
                "type": "capability_misroute_rate",
                "rate": round(cap_rate, 4),
                "threshold": DRIFT_ALERT_THRESHOLDS["capability_misroute_rate"],
            }
        )

    comp_rate = _rate(_DRIFT_COUNTERS["comparison_conflict"], total)
    if comp_rate > DRIFT_ALERT_THRESHOLDS["comparison_misroute_rate"]:
        alerts.append(
            {
                "level": "CRITICAL_ROUTING_ALERT",
                "type": "comparison_misroute_rate",
                "rate": round(comp_rate, 4),
                "threshold": DRIFT_ALERT_THRESHOLDS["comparison_misroute_rate"],
            }
        )

    if _DRIFT_COUNTERS["mission_leakage_signals"] > 0 and DRIFT_ALERT_THRESHOLDS["critical_mission_leakage"] == 0.0:
        alerts.append(
            {
                "level": "CRITICAL_ROUTING_ALERT",
                "type": "critical_mission_leakage",
                "count": _DRIFT_COUNTERS["mission_leakage_signals"],
                "threshold": DRIFT_ALERT_THRESHOLDS["critical_mission_leakage"],
            }
        )

    for alert in alerts:
        logger.warning("CRITICAL_ROUTING_ALERT %s", alert)
        _ALERT_BUFFER.append(alert)
    return alerts


def record_unified_intent_telemetry(event: Dict[str, Any]) -> Dict[str, Any]:
    """Persist telemetry event and update drift counters."""
    drift_event = event.get("drift_event") if isinstance(event.get("drift_event"), dict) else {}
    _update_counters(drift_event)
    logger.info(
        "unified_intent_telemetry trace_id=%s execution_path=%s gate_path=%s drift=%s",
        event.get("trace_id"),
        event.get("execution_path"),
        event.get("gate_path"),
        event.get("drift_detected"),
    )
    evaluate_drift_alerts()
    return event


def reset_telemetry_counters() -> None:
    """Test helper — reset in-memory counters."""
    for key in _DRIFT_COUNTERS:
        _DRIFT_COUNTERS[key] = 0
    _ALERT_BUFFER.clear()


def get_drift_counters() -> Dict[str, int]:
    return dict(_DRIFT_COUNTERS)


__all__ = [
    "DRIFT_ALERT_THRESHOLDS",
    "build_shadow_normalized",
    "build_unified_telemetry_event",
    "evaluate_drift_alerts",
    "get_drift_counters",
    "record_unified_intent_telemetry",
    "reset_telemetry_counters",
]
