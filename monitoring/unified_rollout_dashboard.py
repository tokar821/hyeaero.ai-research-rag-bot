"""
Unified rollout dashboard — JSON snapshot of production rollout observability.

Aggregates Phase 6/7/9 telemetry without modifying routing behavior.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional

from monitoring.drift_capture import get_drift_capture
from monitoring.live_benchmark import get_live_benchmark
from monitoring.live_path_analytics import get_live_path_analytics
from services.routing.unified_emergency_rollback import evaluate_emergency_rollback
from services.routing.unified_intent_production_metrics import get_production_metrics
from services.telemetry.unified_intent_telemetry import evaluate_drift_alerts, get_drift_counters
from services.telemetry.unified_rollout_telemetry import get_rollout_telemetry_snapshot


def _configured_rollout_percent() -> int:
    raw = (os.getenv("UNIFIED_INTENT_ROLLOUT_PERCENT") or "0").strip()
    try:
        return max(0, min(100, int(raw)))
    except ValueError:
        return 0


def build_rollout_dashboard_snapshot(
    *,
    drift_limit: int = 100,
) -> Dict[str, Any]:
    """Build JSON-serializable rollout dashboard snapshot from in-memory telemetry."""
    rollout = get_rollout_telemetry_snapshot()
    production = get_production_metrics()
    drift_counters = get_drift_counters()
    path_analytics = get_live_path_analytics().snapshot()
    live_bench = get_live_benchmark().snapshot()
    rollback = evaluate_emergency_rollback(
        production_metrics=production,
        rollout_telemetry=rollout,
    )
    alerts = evaluate_drift_alerts()

    total_rollout = int(rollout.get("total_rollout_events") or 0)
    unified_n = int(rollout.get("unified_selected_count") or 0)
    legacy_n = int(rollout.get("legacy_selected_count") or 0)
    unified_pct = round(unified_n / total_rollout * 100, 2) if total_rollout else 0.0
    legacy_pct = round(legacy_n / total_rollout * 100, 2) if total_rollout else 0.0

    events = get_drift_capture().export(limit=drift_limit)
    path_dist = Counter(e.get("execution_path") or "none" for e in events)
    category_dist = Counter(e.get("path_category") or "UNKNOWN" for e in events)

    hardening_failures = int(production.get("hardening_failure_count") or 0)
    configured_percent = _configured_rollout_percent()

    return {
        "rollout": {
            "configured_rollout_percentage": configured_percent,
            "observed_rollout_percentage": rollout.get("rollout_percentage", configured_percent),
            "unified_traffic_percent": unified_pct,
            "legacy_traffic_percent": legacy_pct,
            "unified_selected_count": unified_n,
            "legacy_selected_count": legacy_n,
            "total_rollout_events": total_rollout,
        },
        "hardening": {
            "hardening_failure_count": hardening_failures,
            "execution_path_none_count": int(production.get("execution_path_none_count") or 0),
            "legacy_fallback_rate": int(production.get("legacy_fallback_rate") or 0),
            "capability_without_model_rate": int(
                production.get("capability_without_model_rate") or 0
            ),
            "ambiguity_by_intent": production.get("ambiguity_rate_by_intent") or {},
        },
        "rollback": rollback.to_dict(),
        "authority": {
            "divergence_count": int(rollout.get("authority_divergence_count") or 0),
            "divergence_rate": rollout.get("authority_divergence_rate", 0.0),
            "rollback_trigger_count": int(rollout.get("rollback_trigger_count") or 0),
            "live_benchmark": live_bench,
        },
        "drift_counters": drift_counters,
        "alerts": alerts,
        "execution_path_distribution": dict(path_dist),
        "path_category_distribution": dict(category_dist),
        "live_path_analytics": path_analytics,
        "drift_capture_size": get_drift_capture().count(),
    }


def dashboard_snapshot_json(*, indent: int = 2) -> str:
    return json.dumps(build_rollout_dashboard_snapshot(), indent=indent)


__all__ = [
    "build_rollout_dashboard_snapshot",
    "dashboard_snapshot_json",
]
