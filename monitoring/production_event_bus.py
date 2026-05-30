"""
Production event bus — ingest live consultant turns into Phase 9 monitoring stores.

Operational instrumentation only — does not alter routing or gate decisions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from monitoring.drift_capture import DriftEvent, capture_drift_event
from monitoring.live_benchmark import compare_live_metadata
from monitoring.live_path_analytics import get_live_path_analytics, infer_path_category


def _dig(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    val = data.get(key, default)
    return default if val is None else val


def ingest_consultant_turn(
    data_used: Dict[str, Any],
    *,
    query: str,
    qri_intent: str = "",
    latency_ms: float = 0.0,
    unified_selected: bool = False,
    unified_enforced: bool = False,
    execution_path: str = "",
    legacy_output_length: int = 0,
    unified_output_length: int = 0,
    legacy_latency_ms: float = 0.0,
) -> None:
    """
  Ingest observability payload from a consultant turn into drift capture and analytics.

  Safe to call on every turn; failures are swallowed by caller.
  """
    shadow = _dig(data_used, "unified_intent_shadow", {})
    if not isinstance(shadow, dict):
        shadow = {}

    hardening_flags = _dig(shadow, "hardening_flags", {})
    if not isinstance(hardening_flags, dict):
        hardening_flags = {}

    rollback = _dig(data_used, "unified_emergency_rollback", {})
    if not isinstance(rollback, dict):
        rollback = {}

    route = data_used.get("unified_intent_shadow") or shadow
    resolved_path = (execution_path or "").strip()
    model = None
    if isinstance(route, dict):
        if not resolved_path:
            resolved_path = str(
                route.get("unified_execution_path")
                or route.get("router_execution_path")
                or ""
            )
        model = route.get("model_resolved") or route.get("unified_model")

    if not resolved_path:
        du_path = data_used.get("unified_execution_path")
        if du_path:
            resolved_path = str(du_path)

    if not resolved_path:
        telem = _dig(data_used, "unified_intent_telemetry", {})
        if isinstance(telem, dict):
            resolved_path = str(telem.get("execution_path") or "")

    comparison = _dig(data_used, "unified_authority_comparison", {})
    authority_aligned = bool(
        isinstance(comparison, dict) and comparison.get("aligned")
    )

    category = infer_path_category(resolved_path or "none", qri_intent=qri_intent)
    fallback = not unified_enforced or bool(
        hardening_flags.get("fallback_triggered")
    )

    get_live_path_analytics().record(
        category,
        unified_enforced=unified_enforced,
        fallback=fallback,
        latency_ms=latency_ms,
    )

    capture_drift_event(
        DriftEvent(
            query=query or "",
            execution_path=resolved_path or "none",
            model=model if isinstance(model, str) else None,
            hardening_flags={
                "routing_failure": bool(hardening_flags.get("routing_failure")),
                "ambiguity_detected": bool(hardening_flags.get("ambiguity_detected")),
                "fallback_triggered": bool(hardening_flags.get("fallback_triggered")),
            },
            rollback_status=rollback,
            qri_intent=qri_intent or "",
            rollout_enabled=bool(unified_selected),
            unified_enforced=bool(unified_enforced),
            path_category=category,
        )
    )

    compare_live_metadata(
        unified_execution_path=resolved_path or "none",
        legacy_qri_intent=qri_intent or str(shadow.get("qri_intent") or ""),
        authority_aligned=authority_aligned,
        unified_latency_ms=latency_ms if unified_enforced else 0.0,
        legacy_latency_ms=legacy_latency_ms,
        unified_output_length=unified_output_length,
        legacy_output_length=legacy_output_length,
    )


__all__ = ["ingest_consultant_turn"]
