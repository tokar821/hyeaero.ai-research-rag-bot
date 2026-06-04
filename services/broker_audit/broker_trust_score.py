"""
Phase 51 — composite broker trust score (diagnostics only).

Aggregates authority, recommendation, budget, mission, and consistency signals
from existing metadata. Does not alter answers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from services.broker_audit.broker_trace import build_broker_trace


def _subscore_from_quality(data_used: Dict[str, Any]) -> float:
    blob = data_used.get("broker_quality_score") or {}
    if isinstance(blob, dict) and blob.get("total") is not None:
        try:
            return float(blob["total"])
        except (TypeError, ValueError):
            pass
    return 75.0


def _authority_accuracy(data_used: Dict[str, Any], trace_dict: Dict[str, Any]) -> float:
    if trace_dict.get("authority_selected"):
        if data_used.get("authority_dispatch_safety_fallback"):
            return 40.0
        return 92.0
    if data_used.get("intent_lock"):
        return 85.0
    return 55.0


def _recommendation_accuracy(data_used: Dict[str, Any], trace_dict: Dict[str, Any]) -> float:
    if trace_dict.get("executive_primary"):
        if data_used.get("acquisition_budget_infeasible") and trace_dict.get("executive_primary"):
            return 35.0
        return 90.0
    if data_used.get("executive_broker_layer_applied"):
        return 75.0
    return 50.0


def _budget_accuracy(data_used: Dict[str, Any], trace_dict: Dict[str, Any]) -> float:
    if data_used.get("acquisition_budget_infeasible") or data_used.get("mission_budget_conflict"):
        return 88.0
    if trace_dict.get("budget_detected") is not None:
        return 85.0
    return 70.0


def _mission_accuracy(data_used: Dict[str, Any], trace_dict: Dict[str, Any]) -> float:
    if data_used.get("mission_budget_conflict"):
        return 90.0
    br = data_used.get("broker_reasoning") or {}
    if isinstance(br, dict) and (br.get("mission") or br.get("mission_conflict")):
        return 82.0
    return 78.0


def _consistency_accuracy(data_used: Dict[str, Any]) -> float:
    audit = data_used.get("recommendation_consistency_audit_v2") or {}
    if not isinstance(audit, dict):
        return 80.0
    if audit.get("unjustified_recommendation_drift") or audit.get("recommendation_drift"):
        sev = audit.get("drift_severity")
        if sev == "HIGH":
            return 35.0
        return 55.0
    return 92.0


def compute_broker_trust_score(
    answer: str,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Produce 0–100 trust score with dimensional breakdown.
    """
    du = data_used if isinstance(data_used, dict) else {}
    trace = build_broker_trace(answer, query=query, data_used=du)
    td = trace.to_dict()

    breakdown = {
        "authority_accuracy": round(_authority_accuracy(du, td), 2),
        "recommendation_accuracy": round(_recommendation_accuracy(du, td), 2),
        "budget_accuracy": round(_budget_accuracy(du, td), 2),
        "mission_accuracy": round(_mission_accuracy(du, td), 2),
        "consistency_accuracy": round(_consistency_accuracy(du), 2),
        "quality_score": round(_subscore_from_quality(du), 2),
    }

    weights = {
        "authority_accuracy": 0.2,
        "recommendation_accuracy": 0.25,
        "budget_accuracy": 0.2,
        "mission_accuracy": 0.15,
        "consistency_accuracy": 0.2,
    }
    total = sum(breakdown[k] * weights[k] for k in weights)

    result = {
        "total": round(total, 2),
        "breakdown": breakdown,
        "grade": _grade(total),
        "trace_summary": {
            "authority": td.get("authority_selected"),
            "executive_primary": td.get("executive_primary"),
            "budget_musd": td.get("budget_detected"),
        },
    }
    return result


def _grade(total: float) -> str:
    if total >= 90:
        return "A"
    if total >= 80:
        return "B"
    if total >= 70:
        return "C"
    if total >= 60:
        return "D"
    return "F"


def attach_broker_trust_score(
    answer: str,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    du = data_used if isinstance(data_used, dict) else {}
    result = compute_broker_trust_score(answer, query=query, data_used=du)
    du["broker_trust_score"] = result
    return result


__all__ = ["attach_broker_trust_score", "compute_broker_trust_score"]
