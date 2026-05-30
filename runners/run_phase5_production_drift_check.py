#!/usr/bin/env python3
"""Phase 5 production drift check — golden-set routing authority validation."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.recommendation.query_recommendation_intent import classify_query_recommendation_intent
from services.routing.unified_intent_drift_monitor import (
    detect_intent_drift,
    is_critical_router_gate_drift,
)
from services.routing.unified_intent_router import UnifiedExecutionPath, classify_unified_intent
from services.routing.unified_pipeline_gate import evaluate_pipeline_gate


@dataclass(frozen=True)
class GoldenCase:
    query: str
    expected_path: UnifiedExecutionPath
    label: str


GOLDEN_SET: Tuple[GoldenCase, ...] = (
    GoldenCase("How many seats does a Falcon 8X have?", UnifiedExecutionPath.AIRCRAFT_FACT, "fact_seats"),
    GoldenCase("What is a Challenger 3500 worth?", UnifiedExecutionPath.AIRCRAFT_MARKET_FACT, "market_worth"),
    GoldenCase(
        "Can a Falcon 8X fly nonstop from New York to London?",
        UnifiedExecutionPath.CAPABILITY,
        "capability_route",
    ),
    GoldenCase(
        "Compare Challenger 650 vs Praetor 600",
        UnifiedExecutionPath.COMPARISON,
        "comparison_explicit",
    ),
    GoldenCase(
        "What are credible alternatives to a Gulfstream G650?",
        UnifiedExecutionPath.ALTERNATIVE,
        "alternative_replacement",
    ),
    GoldenCase(
        "Recommend the best jet for New York to London",
        UnifiedExecutionPath.NONE,
        "mission_legacy",
    ),
)


def _check_case(case: GoldenCase) -> Dict[str, Any]:
    route = classify_unified_intent(case.query)
    qri = classify_query_recommendation_intent(case.query)
    gate = evaluate_pipeline_gate(
        route,
        enforce_fact=True,
        enforce_capability=True,
        enforce_comparison=True,
        enforce_alternative=True,
    )
    drift = detect_intent_drift(route, qri_intent=qri.intent.value, gate_execution_path=gate.execution_path.value)
    return {
        "label": case.label,
        "query": case.query,
        "expected_path": case.expected_path.value,
        "actual_path": route.execution_path.value,
        "gate_path": gate.execution_path.value,
        "qri_intent": qri.intent.value,
        "path_match": route.execution_path == case.expected_path,
        "router_gate_aligned": route.execution_path.value == gate.execution_path.value,
        "drift": drift,
    }


def run_phase5_production_drift_check() -> int:
    results: List[Dict[str, Any]] = []
    failures: List[str] = []
    critical_drifts: List[str] = []

    for case in GOLDEN_SET:
        row = _check_case(case)
        results.append(row)

        if not row["path_match"]:
            failures.append(f"{case.label}: expected {case.expected_path.value}, got {row['actual_path']}")

        if is_critical_router_gate_drift(row["actual_path"], row["gate_path"]):
            critical_drifts.append(f"FLAG_CRITICAL_DRIFT:{case.label}")

        if case.expected_path == UnifiedExecutionPath.AIRCRAFT_FACT and row["actual_path"] == UnifiedExecutionPath.NONE:
            failures.append(f"FAIL_BUILD:fact_query_lost_authority:{case.label}")

        if case.expected_path == UnifiedExecutionPath.CAPABILITY and row["actual_path"] == UnifiedExecutionPath.COMPARISON:
            failures.append(f"FAIL_BUILD:capability_to_comparison_leak:{case.label}")

        if case.expected_path == UnifiedExecutionPath.COMPARISON and row["actual_path"] == UnifiedExecutionPath.NONE:
            if case.label == "comparison_explicit":
                failures.append(f"FAIL_BUILD:comparison_lost_authority:{case.label}")

    aligned = sum(1 for r in results if r["path_match"])
    alignment_rate = aligned / len(results) if results else 0.0

    report = {
        "golden_cases": len(results),
        "alignment_rate": round(alignment_rate, 4),
        "failures": failures,
        "critical_drifts": critical_drifts,
        "results": results,
    }
    print(json.dumps(report, indent=2))

    if failures:
        return 1
    if critical_drifts:
        return 2
    if alignment_rate < 1.0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(run_phase5_production_drift_check())
