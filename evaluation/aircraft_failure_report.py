"""
Aircraft failure analysis — identify weak catalog entities from golden evaluation.

Evaluation-only; does not modify production behavior.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List

from evaluation.golden_dataset import GoldenTestCase
from evaluation.unified_evaluator import EvaluationResult


def build_aircraft_failure_report(
    cases: List[GoldenTestCase],
    results: List[EvaluationResult],
) -> Dict[str, Any]:
    by_id = {r.case_id: r for r in results}
    failures: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"failures": 0, "categories": set()})

    for case in cases:
        result = by_id.get(case.id)
        if result is None:
            continue
        failed = not (result.route_correct and result.model_correct and result.behavior_correct)
        if not failed:
            continue
        models = case.expected_models or []
        if not models and case.category in ("MISSION", "BUY_DECISION"):
            continue
        if not models:
            models = ["(unresolved)"]
        for model in models:
            entry = failures[model]
            entry["failures"] += 1
            entry["categories"].add(case.category)

    report: Dict[str, Any] = {}
    for model, data in sorted(failures.items(), key=lambda x: -x[1]["failures"]):
        report[model] = {
            "failures": data["failures"],
            "categories": sorted(data["categories"]),
        }

    return {
        "aircraft_failures": report,
        "total_aircraft_with_failures": len(report),
        "total_failure_events": sum(v["failures"] for v in report.values()),
    }


def format_aircraft_failure_console(report: Dict[str, Any], *, top_n: int = 15) -> str:
    lines = ["=== Aircraft Failure Report ===", ""]
    failures = report.get("aircraft_failures") or {}
    sorted_items = sorted(failures.items(), key=lambda x: -x[1]["failures"])[:top_n]
    for model, data in sorted_items:
        cats = ", ".join(data["categories"])
        lines.append(f"  {model}: {data['failures']} failures [{cats}]")
    lines.append("")
    lines.append(f"Total aircraft with failures: {report.get('total_aircraft_with_failures', 0)}")
    return "\n".join(lines)


def write_aircraft_failure_json(report: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


__all__ = [
    "build_aircraft_failure_report",
    "format_aircraft_failure_console",
    "write_aircraft_failure_json",
]
