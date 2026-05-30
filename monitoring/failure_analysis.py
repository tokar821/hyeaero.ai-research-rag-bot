"""
Live failure analysis — top failing aircraft and execution paths from drift capture.

Operational reporting only.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List

from monitoring.drift_capture import get_drift_capture


def build_live_failure_reports(*, top_n: int = 10) -> Dict[str, Any]:
    """Derive top failing aircraft and path categories from captured drift events."""
    events = get_drift_capture().export()
    aircraft_failures: Counter[str] = Counter()
    aircraft_categories: Dict[str, set[str]] = defaultdict(set)
    path_failures: Counter[str] = Counter()
    category_failures: Counter[str] = Counter()

    for ev in events:
        flags = ev.get("hardening_flags") or {}
        failed = bool(flags.get("routing_failure")) or bool(flags.get("fallback_triggered"))
        if not failed:
            continue
        path = ev.get("execution_path") or "none"
        cat = ev.get("path_category") or "UNKNOWN"
        path_failures[path] += 1
        category_failures[cat] += 1
        model = ev.get("model")
        if model:
            aircraft_failures[str(model)] += 1
            aircraft_categories[str(model)].add(cat)

    top_aircraft = [
        {
            "model": model,
            "failures": count,
            "categories": sorted(aircraft_categories.get(model, set())),
        }
        for model, count in aircraft_failures.most_common(top_n)
    ]
    top_paths = [
        {"execution_path": path, "failures": count}
        for path, count in path_failures.most_common(top_n)
    ]
    top_categories = [
        {"category": cat, "failures": count}
        for cat, count in category_failures.most_common(top_n)
    ]

    return {
        "top_failing_aircraft": top_aircraft,
        "top_failing_execution_paths": top_paths,
        "top_failing_categories": top_categories,
    }


__all__ = ["build_live_failure_reports"]
