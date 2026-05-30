#!/usr/bin/env python3
"""Phase 8 golden dataset evaluation runner — measurement only, no production changes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.aircraft_failure_report import (
    build_aircraft_failure_report,
    format_aircraft_failure_console,
    write_aircraft_failure_json,
)
from evaluation.golden_dataset import dataset_summary, load_golden_cases
from evaluation.legacy_unified_benchmark import benchmark_cases, benchmark_summary
from evaluation.path_accuracy_report import (
    build_path_accuracy_report,
    format_path_accuracy_console,
    write_path_accuracy_json,
)
from evaluation.rollout_readiness import compute_rollout_readiness_from_evaluation
from evaluation.unified_evaluator import evaluate_unified_cases
from services.routing.unified_intent_production_metrics import get_production_metrics
from services.telemetry.unified_rollout_telemetry import get_rollout_telemetry_snapshot


def run_phase8_evaluation(*, output_dir: Path | None = None) -> int:
    out = output_dir or (_ROOT / "evals" / "phase8")
    out.mkdir(parents=True, exist_ok=True)

    cases = load_golden_cases()
    summary = dataset_summary(cases)
    print(f"Loaded {summary['total']} golden cases")
    for cat in sorted(k for k in summary if k != "total"):
        print(f"  {cat}: {summary[cat]}")

    print("\nRunning unified evaluation...")
    unified_results = evaluate_unified_cases(cases, enforce=True)

    print("Running legacy vs unified benchmark...")
    benchmark_results = benchmark_cases(cases, enforce_unified=True)
    bench_summary = benchmark_summary(benchmark_results)

    path_report = build_path_accuracy_report(cases, unified_results)
    aircraft_report = build_aircraft_failure_report(cases, unified_results)

    rollback_metrics = {
        **get_production_metrics(),
        **get_rollout_telemetry_snapshot(),
    }
    readiness = compute_rollout_readiness_from_evaluation(
        cases,
        unified_results,
        benchmark_results,
        rollback_metrics=rollback_metrics,
    )

    full_report = {
        "dataset_summary": summary,
        "path_accuracy": path_report,
        "aircraft_failures": aircraft_report,
        "legacy_unified_benchmark": bench_summary,
        "rollout_readiness": readiness.to_dict(),
        "rollback_metrics_snapshot": rollback_metrics,
    }

    report_path = out / "phase8_evaluation_report.json"
    report_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")

    write_path_accuracy_json(path_report, str(out / "path_accuracy_report.json"))
    write_aircraft_failure_json(aircraft_report, str(out / "aircraft_failure_report.json"))
    (out / "benchmark_summary.json").write_text(
        json.dumps(bench_summary, indent=2), encoding="utf-8"
    )

    print("\n" + format_path_accuracy_console(path_report))
    print("\n" + format_aircraft_failure_console(aircraft_report))
    print("\n=== Legacy vs Unified Benchmark ===")
    print(json.dumps(bench_summary, indent=2))
    print("\n=== Rollout Readiness ===")
    print(json.dumps(readiness.to_dict(), indent=2))
    print(f"\nFull report written to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_phase8_evaluation())
