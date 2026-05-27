#!/usr/bin/env python3
"""
Production aviation mission benchmark runner.

Usage:
  python scripts/run_aviation_mission_benchmark.py
  python scripts/run_aviation_mission_benchmark.py --category runway_flexibility
  python scripts/run_aviation_mission_benchmark.py --out results/aviation_benchmark.json
  python scripts/run_aviation_mission_benchmark.py --fail-fast
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(_HERE)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from evals.aviation_benchmark_runner import run_aviation_mission_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Aviation mission benchmark suite")
    parser.add_argument(
        "--suite",
        default=os.path.join(_BACKEND_ROOT, "evals", "aviation_mission_suite.json"),
        help="Path to benchmark JSON",
    )
    parser.add_argument("--category", action="append", help="Filter by category (repeatable)")
    parser.add_argument("--id", action="append", dest="case_ids", help="Run specific case id(s)")
    parser.add_argument(
        "--mode",
        choices=("intelligence", "structured", "full"),
        default="intelligence",
        help="Evaluation mode (default: intelligence, no API)",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(_BACKEND_ROOT, "evals", "results", "aviation_benchmark_latest.json"),
        help="Write JSON report to path",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit 1 if any case fails or has automated failures",
    )
    args = parser.parse_args()

    report = run_aviation_mission_benchmark(
        suite_path=args.suite,
        case_ids=args.case_ids,
        categories=args.category,
        mode=args.mode,
        write_json_path=args.out,
    )

    summary = report.get("summary") or {}
    print(json.dumps(summary, indent=2))
    print(f"\nDimension scores: {json.dumps(report.get('dimension_scores'), indent=2)}")
    print(f"Contamination: {json.dumps(report.get('contamination_report'), indent=2)}")
    print(f"Realism score: {report.get('realism_score')}")
    print(f"Diversity score: {report.get('aircraft_diversity_score')}")
    print(f"Recommendation precision: {report.get('recommendation_precision')}")
    print(f"\nReport written: {args.out}")

    if args.fail_fast:
        if summary.get("failed", 0) > 0:
            return 1
        for case in report.get("failed_cases") or []:
            if case.get("automated_failures"):
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
