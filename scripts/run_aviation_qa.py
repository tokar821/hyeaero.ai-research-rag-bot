#!/usr/bin/env python3
"""
Aviation QA / evaluation runner — automated trust, realism, and tone checks.

Usage:
  python scripts/run_aviation_qa.py
  python scripts/run_aviation_qa.py --category westbound_winter_constraints
  python scripts/run_aviation_qa.py --case asia_003 transatlantic_001
  python scripts/run_aviation_qa.py --out evals/results/aviation_qa_latest.json
"""

from __future__ import annotations

import argparse
import os
import sys

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from evals.aviation_qa.runner import run_aviation_qa_suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Run aviation QA evaluation suite")
    parser.add_argument("--suite", default=os.path.join(_BACKEND_ROOT, "evals", "aviation_mission_suite.json"))
    parser.add_argument("--category", action="append", default=[], help="Filter by category (repeatable)")
    parser.add_argument("--case", action="append", default=[], dest="cases", help="Run specific case IDs")
    parser.add_argument(
        "--out",
        default=os.path.join(_BACKEND_ROOT, "evals", "results", "aviation_qa_latest.json"),
        help="Write full JSON report here",
    )
    parser.add_argument("--mode", default="intelligence", choices=("intelligence", "structured", "full"))
    args = parser.parse_args()

    report = run_aviation_qa_suite(
        suite_path=args.suite,
        case_ids=args.cases or None,
        categories=args.category or None,
        mode=args.mode,
        write_json_path=args.out,
    )

    summary = report.get("summary") or {}
    print(f"QA cases: {summary.get('total_cases')}")
    print(f"Evaluator pass rate: {summary.get('evaluator_pass_rate')}")
    print(f"Mean trust score: {summary.get('mean_trust_score')}")
    print(f"Suite repetition score: {(report.get('repetition') or {}).get('suite_repetition_score')}")
    print(f"Report written: {args.out}")

    plan = report.get("improvement_plan") or {}
    print("\nPriority fixes:")
    for step in (plan.get("next_steps") or [])[:5]:
        print(f"  - {step}")

    failed = int(summary.get("total_cases", 0)) - int(summary.get("evaluator_passed", 0))
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
