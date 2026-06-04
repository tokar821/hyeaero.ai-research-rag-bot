#!/usr/bin/env python3
"""Phase 54 — benchmark integrity + hardening runner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def main() -> int:
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND)

    steps = [
        ([sys.executable, "scripts/audit_benchmark_assertions.py"], "assertion audit"),
        (
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_acquisition_tier_catalog.py",
                "tests/test_benchmark_assertion_integrity.py",
                "tests/e2e/test_execution_path_parity.py",
                "tests/e2e/listing_format_coverage_suite.py",
                "tests/verification/test_failure_injection.py",
                "-q",
                "--tb=no",
            ],
            "fast hardening tests",
        ),
        (
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/e2e/real_aircraft_benchmark.py",
                "tests/e2e/listing_validation_suite.py",
                "tests/e2e/market_recommendation_audit.py",
                "-q",
                "--tb=no",
            ],
            "certification suites",
        ),
    ]

    limit = env.get("PHASE53_REPLAY_LIMIT", "")
    if limit != "0":
        steps.append(
            (
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/e2e/production_query_replay_suite.py",
                    "-q",
                    "--tb=no",
                ],
                "production replay",
            )
        )

    for cmd, label in steps:
        print(f"--- {label} ---")
        proc = subprocess.run(cmd, cwd=str(BACKEND), env=env)
        if proc.returncode != 0:
            print(f"FAILED: {label}")
            return proc.returncode

    subprocess.run([sys.executable, "scripts/generate_phase54_audit_reports.py"], cwd=str(BACKEND), env=env)
    print("Phase 54 audit complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
