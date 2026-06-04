#!/usr/bin/env python3
"""Phase 53 — production reality audit runner + health dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))


def _run_pytest(target: str) -> str:
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q", "--tb=no"],
        cwd=str(BACKEND),
        env=env,
        capture_output=True,
        text=True,
    )
    return (proc.stdout or "") + (proc.stderr or "")


def _parse_pass_rate(output: str) -> str:
    for line in output.splitlines():
        if "passed" in line and ("failed" in line or "error" in line or line.strip().endswith("passed")):
            return line.strip()
        if line.strip().endswith("passed"):
            return line.strip()
    return output.strip()[-200:] if output else "no output"


def _read_report_metric(report_path: Path, label: str) -> str:
    if not report_path.exists():
        return "n/a"
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if label.lower() in line.lower() and "|" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 2 and parts[0].lower().startswith(label.split()[0].lower()):
                return parts[-1]
            if "**Pass rate**" in line or "accuracy |" in line.lower():
                m = __import__("re").search(r"([\d.]+%)", line)
                if m:
                    return m.group(1)
    return "n/a"


def main() -> int:
    reports = BACKEND / "reports"
    suites = [
        ("tests/e2e/real_aircraft_benchmark.py", "real_aircraft_benchmark_report.md"),
        ("tests/e2e/listing_validation_suite.py", "listing_validation_report.md"),
        ("tests/e2e/tail_investigation_suite.py", "tail_investigation_report.md"),
        ("tests/e2e/market_recommendation_audit.py", "market_recommendation_audit_report.md"),
        ("tests/e2e/production_query_replay_suite.py", "production_query_replay_report.md"),
        ("tests/test_alias_expansion_engine.py", ""),
    ]

    sections: dict[str, str] = {}
    for target, report_name in suites:
        out = _run_pytest(target)
        title = target.split("/")[-1].replace(".py", "")
        sections[title] = f"Pytest: `{_parse_pass_rate(out)}`"
        if report_name:
            metric = _read_report_metric(reports / report_name, "Pass rate") or _read_report_metric(
                reports / report_name, "accuracy"
            )
            sections[title] += f"\n\nReport metric: **{metric}**"

    sections["KPI targets"] = (
        "| KPI | Target | Source |\n|-----|--------|--------|\n"
        "| Real Aircraft Benchmark | >90% | real_aircraft_benchmark_report.md |\n"
        "| Listing Validation | >90% | listing_validation_report.md |\n"
        "| Tail Accuracy | >95% | tail_investigation_report.md |\n"
        "| Recommendation Drift | <3% | production_query_replay_report.md |\n"
        "| Authority Error | <2% | production_query_replay_report.md |\n"
        "| Broker Trust Score | >95 | production_query_replay_report.md |\n"
    )

    from tests.e2e.production_audit_helpers import write_health_dashboard

    write_health_dashboard(sections)
    print("Phase 53 audit complete. See reports/hyeaero_health_dashboard.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
