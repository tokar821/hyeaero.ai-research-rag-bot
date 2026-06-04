#!/usr/bin/env python3
"""Phase 54 — aggregate hardening audit reports."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPORTS = BACKEND / "reports"


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(BACKEND), capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def main() -> int:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections: list[str] = []

    code, out = _run([sys.executable, "scripts/audit_benchmark_assertions.py"])
    sections.append(f"### Benchmark assertion audit\n\nExit: {code}\n\n```\n{out.strip()}\n```")

    parity_report = REPORTS / "phase54_execution_path_parity.md"
    parity_excerpt = ""
    if parity_report.exists():
        parity_excerpt = parity_report.read_text(encoding="utf-8")[-1200:]

    code2, out2 = _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/e2e/test_execution_path_parity.py",
            "tests/test_acquisition_tier_catalog.py",
            "tests/test_benchmark_assertion_integrity.py",
            "-q",
            "--tb=no",
        ]
    )
    sections.append(f"### Parity + tier checksum tests\n\nExit: {code2}\n\n```\n{out2.strip()[-500:]}\n```")
    if parity_excerpt:
        sections.append(f"### Execution path parity (excerpt)\n\n```\n{parity_excerpt}\n```")

    body = "\n\n".join(
        [
            "# Phase 54 Hardening Audit Summary",
            "",
            f"Generated: {ts}",
            "",
            *sections,
            "",
            "## Regenerate",
            "",
            "```powershell",
            "cd backend",
            "$env:PYTHONPATH = '.'",
            "python scripts/generate_phase54_audit_reports.py",
            "python runners/run_phase54_audit.py",
            "```",
        ]
    )
    (REPORTS / "phase54_hardening_summary.md").write_text("\n".join(body), encoding="utf-8")
    print("Wrote reports/phase54_hardening_summary.md")
    return 0 if code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
