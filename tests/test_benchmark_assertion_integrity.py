"""CI gate: certification benchmark suites must assert semantic correctness."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_benchmark_assertion_audit_passes():
    backend = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/audit_benchmark_assertions.py"],
        cwd=str(backend),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
