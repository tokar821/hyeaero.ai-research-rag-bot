"""
Phase 51 — shared helpers for retrieval / recommendation / listing benchmark suites.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def model_in_text(model: str, text: str) -> bool:
    if not model or not text:
        return False
    compact = text.lower().replace(" ", "")
    full = model.lower().replace(" ", "")
    if full in compact:
        return True
    token = model.split()[-1].lower().replace(" ", "") if " " in model else ""
    return bool(token and len(token) >= 4 and token in compact)


def any_model_in_text(models: Sequence[str], text: str) -> bool:
    return any(model_in_text(m, text) for m in models)


def authority_matches(actual: str, expected_substr: str) -> bool:
    act = _normalize(actual)
    exp = _normalize(expected_substr)
    if not exp:
        return True
    return exp in act or act == exp


def attach_audit_metadata(answer: str, query: str, data_used: dict) -> None:
    """Ensure trace and trust score exist on data_used (diagnostics only)."""
    try:
        from services.broker_audit.broker_trace import attach_broker_trace
        from services.broker_audit.broker_trust_score import attach_broker_trust_score

        attach_broker_trace(answer, query=query, data_used=data_used)
        attach_broker_trust_score(answer, query=query, data_used=data_used)
    except Exception:
        pass


@dataclass
class BenchmarkRow:
    scenario_id: str
    passed: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class BenchmarkRecorder:
    title: str
    report_filename: str
    rows: List[BenchmarkRow] = field(default_factory=list)

    def record(self, row: BenchmarkRow) -> None:
        self.rows.append(row)

    def write_report(self, summary_lines: List[str], path: Optional[Path] = None) -> Path:
        out = path or (REPORTS_DIR / self.report_filename)
        out.parent.mkdir(parents=True, exist_ok=True)
        total = len(self.rows)
        passed = sum(1 for r in self.rows if r.passed)
        rate = (100.0 * passed / total) if total else 0.0

        lines = [
            f"# {self.title}",
            "",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Scenarios | {total} |",
            f"| Passed | {passed} |",
            f"| **Pass rate** | **{rate:.1f}%** |",
            "",
        ]
        lines.extend(summary_lines)
        lines.append("")
        lines.append("## Scenario results")
        lines.append("")
        for r in self.rows:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"- **{r.scenario_id}** [{status}]: {r.notes or r.metrics}")
        lines.extend(
            [
                "",
                "## Regenerate",
                "",
                "```bash",
                "cd backend",
                f"PYTHONPATH=. pytest tests/e2e/{self.report_filename.replace('.md', '_suite.py').replace('_report', '')} -q",
                "```",
            ]
        )
        out.write_text("\n".join(lines), encoding="utf-8")
        return out


def budget_close(detected: Optional[float], expected: Optional[float], tol: float = 1.0) -> bool:
    if expected is None:
        return True
    if detected is None:
        return False
    return abs(float(detected) - float(expected)) <= tol
