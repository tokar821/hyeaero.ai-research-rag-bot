"""Phase 32 — Report generation for broker QA and executive readiness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def generate_broker_review(report: Dict[str, Any], path: Path | None = None) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = path or REPORTS_DIR / "production_broker_review.md"
    lines = [
        "# Production Broker QA Review Pack",
        "",
        "## Summary",
        "",
        f"- **Broker Quality Score:** {report.get('broker_quality_score', 0)}",
        f"- **Routing Accuracy:** {report.get('routing_accuracy_pct', 0)}%",
        f"- **Dispatch Accuracy:** {report.get('dispatch_accuracy_pct', 0)}%",
        f"- **Hallucination Rate:** {report.get('hallucination_rate_pct', 0)}%",
        "",
        "---",
        "",
        "## Top 50 Highest-Confidence Outputs",
        "",
    ]
    for i, row in enumerate(report.get("high_confidence_queries") or [], 1):
        lines.extend([
            f"### {i}. `{row.get('query_id')}`",
            "",
            f"**Query:** {row.get('query')}",
            "",
            f"**Recommendation:** {row.get('recommendation', '')[:200]}",
            "",
            f"**Dispatch path:** `{row.get('execution_path')}`",
            "",
            f"**IntentLock:** {row.get('intent_lock_summary', '')}",
            "",
            f"**Authority models:** {', '.join(row.get('models') or [])}",
            "",
        ])

    lines.extend(["---", "", "## Top 50 Highest-Risk Outputs", ""])
    for i, row in enumerate(report.get("high_risk_queries") or [], 1):
        lines.extend([
            f"### {i}. `{row.get('query_id')}`",
            "",
            f"**Query:** {row.get('query')}",
            "",
            f"**Recommendation:** {row.get('recommendation', '')[:200]}",
            "",
            f"**Dispatch path:** `{row.get('execution_path')}`",
            "",
            f"**IntentLock:** {row.get('intent_lock_summary', '')}",
            "",
            f"**Authority models:** {', '.join(row.get('models') or [])}",
            "",
            f"**Issues:** {', '.join(row.get('issues') or [])}",
            "",
        ])

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def generate_readiness_report(report: Dict[str, Any], path: Path | None = None) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = path or REPORTS_DIR / "production_readiness_report.md"
    scores = report.get("category_scores") or {}
    criteria = [
        ("Routing accuracy", report.get("routing_accuracy_pct", 0), 99.0),
        ("Authority / dispatch accuracy", report.get("dispatch_accuracy_pct", 0), 99.0),
        ("Mission fit accuracy", report.get("mission_fit_accuracy_pct", 0), 95.0),
        ("Hallucination rate", report.get("hallucination_rate_pct", 0), 1.0, True),
        ("Fail-closed accuracy", report.get("fail_closed_accuracy_pct", 0), 100.0),
        ("Broker quality score", report.get("broker_quality_score", 0), 90.0),
    ]
    lines = [
        "# Production Readiness Report",
        "",
        "## Executive Metrics",
        "",
        "| Metric | Value | Target | Status |",
        "|--------|-------|--------|--------|",
    ]
    for item in criteria:
        if len(item) == 4:
            name, val, target, lower_is_better = item
            if lower_is_better:
                status = "PASS" if val < target else "FAIL"
                lines.append(f"| {name} | {val}% | <{target}% | {status} |")
            else:
                status = "PASS" if val >= target else "FAIL"
                lines.append(f"| {name} | {val}% | >={target}% | {status} |")
        else:
            name, val, target = item
            status = "PASS" if val >= target else "FAIL"
            lines.append(f"| {name} | {val} | >={target} | {status} |")

    lines.extend([
        "",
        "## Category Scores",
        "",
        "| Category | Score |",
        "|----------|-------|",
    ])
    for k, v in scores.items():
        lines.append(f"| {k.replace('_', ' ').title()} | {v} |")

    lines.extend([
        "",
        "## Category Breakdown",
        "",
        "```json",
        json.dumps(report.get("category_breakdown") or {}, indent=2),
        "```",
        "",
        "## Hallucination Audit",
        "",
        f"- Audited: {report.get('hallucination_audit', {}).get('total_audited', 0)}",
        f"- Flagged: {report.get('hallucination_audit', {}).get('hallucination_count', 0)}",
        "",
        "## Mission Fit Audit",
        "",
        f"- Mission queries: {report.get('mission_fit_audit', {}).get('total_mission_queries', 0)}",
        f"- Fit accuracy: {report.get('mission_fit_accuracy_pct', 0)}%",
        "",
        f"**Total queries validated:** {report.get('total_queries', 0)}",
        "",
    ])
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def generate_all_reports(report: Dict[str, Any]) -> Dict[str, str]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "production_validation_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    broker = generate_broker_review(report)
    readiness = generate_readiness_report(report)
    return {
        "validation_json": str(json_path),
        "broker_review": str(broker),
        "readiness_report": str(readiness),
    }
