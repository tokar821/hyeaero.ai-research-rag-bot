#!/usr/bin/env python3
"""Phase 52 — generate accuracy gap, misrouting, and results reports."""

from __future__ import annotations

import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPORTS = BACKEND / "reports"
sys.path.insert(0, str(BACKEND))

from services.broker_audit.root_cause_analyzer import FailureCause, analyze_root_cause
from tests.e2e.benchmark_audit_helpers import (
    attach_audit_metadata,
    authority_matches,
    budget_close,
    model_in_text,
    any_model_in_text,
)
from tests.e2e.broker_certification_helpers import broker_certify
from tests.e2e.listing_realism_suite import (
    LISTING_SCENARIOS,
    infer_listing_label,
)
from tests.e2e.recommendation_accuracy_suite import RECOMMENDATION_SCENARIOS
from tests.e2e.retrieval_accuracy_suite import RETRIEVAL_SCENARIOS


def _run_retrieval_row(scenario):
    answer, du, path = broker_certify(scenario.query, prefer_e2e=True)
    attach_audit_metadata(answer, scenario.query, du)
    trace = du.get("broker_trace") or {}
    auth = str(du.get("authority_dispatch_kind") or trace.get("authority_selected") or "")
    sources = " ".join(str(s) for s in (trace.get("retrieval_sources") or []))
    auth_ok = authority_matches(auth, scenario.expected_authority_substr) or authority_matches(
        sources, scenario.expected_authority_substr
    )
    pool = list(trace.get("aircraft_detected") or [])
    prim = trace.get("executive_primary")
    if prim:
        pool.insert(0, str(prim))
    top3_targets = list(scenario.expected_aircraft_top3) or (
        [scenario.expected_aircraft_top1] if scenario.expected_aircraft_top1 else []
    )
    top3_ok = not top3_targets or any_model_in_text(top3_targets, " ".join(pool) + answer)
    passed = auth_ok and top3_ok
    rc = analyze_root_cause(
        query=scenario.query,
        answer=answer,
        data_used=du,
        expected_authority=scenario.expected_authority_substr or None,
        failure_reasons=[] if passed else ["authority mismatch"],
    )
    return {
        "id": scenario.scenario_id,
        "query": scenario.query,
        "passed": passed,
        "auth": auth,
        "expected_auth": scenario.expected_authority_substr,
        "pool": pool[:5],
        "expected_aircraft": list(top3_targets),
        "root_cause": rc.cause.value,
        "path": path,
    }


def _run_recommendation_row(scenario):
    answer, du, path = broker_certify(scenario.query, prefer_e2e=True)
    attach_audit_metadata(answer, scenario.query, du)
    trace = du.get("broker_trace") or {}
    primary = str(trace.get("executive_primary") or "")
    if scenario.expect_infeasible:
        ok = du.get("acquisition_budget_infeasible") or answer[:80].lower().startswith(("no.", "not realistically"))
        gap = "budget_reality" if not ok else ""
    elif scenario.expected_primary:
        ok = model_in_text(scenario.expected_primary, primary) or model_in_text(scenario.expected_primary, answer)
        gap = "aircraft_family" if not ok else ""
    else:
        ok = any_model_in_text(scenario.expected_alternatives, answer)
        gap = "aircraft_family" if not ok else ""
    rc = analyze_root_cause(
        query=scenario.query,
        answer=answer,
        data_used=du,
        expected_primary=scenario.expected_primary or None,
        expect_infeasible=scenario.expect_infeasible,
        failure_reasons=[] if ok else [gap or "recommendation"],
    )
    return {
        "id": scenario.scenario_id,
        "query": scenario.query,
        "passed": ok,
        "primary": primary,
        "expected": scenario.expected_primary or str(scenario.expected_alternatives),
        "gap": gap,
        "root_cause": rc.cause.value,
        "path": path,
    }


def write_gap_analysis(ret_rows, rec_rows) -> Path:
    lines = [
        "# Phase 52 — Accuracy Gap Analysis",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]
    for section, rows in ("Retrieval", ret_rows), ("Recommendation", rec_rows):
        lines.append(f"## {section} failures\n")
        for r in rows:
            if r["passed"]:
                continue
            lines.extend(
                [
                    f"### {r['id']}",
                    f"- **Query:** {r['query']}",
                    f"- **Root cause:** `{r['root_cause']}`",
                ]
            )
            if "auth" in r:
                lines.append(f"- **Selected authority:** `{r['auth']}`")
                lines.append(f"- **Expected authority:** `{r['expected_auth']}`")
                lines.append(f"- **Selected aircraft:** {r['pool']}")
                lines.append(f"- **Expected aircraft:** {r['expected_aircraft']}")
            else:
                lines.append(f"- **Selected primary:** `{r['primary']}`")
                lines.append(f"- **Expected:** `{r['expected']}`")
                lines.append(f"- **Gap type:** {r.get('gap') or 'n/a'}")
            lines.append("")
    out = REPORTS / "phase52_accuracy_gap_analysis.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_misrouting_matrix(ret_rows) -> Path:
    matrix: Counter[tuple[str, str, str]] = Counter()
    for r in ret_rows:
        matrix[(r["query"], r["expected_auth"], r["auth"])] += 1
    lines = [
        "# Authority Misrouting Matrix (Phase 52)",
        "",
        "| Query | Expected authority | Actual authority | Frequency |",
        "|-------|-------------------|------------------|-----------|",
    ]
    for (q, exp, act), freq in sorted(matrix.items(), key=lambda x: -x[1]):
        lines.append(f"| {q[:60]} | {exp} | {act or '(none)'} | {freq} |")
    out = REPORTS / "authority_misrouting_matrix.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_recommendation_gap_matrix(rec_rows) -> Path:
    lines = [
        "# Recommendation Gap Matrix (Phase 52)",
        "",
        "| Scenario | Gap | Selected | Expected | Root cause |",
        "|----------|-----|----------|----------|------------|",
    ]
    for r in rec_rows:
        gap = r.get("gap") or ("ok" if r["passed"] else "unknown")
        lines.append(
            f"| {r['id']} | {gap} | {r['primary'][:40]} | {str(r['expected'])[:40]} | {r['root_cause']} |"
        )
    out = REPORTS / "recommendation_gap_matrix.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    ret_rows = [_run_retrieval_row(s) for s in RETRIEVAL_SCENARIOS]
    rec_rows = [_run_recommendation_row(s) for s in RECOMMENDATION_SCENARIOS]

    write_gap_analysis(ret_rows, rec_rows)
    write_misrouting_matrix(ret_rows)
    write_recommendation_gap_matrix(rec_rows)

    ret_pass = sum(1 for r in ret_rows if r["passed"]) / len(ret_rows) * 100
    rec_pass = sum(1 for r in rec_rows if r["passed"]) / len(rec_rows) * 100

    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/e2e/retrieval_accuracy_suite.py",
            "tests/e2e/recommendation_accuracy_suite.py",
            "-q",
        ],
        cwd=str(BACKEND),
        env=env,
        check=False,
    )

    cert = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/e2e/test_broker_certification_v2.py",
            "-q",
            "--tb=no",
        ],
        cwd=str(BACKEND),
        env=env,
        capture_output=True,
        text=True,
    )
    cert_tail = (cert.stdout or "") + (cert.stderr or "")

    results = [
        "# Phase 52 Results",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Benchmark pass rates (audit runner)",
        "",
        f"| Suite | Pass rate | Target |",
        f"|-------|-----------|--------|",
        f"| Retrieval accuracy | {ret_pass:.1f}% | >80% |",
        f"| Recommendation accuracy | {rec_pass:.1f}% | >80% |",
        "",
        "## Certification V2",
        "",
        "```",
        cert_tail.strip()[-500:],
        "```",
        "",
        "## Reports generated",
        "",
        "- `phase52_accuracy_gap_analysis.md`",
        "- `authority_misrouting_matrix.md`",
        "- `recommendation_gap_matrix.md`",
        "- `retrieval_accuracy_report.md` (pytest)",
        "- `recommendation_accuracy_report.md` (pytest)",
        "",
    ]
    (REPORTS / "phase52_results.md").write_text("\n".join(results), encoding="utf-8")
    print(f"Retrieval: {ret_pass:.1f}%  Recommendation: {rec_pass:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
