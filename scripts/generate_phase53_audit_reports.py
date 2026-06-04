"""Generate Phase 53 audit and recertification markdown reports from live benchmark runs."""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPORTS = BACKEND / "reports"


def _run_pytest(target: str) -> tuple[int, str]:
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
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def _parse_pass_rate(report_path: Path) -> tuple[int, int, float]:
    if not report_path.exists():
        return 0, 0, 0.0
    text = report_path.read_text(encoding="utf-8")
    scenarios = passed = 0
    rate = 0.0
    for line in text.splitlines():
        if "| Scenarios |" in line or "| Cases |" in line:
            m = re.search(r"\|\s*(\d+)\s*\|", line)
            if m:
                scenarios = int(m.group(1))
        if "**Pass rate**" in line or "accuracy |" in line.lower():
            m = re.search(r"([\d.]+)%", line)
            if m:
                rate = float(m.group(1))
        if "| Passed |" in line:
            m = re.search(r"\|\s*(\d+)\s*\|", line)
            if m:
                passed = int(m.group(1))
    if scenarios and not passed and rate:
        passed = int(round(scenarios * rate / 100.0))
    return passed, scenarios, rate


def main() -> int:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    suites = [
        ("tests/e2e/real_aircraft_benchmark.py", "real_aircraft_benchmark_report.md"),
        ("tests/e2e/listing_validation_suite.py", "listing_validation_report.md"),
        ("tests/e2e/market_recommendation_audit.py", "market_recommendation_audit_report.md"),
    ]
    results: dict[str, tuple[int, int, float]] = {}
    for target, report_name in suites:
        _run_pytest(target)
        results[report_name] = _parse_pass_rate(REPORTS / report_name)

    ra_p, ra_n, ra_r = results.get("real_aircraft_benchmark_report.md", (0, 0, 0.0))
    lv_p, lv_n, lv_r = results.get("listing_validation_report.md", (0, 0, 0.0))

    pipeline = f"""# Phase 53 Pipeline Audit

Generated: {ts}

## Root cause of summary vs benchmark mismatch

The file `phase53_production_reality_summary.md` claimed **100%** real-aircraft and listing pass rates while `real_aircraft_benchmark_report.md` and `listing_validation_report.md` showed **79%** and **25%**.

| Issue | Evidence | Root cause |
|-------|----------|------------|
| Pytest green, KPI red | `pytest` reported 100 passed while reports showed 79/100 and 5/20 | `test_real_aircraft_recommendation` and `test_listing_validation` only asserted `path in (e2e, layers)`, **not** recorder `passed` / `correct` |
| Stale executive summary | `phase53_production_reality_summary.md` manually authored | Not wired to `BenchmarkRecorder.write_report()` output |
| Listing inferred SUSPICIOUS | 15/20 cases `inferred=SUSPICIOUS` | `infer_listing_verdict()` applied broad `LISTING_SKEPTICISM_MARKERS` (`below`, `verify`, `bargain`) **before** tier/deal bands; `deal_quality` absent when DB missing |
| Real-aircraft failures | Mission buy routed to `BUY_OR_WAIT` → primary `Timing guidance` | `_BUY_WAIT_RE` matched `should i buy` inside `what should I buy` |
| Impossible listings endorsed | `g700_12m`, `cheap_g650_probe` | `_should_reject_infeasible_acquisition(listing_ok=True)` returned False for all listing queries, blocking infeasible acquisition answers |
| KPI parser fragility | Health dashboard `****79.0%****` | `_read_report_metric()` split markdown tables incorrectly |

## Pipeline map (authoritative)

```
run_phase53_audit.py
  └─ pytest subprocess per suite
       ├─ real_aircraft_benchmark.py → BenchmarkRecorder → reports/real_aircraft_benchmark_report.md
       ├─ listing_validation_suite.py → reports/listing_validation_report.md
       └─ …
  └─ _read_report_metric() → write_health_dashboard()
```

**Summary KPI source (correct):** session-end `BenchmarkRecorder.write_report()` in each suite module.  
**Incorrect KPI source:** hand-edited `phase53_production_reality_summary.md`.

## Fixes applied (Phase 53 recertification)

1. Assert recorder pass bit in benchmark tests (pytest now fails when KPI fails).
2. `decision_intent_detector`: mission buy before `BUY_OR_WAIT`; light-jet budget match.
3. `acquisition_budget_reality` / `executive_broker_layer`: listing-price infeasible path; unicode-safe budget parse; `only have` acquisition reject.
4. `market_intelligence_engine._band_from_catalog_tier` + `market_reality_layer` → populate `deal_quality` without DB.
5. `listing_confidence_analyzer`: `ask < mid * 0.45` → `POTENTIAL_DATA_ERROR`.
6. `recommendation_selector`: mission budget rows; query-focus primary for `Model for $X`; Europe–US G650 boost.
7. `infer_listing_verdict`: tier/deal before skepticism markers.
8. Benchmarks use `prefer_e2e=False` for deterministic layers measurement.

## Current measured KPIs (this run)

| Suite | Passed | Total | Rate |
|-------|--------|-------|------|
| Real Aircraft | {ra_p} | {ra_n} | {ra_r:.1f}% |
| Listing Validation | {lv_p} | {lv_n} | {lv_r:.1f}% |
"""
    (REPORTS / "phase53_pipeline_audit.md").write_text(pipeline, encoding="utf-8")

    recert = f"""# Phase 53 Recertification

Generated: {ts}

## Pass rates

| KPI | Target | Measured | Status |
|-----|--------|----------|--------|
| Real Aircraft Benchmark | ≥95% | **{ra_r:.1f}%** ({ra_p}/{ra_n}) | {"PASS" if ra_r >= 95 else "FAIL"} |
| Listing Validation | ≥90% | **{lv_r:.1f}%** ({lv_p}/{lv_n}) | {"PASS" if lv_r >= 90 else "FAIL"} |

## Remaining failures

See `real_aircraft_benchmark_report.md` and `listing_validation_report.md` scenario sections (FAIL rows).

## Root causes remaining

- **Txn shorthand probes** (`CJ4 for $5M`, `Falcon 7X for $22M`): market-reality layer can replace answer before executive pins query-focus model; executive now applies for `Model for $X` but market prose may omit canonical name.
- **Manufacturer discovery** (`dassault_25m`): category answer may not name Falcon variants in prose.
- **Global 7500 @ $40M**: executive primary selection fixed in selector but answer trace may lag `Challenger 650` in edge paths.

## Commands

```powershell
cd backend
$env:PYTHONPATH = "."
pytest tests/e2e/real_aircraft_benchmark.py tests/e2e/listing_validation_suite.py -q
python scripts/generate_phase53_audit_reports.py
```
"""
    (REPORTS / "phase53_recertification.md").write_text(recert, encoding="utf-8")

    # Update stale summary from live metrics
    summary = f"""# Phase 53 — Production Reality Audit Summary

Generated: {ts.split()[0]}

## KPI Results (measured — live reports)

| KPI | Target | Measured | Status |
|-----|--------|----------|--------|
| Real Aircraft Benchmark | >90% | **{ra_r:.1f}%** ({ra_p}/{ra_n}) | {"Pass" if ra_r >= 90 else "Fail"} |
| Listing Validation | >90% | **{lv_r:.1f}%** ({lv_p}/{lv_n}) | {"Pass" if lv_r >= 90 else "Fail"} |

Regenerate: `python runners/run_phase53_audit.py` or `python scripts/generate_phase53_audit_reports.py`
"""
    (REPORTS / "phase53_production_reality_summary.md").write_text(summary, encoding="utf-8")
    print(f"Wrote pipeline audit and recertification ({ra_r:.1f}% real, {lv_r:.1f}% listing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
