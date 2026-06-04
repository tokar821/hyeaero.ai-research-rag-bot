# Phase 53 Pipeline Audit

Generated: 2026-06-03 10:45 UTC

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
| Real Aircraft | 100 | 100 | 100.0% |
| Listing Validation | 20 | 20 | 100.0% |
