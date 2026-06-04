# Phase 53 Recertification

Generated: 2026-06-03 10:45 UTC

## Pass rates

| KPI | Target | Measured | Status |
|-----|--------|----------|--------|
| Real Aircraft Benchmark | ≥95% | **100.0%** (100/100) | PASS |
| Listing Validation | ≥90% | **100.0%** (20/20) | PASS |

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
