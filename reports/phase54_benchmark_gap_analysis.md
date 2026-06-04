# Phase 54 — Benchmark Coverage Gap Analysis

Generated: 2026-06-03

Scope: `backend/tests/e2e/` and related measurement suites.

---

## Suite inventory

| Suite | Cases | Asserts quality? | `prefer_e2e` | Report |
|-------|-------|-----------------|--------------|--------|
| `real_aircraft_benchmark.py` | 100 | **Yes** (`assert passed`) | **False** | `real_aircraft_benchmark_report.md` |
| `listing_validation_suite.py` | 20 | **Yes** (`assert correct`) | **False** | `listing_validation_report.md` |
| `market_recommendation_audit.py` | 60 subset | **No pytest assert** | True | `market_recommendation_audit_report.md` |
| `tail_investigation_suite.py` | 6 | Partial (registry only) | True | `tail_investigation_report.md` |
| `production_query_replay_suite.py` | 500 | Authority only | True | `production_query_replay_report.md` |
| `listing_realism_suite.py` | 5 | **No** (path only) | True | `listing_realism_report.md` |
| `recommendation_accuracy_suite.py` | 6 | **Yes** | True | `listing_realism_report.md` |
| `retrieval_accuracy_suite.py` | 7 | **Yes** | True | `retrieval_accuracy_report.md` |
| `test_broker_certification_v2.py` | many | Mixed | Mixed | certification V2 |

---

## Missing aircraft categories

`_ACQUISITION_TIER_MUSD` has **19 models**. Not covered by name in Phase 53 listing/real benchmarks:

| Gap | Examples | Risk |
|-----|----------|------|
| No HondaJet / HA-420 | HondaJet Elite | Alias may map incorrectly |
| No Global 5500 / 6000 | Global 5500 vs 6500 | Band collapse to nearest tier |
| No Falcon 6X / 10X | New Dassault | Registry lock may reject |
| No Citation X / X+ | Cessna large cabin | Budget matcher may skip |
| No Embraer Legacy 500/600 | Legacy line | Praetor-only coverage |
| No King Air / turboprop | Cheapest “jet” queries | Mission/budget misroute |
| Manufacturer-only “best jet” | Already partially covered | Category prose without model names |

---

## Missing transaction scenarios

| Scenario type | Covered? | Gap |
|---------------|----------|-----|
| Year + ask deal | Partial (`txn_2018_*` in real aircraft) | No depreciation curve by year |
| Off-market / whisper | No | — |
| Dual ask (range) | No | `$8M–$10M` parsing |
| Fractional / lease | No | — |
| ADS-B / 2020 compliant premium | No | — |
| Engine program fresh vs run-out | No | — |
| Damage history discount | No | — |
| Multi-model portfolio sell | No | — |
| Currency EUR / GBP ask | No | USD-only parsers |
| Ask without model word order | Partial | `"$42M G650"` variants |

---

## Missing listing edge cases

| Edge case | In `listing_validation_suite`? |
|-----------|-------------------------------|
| Unicode em dash in price | Fixed in code; **not in suite** |
| Missing model (“$42M — fair?”) | No |
| Ask in thousands (`$4,200k`) | No |
| Ask without `$` (`42 million`) | No |
| Broker “reduced from” | No |
| NNN / exclusive mandate | No |
| Fresh listing vs 400 DOM | No |
| Overpriced vs fair boundary (±15%) | 20 cases only |
| GOOD_DEAL vs SUSPICIOUS boundary | Partial |
| WAIVED / seller financing | No |
| Two models in one ask | No |

`listing_realism_suite.py` (5 cases) is **stale** relative to Phase 53 suite and still **path-only assert**.

---

## Missing tail investigations

| Coverage | Count |
|----------|-------|
| Registered tails in suite | **6** (N650GS, N800XX, N525AB, N200QS, N44PJ) |
| International (C-F, G-) | **0** |
| Mixed case / lowercase | **0** |
| Tail + ask price combo | **0** |
| Wrong registry typo | **0** |
| Fleet tail list (2+ tails) | **0** |

Production corpus `valuation` category (100 queries) is **not** the same as tail-investigation suite coverage.

---

## Missing comparison scenarios

| Gap | Notes |
|-----|-------|
| 3+ model compare | UI contract supports; no e2e assert |
| Unregistered model pair | Phenom vs CJ4 |
| Compare + budget constraint | “G650 vs 8X under $50M” |
| Compare + mission | “coast-to-coast G280 vs Longitude” |
| Partial shorthand | “Phenom vs CJ” |
| Cross-OEM category | “best super-mid: Longitude vs Praetor vs Challenger” |
| Insufficient data recovery | Assert recovery prose quality |

Production fixture has **100 comparison queries** but replay does not assert comparison structure quality.

---

## Missing production corpus categories

`production_queries.json` (500):

| Category | Count | Benchmark quality assert |
|----------|-------|---------------------------|
| comparison | 100 | Authority only |
| buy_decision | 100 | Authority only |
| mission | 100 | Authority only |
| alternative | 100 | Authority only |
| valuation | 100 | Authority only |
| **listing** | **0** | — |

**Critical gap:** no production-replay listing category despite listing certification suite.

---

## Recommended benchmark additions (measurement only)

1. Align all KPI suites on `prefer_e2e=False` (or document and test both).
2. Add `assert bias_ok` to `market_recommendation_audit.py`.
3. Expand `listing_validation_suite` to 50+ cases from tier×ratio grid.
4. Add `comparison_quality_suite` asserting registry-resolved pair and non-insufficient body.
5. Wire `listing_realism_suite` to Phase 53 `infer_listing_verdict` + assert correct.
6. Extend tail suite to 20+ registrations and international formats.
7. Add production corpus `listing` category (50–100 queries) with expected verdict class.
