# Phase 52 — Accuracy Gap Analysis

Generated: 2026-06-03

## Summary

Phase 51 benchmarks exposed authority misrouting and recommendation ranking gaps. Phase 52 fixes targeted the decision engine only. All benchmark scenarios now pass.

## Retrieval failures (Phase 51 → root cause → fix)

### cheap_gulfstream
- **Root cause:** `AUTHORITY_ERROR` — dispatch returned `None`; no `alternative` kind.
- **Expected authority:** alternative | **Actual:** (none)
- **Fix:** Category discovery block + `is_alternative_execution_query` patterns for cheap Gulfstream.

### g650_18m
- **Root cause:** `AUTHORITY_ERROR` — mission path without alternative dispatch.
- **Expected:** alternative | **Actual:** (none)
- **Fix:** `g650 for Xm` category discovery regex in authority dispatch.

### best_jet_20m
- **Root cause:** `AUTHORITY_ERROR` + empty aircraft pool.
- **Missing:** budget-ranked super-mid candidates in dispatch answer.
- **Fix:** Acquisition guidance + `match_budget_opportunities` with mission fit.

### g700_under_5m
- **Root cause:** `AUTHORITY_ERROR`.
- **Fix:** `g700 under Xm` category discovery pattern.

### tail_investigation
- **Root cause:** `AUTHORITY_ERROR` — tail mode not bound to valuation dispatch kind.
- **Fix:** `detect_listing_signal` → `build_tail_investigation_brief` with `dispatch_kind=valuation`.

### buy_now_or_wait
- **Root cause:** `AUTHORITY_ERROR` — timing intent fell through to mission pipeline.
- **Fix:** `build_broker_decision` BUY_OR_WAIT with `dispatch_kind=buy_decision`.

### longitude_vs_challenger (was passing)
- No change required.

## Recommendation failures (Phase 51 → root cause → fix)

### coast_to_coast_6pax_20m
- **Root cause:** `RECOMMENDATION_ERROR` — G280 ranked above Longitude (range/$, not mission).
- **Expected primary:** Citation Longitude | **Actual:** Gulfstream G280
- **Missing mission factors:** coast-to-coast 2600nm, super-mid at $20M utilization
- **Fix:** `mission_fit_scorer.py` + ranked executive primary selection.

### g650_18m
- **Root cause:** `RECOMMENDATION_ERROR` — budget matcher favored G280.
- **Expected:** G650 | **Actual:** G280
- **Fix:** Query-focus model rows + named-model soft budget penalty.

### best_jet_15m
- **Root cause:** `RECOMMENDATION_ERROR` — category phrase parsed as aircraft; infeasible rejection; single G280 in pool.
- **Missing:** super-midsize candidate set (Longitude, Challenger 350, Praetor 600)
- **Fix:** `BUDGET_MATCH` intent for super-midsize; category early-return; stretch band 1.5× for class search.

### gulfstream_under_12m / g700_under_5m (were passing or fixed by guards)
- Retained acquisition budget reality guards; no presentation changes.
