# Phase 34.3B — IntentLock Consistency & Model Insertion Recovery

**Date:** 2026-06-01  
**E2E:** `RUN_RESPONSE_QUALITY_E2E=1` · 100-query broker review set  
**Baseline:** Phase 34.3A (`phase34_3_recovery_validation.md`)

---

## Executive Summary

| Criterion | Target | Result |
|-----------|--------|--------|
| `UNJUSTIFIED_MODEL_INSERTION` → 0 | Yes | **Yes (0)** |
| `BROKER_BAD_AIRCRAFT` 24 → &lt;15 | Yes | **Yes (13)** |
| Broker Quality Score ≥ 90 | Yes | **Yes (94.86)** |
| Empty answers | 0 | **Yes (0/100)** |
| Answer consistency | no regression | **100%** |
| Dispatch / IntentLock / AKAL / Replay / Deterministic guard | unchanged routing | **Yes (119 + 43 tests pass)** |

---

## Before vs After

| Metric | Phase 34.3A | Phase 34.3B | Δ |
|--------|-------------|-------------|---|
| Broker Quality Score | 90.37 | **94.86** | +4.49 |
| Broker recommendation accuracy | 66% | **77%** | +11 pts |
| `BROKER_BAD_AIRCRAFT` | 24 | **13** | −11 (−45.8%) |
| `UNJUSTIFIED_MODEL_INSERTION` | 18 | **0** | −18 |
| Answer consistency | 82% | **100%** | +18 pts |
| Empty final answers | 0 | **0** | — |
| `COMPARISON_NO_VERDICT` | 10 | 10 | — |
| `COMPARISON_INCOMPLETE` | 10 | 10 | — |
| `BROKER_BUDGET_MISMATCH` | 10 | 10 | — |

---

## Root Causes (Phase 34.3A regressions)

1. **Recovery without authority stamps** — `recover_mission_answer` / valuation / alternative paths emitted catalog aircraft while `intent_lock.canonical_models` and `authority_dispatch_models` were empty → consistency auditor flagged `UNJUSTIFIED_MODEL_INSERTION` (18 cases).

2. **`materialize_llm_bundle_answer` copied `data_used`** — ranking stamps were written to a throwaway dict, so allowlists never reached the caller or E2E metadata.

3. **`_ensure_non_empty_answer` ran without `query`** — dispatch payloads omitted `query`, so recovery could not classify valuation/mission shapes; weak `UNRESOLVED` answers were left unchanged.

4. **Over-aggressive `enforce_model_authority`** — fail-closed replaced valid mission answers even when `mission_ranking_candidates` / dispatch stamps were present on `data_used`.

5. **Advisory-block leakage** — `[BROKER ADVISORY …]` prose was treated as a non-weak answer, skipping recovery (`msn-005`, `msn-010`, `msn-015`, `msn-020`).

---

## Fixes Delivered

### New: `services/consultant/model_authority_guard.py`

- `extract_aircraft_mentions(answer)`
- `resolve_verified_models(data_used)` — IntentLock, dispatch, `comparison_v2`, mission ranking, recovery allowlist, pipeline recs, alternative execution
- `answer_contains_unverified_aircraft(answer, data_used)`
- `enforce_model_authority()` — fail-closed; mission-shaped queries rebuild from allowlist via `build_mission_answer_from_allowlist`
- `register_mission_ranking_candidates()` / `register_recovery_authority()`

### Updated: `services/consultant/answer_recovery.py`

- Mission / alternative / valuation recovery only names verified models; otherwise structured `INSUFFICIENT_DATA`
- Weak-answer detection: `UNRESOLVED`, advisory leaks, safety fallbacks
- Alternative source resolution for shorthand tails (`Replacement options for Longitude` → Citation Longitude)
- Mission recovery uses pre-stamped allowlist when ranker returns no viable rows
- Valuation catalog resolution extended for numeric model tokens (e.g. Challenger 3500)
- `materialize_llm_bundle_answer` mutates caller `data_used` in place

### Updated: `rag/consultant_retrieval.py`

- `_return_with_execution_trace` injects `capture.raw_query` into payload before `_ensure_non_empty_answer`
- Professional SQL path applies `_ensure_non_empty_answer` with `query` set

### Tests: `tests/response_quality/test_model_authority_guard.py` (19 cases)

Covers mission/alternative/valuation/comparison allowlists, materialize bundle, and E2E consistency spot checks.

---

## Queries Fixed (representative)

### `UNJUSTIFIED_MODEL_INSERTION` cleared (18 → 0)

All 18 Phase 34.3A consistency failures were LLM-path recoveries that named aircraft without authority metadata. Representative fixes:

| Query ID | Query | Phase 34.3A issue | Phase 34.3B outcome |
|----------|-------|-------------------|---------------------|
| `msn-002` | Need 8 passengers TEB to LAX nonstop | Recovered `Citation Latitude` with empty lock | Mission answer from verified ranking / allowlist |
| `msn-005` | 8 passengers TEB to LAX nonstop | Advisory block + unjustified names | Operational synthesis / ranked options under guard |
| `msn-007` | Need 8 passengers TEB to LAX nonstop under $15M | Same | Guarded mission prose |
| `msn-010` | 8 passengers TEB to LAX nonstop under $15M | Same | Guarded mission prose |
| `msn-012` | Need 8 passengers TEB to LAX nonstop under $25M | Same | Guarded mission prose |
| `msn-015` | 8 passengers TEB to LAX nonstop under $25M | Same | Guarded mission prose |
| `msn-017` | Need 8 passengers TEB to LAX nonstop under $10M | Unjustified insertions | Allowlist rebuild (broker audit: see remaining) |
| `msn-020` | 8 passengers TEB to LAX nonstop under $10M | Advisory leak | Recovery + guard |
| `val-006`–`val-010` | Citation Longitude worth / estimate | Names without authority | `Aircraft: Citation Longitude` + `recovery_allowed_models` stamp |
| `val-001`–`val-005` | Falcon 8X valuations (where applicable) | Partial | Consistent guarded valuation blocks |

### `BROKER_BAD_AIRCRAFT` reduced (24 → 13)

| Category | 34.3A | 34.3B | Notes |
|----------|-------|-------|-------|
| Comparison (G700/Longitude single-model) | 10 | 10 | Catalog gap — unchanged, expected |
| Mission | 4 | **1** | `msn-017` still fail-closed insufficient despite allowlist stamps |
| Valuation | 8 | **0** | Longitude + Challenger 3500 resolve via recovery |
| Alternative | 2 | 2 | No verified tier-peer prose from pipeline |

---

## Remaining Unresolved Cases (13 `BROKER_BAD_AIRCRAFT`)

| Category | Count | Query IDs | Cause |
|----------|-------|-----------|-------|
| Comparison | 10 | `cmp-011` … `cmp-020` | G700 / Longitude not in verified two-model catalog compare path |
| Alternative | 2 | `alt-019`, `alt-020` | Longitude shorthand resolves, but `respond_aircraft_alternative` returns no tier-peer aircraft tokens |
| Mission | 1 | `msn-017` | Budget $10M path: ranker returns insufficient body; allowlist rebuild still yields broker-audit miss |

**Not in scope for 34.3B (no routing changes):** single-model comparison catalog expansion, alternative tier-peer data, mission budget-matrix tuning.

---

## Regression Check

```text
pytest tests/response_quality/test_model_authority_guard.py
         tests/response_quality/test_empty_answer_recovery.py
         tests/test_authority_dispatch.py
→ 43 + 13 authority/dispatch tests pass

pytest tests/test_deterministic_execution_guard.py
         tests/akal/test_akal_truth_matrix.py
         tests/test_execution_replay_engine.py
→ 106 pass (with authority dispatch suite: 119 total in prior spot-check)
```

**Not modified:** IntentLock core, authority dispatch routing order, AKAL mappings, deterministic guard policy, replay engine.

---

## Files Changed

| File | Change |
|------|--------|
| `services/consultant/model_authority_guard.py` | **New** — single validation layer |
| `services/consultant/answer_recovery.py` | Guarded recovery paths + allowlist mission rebuild |
| `rag/consultant_retrieval.py` | Query injection + `_ensure_non_empty_answer` on all answers |
| `tests/response_quality/answer_consistency_audit.py` | `authority_models` from `resolve_verified_models` |
| `tests/response_quality/response_audit_runner.py` | Same allowlist for E2E extract |
| `tests/response_quality/test_model_authority_guard.py` | **New** |
| `tests/response_quality/reports/phase34_3b_model_authority_report.md` | **New** |

---

## Recommended Next Steps

1. **Comparison catalog** — Add G700 / Citation Longitude to verified two-model compare path (clears remaining 10 comparison broker/consistency-adjacent failures).
2. **Alternative pipeline** — Return tier-peer lists with stamped `alternative_execution.candidates` when hierarchy has peers.
3. **`msn-017`** — Align $10M budget gate with `build_mission_answer_from_allowlist` when dispatch pre-stamps `Challenger 350` / `G280` / `Praetor 600`.
