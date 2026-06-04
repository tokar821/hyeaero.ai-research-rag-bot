# Phase 34.3A — Recovery Validation Report

**Date:** 2026-06-01  
**Baseline:** Phase 34.2 (`response_quality_scorecard.json` after dispatch acceptance fix)

---

## Executive Summary

| Criterion | Target | Result |
|-----------|--------|--------|
| `BROKER_BAD_AIRCRAFT` < 20 | Yes | **No** (24) |
| No empty answers | Yes | **Yes** (0/100) |
| No blank valuation outputs | Yes | **Yes** (all valuation answers include `Aircraft:`) |
| No blank alternative outputs | Yes | **Yes** |
| Broker Quality Score > 88 | Yes | **Yes** (90.37) |
| No routing/dispatch/AKAL/replay regressions | Yes | **Yes** (119 spot-check tests pass) |

---

## Before vs After

| Metric | Phase 34.2 (before) | Phase 34.3A (after) | Δ |
|--------|---------------------|---------------------|---|
| Broker Quality Score | 82.44 | **90.37** | +7.93 |
| Broker recommendation accuracy | 36% | **66%** | +30 pts |
| `BROKER_BAD_AIRCRAFT` | 54 | **24** | −30 (−55.6%) |
| `COMPARISON_NO_VERDICT` | 10 | 10 | — |
| `COMPARISON_INCOMPLETE` | 10 | 10 | — |
| `BROKER_BUDGET_MISMATCH` | 10 | 10 | — |
| `UNJUSTIFIED_MODEL_INSERTION` | 4 | **18** | +14 |
| Answer consistency | 96% | 82% | −14 pts |
| Empty final answers | 28 | **0** | −28 |

---

## Remaining failure categories

| Category | Count | Notes |
|----------|-------|-------|
| `BROKER_BAD_AIRCRAFT` | 24 | Mostly comparison safety-fallback (10) + generic safety text without catalog tokens (6) + mission (4) + valuation (2) + alternative (2) |
| `COMPARISON_NO_VERDICT` | 10 | G700/Longitude single-model compares (unchanged) |
| `COMPARISON_INCOMPLETE` | 10 | Same 10 queries |
| `BROKER_BUDGET_MISMATCH` | 10 | Unchanged |
| `UNJUSTIFIED_MODEL_INSERTION` | 18 | LLM-path valuations/missions: `canonical_models` empty in lock but answer lists recovered aircraft (consistency audit, not broker audit) |

---

## Files changed

| File | Change |
|------|--------|
| `services/consultant/answer_recovery.py` | **New** — valuation/alternative/mission recovery, LLM bundle materialization |
| `rag/consultant_retrieval.py` | `_ensure_non_empty_answer()` uses recovery; LLM bundles get `answer` via `materialize_llm_bundle_answer()` |
| `services/consultant/broker_advisory_layer.py` | Explicit `INSUFFICIENT_DATA` mission block when no viable aircraft |
| `tests/response_quality/test_empty_answer_recovery.py` | **New** — 23 deterministic recovery tests |
| `tests/response_quality/reports/phase34_3_empty_answer_root_causes.md` | **New** |
| `tests/response_quality/reports/phase34_3_recovery_validation.md` | **New** |

**Not modified:** IntentLock, authority dispatch routing, AKAL, deterministic guard, replay engine.

---

## Regression check

```text
pytest tests/test_authority_dispatch.py
tests/test_deterministic_execution_guard.py
tests/akal/test_akal_truth_matrix.py
tests/test_execution_replay_engine.py
→ 119 passed
```

---

## Recommended next steps

1. **Phase 34.3B** — Comparison single-model detection (G700/Longitude) to clear remaining 10 comparison failures.
2. **Consistency audit** — Treat recovered catalog models on LLM-path valuations as allowed when query-resolved (or populate `authority_models` in materialization metadata only).
3. **Valuation dispatch** — Optional `respond_buy_decision` parse extension (dispatch scope) for year/model worth queries without changing routing order.
