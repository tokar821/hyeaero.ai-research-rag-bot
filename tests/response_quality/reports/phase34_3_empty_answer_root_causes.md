# Phase 34.3A — Empty Answer Root Cause Audit

**Source:** `response_quality_results.json` (pre-recovery baseline, Phase 34.2)  
**Total `BROKER_BAD_AIRCRAFT`:** 54

## Classification (baseline)

| Code | Description | Count |
|------|-------------|-------|
| **A** | Empty answer (`answer_preview == ""`) | 28 |
| **B** | Safety fallback (`Insufficient verified data for deterministic execution`) | 22 |
| **C** | Mission answer with no aircraft tokens | 4 |
| **D** | Valuation without `Aircraft:` structure (subset of B) | 12 |
| **E** | Alternative with no aircraft (subset of A) | 8 |

### By query category (baseline)

| Category | Failures |
|----------|----------|
| valuation | 20 |
| mission | 16 |
| comparison | 10 |
| alternative | 8 |

### Root mechanisms

1. **Empty (A)** — `run_consultant_retrieval_bundle` returned `kind=llm` without an `answer` key; E2E audit read `""`.
2. **Safety fallback (B)** — Valuation dispatch `respond_buy_decision` failed parse → generic fallback without aircraft name; comparison single-model pairs (G700/Longitude).
3. **Mission (C)** — QRI `payload_range_analysis` skipped pre-LLM pipeline; no materialized client answer.
4. **Alternative (E)** — `"Replacement options for G650"` did not match `is_alternative_execution_query`; no dispatch, empty LLM bundle.

## Post–Phase 34.3A (recovery run)

| Metric | Baseline (34.2) | After 34.3A |
|--------|-----------------|-------------|
| Empty answers (all queries) | 28 | **0** |
| `BROKER_BAD_AIRCRAFT` | 54 | **24** |
| Broker Quality Score | 82.44 | **90.37** |

Remaining `BROKER_BAD_AIRCRAFT` are predominantly comparison fail-closed (10) and residual safety-fallback pairs without two catalog models — not empty-output failures.
