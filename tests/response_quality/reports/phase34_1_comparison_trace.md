# Phase 34.1 — Comparison E2E Trace Investigation

**Date:** 2026-06-01  
**Scope:** Investigation only — no production code changes.  
**Method:** Live trace via `run_consultant_retrieval_bundle()` (same path as Phase 33 E2E) + isolated `consult_authority_dispatch()` / `respond_aircraft_comparison()` probes.

---

## Executive Summary

| Item | Finding |
| ---- | ------- |
| **Root cause** | `consult_authority_dispatch()` **rejects valid comparison prose** because `respond_aircraft_comparison()` embeds the substring `"Insufficient verified"` in the `VERDICT` block (from Phase 34 `_format_verdict_section` when `verdict.no_fit_reason` is set). The dispatch gate treats that as failure and replaces the entire answer with `_build_dispatch_safety_fallback()`. |
| **First failing component** | `services.routing.authority_dispatch.consult_authority_dispatch()` — line ~490 acceptance check |
| **Comparison responder reached?** | **YES** (for 2-model pairs); **NO** (for G700 / Longitude — fewer than two locked models) |

**Bug class:** **Fallback Logic** (primary) + **Comparison Responder Wiring** (verdict text poisons dispatch acceptance) + **Model resolution** (secondary, G700/Longitude only).

**Not the primary issue:** IntentLock (works for G650/Falcon and G650/Global), AKAL routing order, or consultant_retrieval bypassing dispatch.

---

## Answers to Investigation Questions

### Q1 — Does authority dispatch recognize BOTH aircraft?

| Query | Expected | IntentLock `canonical_models` | Dispatch uses |
| ----- | -------- | ------------------------------ | ------------- |
| G650 vs Falcon 8X | G650, Falcon 8X | `Gulfstream G650`, `Falcon 8X` | Both recognized |
| G650 vs Global 7500 | G650, Global 7500 | `Gulfstream G650`, `Global 7500` | Both recognized |
| G650 vs G700 | G650, G700 | **`Gulfstream G650` only** | G700 rejected by registry lock (`G700` → rejected) |
| G650 vs Longitude | G650, Longitude | **`Gulfstream G650` only** | `detect_models_from_text` never extracts Longitude |

### Q2 — Valid comparison payload?

For **G650 vs Falcon 8X** (representative 2-model success path):

```json
{
  "comparison_v2": {
    "status": "OK",
    "models": ["Gulfstream G650", "Falcon 8X"]
  },
  "comparison_structured_engine": {
    "type": "comparison_v2_json",
    "models": ["Gulfstream G650", "Falcon 8X"]
  },
  "verdict": {
    "best_overall": null,
    "conditional_winner": null,
    "no_fit_reason": "no aircraft met minimum mission fit threshold"
  },
  "aircraft": [
    { "name": "Gulfstream G650", "mission_fit_score": 0.55, "category": "large-cabin", "cost_band": "medium" },
    { "name": "Falcon 8X", "mission_fit_score": 0.55, "category": "large-cabin", "cost_band": "medium" }
  ]
}
```

Payload is built successfully (`comparison_v2.status == OK`). Verdict object is **empty of winners** because `_build_verdict()` requires `mission_fit_score >= 0.6` (`comparison_pipeline_v2.py`).

### Q3 — Is `respond_aircraft_comparison()` called?

| Query | Called? | Evidence |
| ----- | ------- | -------- |
| G650 vs Falcon 8X | **YES** | `data_used.comparison_v2.status == "OK"`; `deterministic_execution.final_responder == "respond_aircraft_comparison"` |
| G650 vs Global 7500 | **YES** | Same |
| G650 vs G700 | **NO** | `len(compare_models) < 2` after lock; no `comparison_v2` metadata |
| G650 vs Longitude | **NO** | `len(compare_models) < 2`; no `comparison_v2` metadata |

**If NO (G700 / Longitude):** execution never enters the `respond_aircraft_comparison` block; falls through directly to `_build_dispatch_safety_fallback()` at `authority_dispatch.py:499`.

### Q4 — Why does `authority_dispatch_safety_fallback` trigger?

| Query | `fallback_reason` | `fallback_trigger_function` | `fallback_conditions` |
| ----- | ----------------- | --------------------------- | --------------------- |
| G650 vs Falcon 8X | `hard_intent_insufficient_resolution` | `authority_dispatch._build_dispatch_safety_fallback()` | `respond_aircraft_comparison()` returned text containing **`"Insufficient verified"`** (in VERDICT tail), failing gate at line 490 |
| G650 vs Global 7500 | Same | Same | Same substring gate |
| G650 vs G700 | Same | Same | **`len(compare_models) < 2`** (only `Gulfstream G650` locked) |
| G650 vs Longitude | Same | Same | **`len(compare_models) < 2`** (Longitude not in AKAL canonical set) |

**Critical gate (2-model cases):**

```python
# services/routing/authority_dispatch.py ~484-499
answer = respond_aircraft_comparison(...)
if answer and "Insufficient verified" not in answer:
    return AuthorityDispatchResult(...)  # SUCCESS
return _build_dispatch_safety_fallback("comparison", data_used)  # FAIL
```

Responder output **ends with**:

```text
VERDICT:
INSUFFICIENT_DATA: Insufficient verified aircraft data to produce a comparison.
```

That substring matches the gate → full answer discarded → safety fallback returned.

### Q5 — Execution divergence

**Expected path (2-model compare):**

```text
run_consultant_retrieval_bundle()
  → build_intent_lock()                    [OK: 2 models]
  → consult_authority_dispatch()           [comparison intent]
  → respond_aircraft_comparison()          [OK: catalog comparison prose]
  → return AuthorityDispatchResult         [FAIL: substring gate]
  → _build_dispatch_safety_fallback()      [actual final answer]
  → resolve_deterministic_bypass_response()
  → _return_with_execution_trace(path=authority_dispatch)
```

**Actual path (G650 vs Falcon 8X):**

```text
A  run_consultant_retrieval_bundle          rag/consultant_retrieval.py
B  build_intent_lock                       services/core/semantic_intent_lock_engine.py
C  consult_authority_dispatch              services/routing/authority_dispatch.py
D  respond_aircraft_comparison             services/comparison/comparison_pipeline_v2_responder.py
X  _build_dispatch_safety_fallback         services/routing/authority_dispatch.py:499
   (skips successful AuthorityDispatchResult return at :493-498)
E  Final answer = _SAFETY_FALLBACK_ANSWERS["comparison"]
```

**Actual path (G650 vs G700 / Longitude):**

```text
A → B → C → X  (never reaches D — compare_models length < 2)
```

---

## Per-Query Trace

### 1. G650 vs Falcon 8X

| Field | Value |
| ----- | ----- |
| **Query** | `G650 vs Falcon 8X` |
| **IntentLock** | `intent_type: comparison`; `canonical_models: [Gulfstream G650, Falcon 8X]`; `dispatch_authority_id: 198595b474a7d2fd1ef22459` |
| **Dispatch models** | `authority_dispatch_models: null` (success path never set — fallback taken) |
| **Dispatch payload** | `comparison_v2: {status: OK, models: [...]}` present in `data_used` but **not** surfaced in final answer |
| **Fallback trigger** | Substring `"Insufficient verified"` in responder output (VERDICT `INSUFFICIENT_DATA` line) |
| **Final answer** | `Insufficient verified data for deterministic execution.\n\nVerified catalog comparison requires two recognized aircraft models.` |
| **execution_trace_v2** | `final_execution_path: authority_dispatch`; `authority_dispatch_result: comparison`; `deterministic_guard_result: bypass`; `llm_invoked: false` |
| **deterministic_execution** | `final_responder: respond_aircraft_comparison`; `trigger_reason: comparison_dispatch`; **`authority_dispatch_safety_fallback: comparison`** |

**Responder excerpt (discarded):**

```text
Verified catalog comparison:
- Gulfstream G650: large-cabin class; practical range 5720 nm; ...
...
VERDICT:
INSUFFICIENT_DATA: Insufficient verified aircraft data to produce a comparison.
```

---

### 2. G650 vs Global 7500

| Field | Value |
| ----- | ----- |
| **IntentLock** | `[Gulfstream G650, Global 7500]` |
| **Fallback trigger** | Same substring gate after successful `comparison_v2` |
| **Final answer** | Same safety fallback template |
| **respond_aircraft_comparison** | **YES** — 569-char answer with `VERDICT:` + poison substring |

---

### 3. G650 vs G700

| Field | Value |
| ----- | ----- |
| **IntentLock** | `[Gulfstream G650]` only |
| **Model resolution** | `detect_models_from_text` → `['G700','G650']`; `lock_comparison_aircraft` → canonical `('Gulfstream G650',)`, **rejected `('G700',)`** |
| **respond_aircraft_comparison** | **NO** — `len(compare_models) < 2` |
| **Fallback trigger** | `missing_second_aircraft` / insufficient canonical pair |
| **Final answer** | Safety fallback (never attempted catalog render) |

---

### 4. G650 vs Longitude

| Field | Value |
| ----- | ----- |
| **IntentLock** | `[Gulfstream G650]` only |
| **Model resolution** | `detect_models_from_text('G650 vs Longitude')` → **`['G650']` only** (Longitude alias not detected) |
| **respond_aircraft_comparison** | **NO** |
| **Fallback trigger** | `missing_second_aircraft` |
| **Final answer** | Safety fallback |

---

## Trace Snippets

### E2E `data_used` (G650 vs Falcon 8X)

```json
{
  "authority_dispatch_kind": "comparison",
  "authority_dispatch_models": null,
  "authority_dispatch_safety_fallback": "comparison",
  "comparison_v2": { "status": "OK", "models": ["Gulfstream G650", "Falcon 8X"] },
  "deterministic_execution": {
    "bypassed_llm": true,
    "trigger_reason": "comparison_dispatch",
    "final_responder": "respond_aircraft_comparison",
    "deterministic_intent": "comparison"
  },
  "intent_lock": {
    "intent_type": "comparison",
    "canonical_models": ["Gulfstream G650", "Falcon 8X"],
    "deterministic_flags": { "execution_path": "comparison", "dispatch_kind": "comparison" }
  }
}
```

### `intent_execution_trace` (abbreviated)

```json
{
  "raw_query": "G650 vs Falcon 8X",
  "qri_intent": "aircraft_comparison",
  "unified_intent": "comparison",
  "resolved_plan": {
    "primary_mode": "comparison",
    "filtered_entities": ["Gulfstream G650", "Falcon 8X"],
    "execution_strategy": "deterministic_only"
  },
  "authority_dispatch_result": "comparison",
  "final_execution_path": "authority_dispatch",
  "llm_invoked": false
}
```

### Isolated dispatch probe

```text
consult_authority_dispatch('G650 vs Falcon 8X')
  → progress_step: path_authority_dispatch_comparison_safety_fallback
  → answer: Insufficient verified data for deterministic execution...
  → comparison_v2 in data_used: OK (responder ran)
```

### Unit test regression signal

`tests/test_authority_dispatch.py::test_comparison_dispatch_g650_vs_falcon_8x` **currently fails** with the same safety-fallback answer — confirms dispatch gate regression independent of E2E stub.

---

## Root Cause Ranking

| Severity | Component | Finding |
| -------- | --------- | ------- |
| **P0** | **Fallback Logic** | `authority_dispatch.py:490` uses substring `"Insufficient verified" not in answer`, which false-negatives valid comparison output containing `INSUFFICIENT_DATA: Insufficient verified...` in VERDICT. |
| **P0** | **Comparison Responder Wiring** | `_format_verdict_section()` appends `INSUFFICIENT_DATA` when `verdict.no_fit_reason` is set, even when catalog comparison body is complete (`comparison_pipeline_v2_responder.py:116-117`). |
| **P1** | **Comparison engine** | `_build_verdict()` nulls winners when `mission_fit_score < 0.6` (scores 0.55 for G650/Falcon), forcing `no_fit_reason` (`comparison_pipeline_v2.py:93-98`). |
| **P2** | **Model resolution (AKAL / detect)** | G700 rejected by registry lock; Longitude not extracted by `detect_models_from_text` → IntentLock single-model → never calls responder. |
| **—** | IntentLock | **Not broken** for standard pairs (G650/Falcon, G650/Global). |
| **—** | Authority Dispatch routing | **Correctly routes** to comparison branch. |
| **—** | consultant_retrieval | **Returns dispatch result**; does not overwrite with a second fallback after dispatch. |

---

## Recommended Fix Locations (for Phase 34.2 — not implemented here)

1. **`services/routing/authority_dispatch.py`** (~490)  
   Replace substring check with structured success signal, e.g. `comparison_v2.status == "OK"`, explicit responder status flag, or match only the **safety-fallback template** (exact prefix), not any `"Insufficient verified"` anywhere in prose.

2. **`services/comparison/comparison_pipeline_v2_responder.py`** (`_format_verdict_section`)  
   When comparison table is complete, emit deterministic `Choose X if … otherwise Y` even when `no_fit_reason` is set; do not append `INSUFFICIENT_DATA` lines that trip dispatch.

3. **`services/comparison/comparison_pipeline_v2.py`** (`_build_verdict`)  
   For explicit catalog comparison mode, lower or bypass mission-fit threshold so `best_overall` is populated for catalog-only compares.

4. **`services/consultant/recommendation_engine.detect_models_from_text` / registry lock**  
   Resolve `Longitude` → `Citation Longitude`; handle `G700` catalog alias or explicit insufficient-data message before dispatch.

---

## Classification Summary

```text
Primary bug bucket:   Fallback Logic
Secondary buckets:    Comparison Responder Wiring
                      Comparison engine (verdict threshold)
Tertiary (2 queries): Model resolution / registry lock (G700, Longitude)

NOT root cause:
  - IntentLock (for valid 2-model pairs)
  - Authority Dispatch routing selection
  - consultant_retrieval dispatch bypass
  - AKAL reordering / replay
```

---

## Investigation Commands (repro)

```powershell
cd backend
python -c "from tests.conftest import run_retrieval; from tests.response_quality.response_audit_service import ResponseAuditService; print(run_retrieval('G650 vs Falcon 8X', svc=ResponseAuditService())[1]['answer'][:200])"

python -c "from services.comparison.comparison_pipeline_v2_responder import respond_aircraft_comparison; a=respond_aircraft_comparison('G650 vs Falcon 8X', compare_models=['Gulfstream G650','Falcon 8X']); print('Insufficient verified' in a, a[-120:])"

python -m pytest tests/test_authority_dispatch.py::test_comparison_dispatch_g650_vs_falcon_8x -v
```
