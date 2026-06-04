# Phase 34 ? Root Cause Analysis (Phase 33 Answer Quality Failures)

Scope: **analysis only**. No production changes in this phase.
Data source: `tests/response_quality/reports/response_quality_results.json` (100-query broker review set).

## Failure counts (observed)

- **BROKER_BAD_AIRCRAFT**: 54
- **COMPARISON_NO_VERDICT**: 20
- **BROKER_BUDGET_MISMATCH**: 11
- **COMPARISON_INCOMPLETE**: 10
- **UNJUSTIFIED_MODEL_INSERTION**: 4

## Root causes by failure category

### BROKER_BAD_AIRCRAFT

- **Count**: 54
- **Generation path**: `rag.consultant_retrieval.run_consultant_retrieval_bundle()` ? (a) deterministic dispatch path for hard intents OR (b) advisory/LLM path for mission/advisory turns.
- **Primary observed mechanisms**:
  - **Empty final answer**: 28 cases (all flagged). Final payload `answer` was empty, so the audit could not extract any aircraft.
  - **Deterministic safety-fallback has no aircraft tokens**: ?Insufficient verified data ?? responses fail the audit?s ?recommended aircraft present? rule.
- **Source modules**:
  - `rag/consultant_retrieval.py` (orchestration, LLM invocation, fail-closed handling).
  - `services/routing/authority_dispatch.py` + `rag/consultant_retrieval._build_hard_deterministic_safety_fallback()` (deterministic safety responses).
  - `services/consultant/llm_explanation_layer.py` + `services/consultant/broker_advisory_layer.py` (advisory prompt context for mission narration).
- **Source data used**: `data_used.intent_lock`, `data_used.authority_dispatch_models`, and broker advisory context blocks (mission summary + feasible aircraft).
- **Why the audit failed**: final answer text contained **no recognizable aircraft** (empty output or generic fail-closed message).
- **Remediation proposals**:
  - Add a **non-empty fallback** if LLM output is empty/invalid (never return empty answer).
  - Make deterministic safety fallback include a structured `INSUFFICIENT_DATA` and echo recognized aircraft (if any) plus next-step questions.
  - Ensure broker advisory prompt blocks never leak as final answers.

### COMPARISON_NO_VERDICT

- **Count**: 20 (out of 20 comparisons).
- **Generation path**: hard deterministic comparison ? authority dispatch ? deterministic comparison renderer.
- **Source module**: `services/comparison/comparison_pipeline_v2_responder.py`
- **Function(s)**: `respond_aircraft_comparison()` / `_format_structured_contrast()`
- **Source data used**: comparison v2 JSON fields (category, range_nm, seats, cost_band).
- **Failure mechanism**: renderer never emits a **verdict** (?Choose X if?, otherwise Y? / `INSUFFICIENT_DATA`).
- **Remediation proposal**: append deterministic verdict clause or explicit `INSUFFICIENT_DATA`.

### COMPARISON_INCOMPLETE

- **Count**: 10
- **Source module/function**: `services/comparison/comparison_pipeline_v2_responder.py::_format_structured_contrast()`
- **Failure mechanism**: missing explicit cabin delta / operating cost delta sentences; cost is present as a band but not compared.
- **Remediation proposal**: add explicit cabin+cost delta lines when data present; otherwise `INSUFFICIENT_DATA`.

### BROKER_BUDGET_MISMATCH

- **Count**: 11
- **Generation path**: mission/advisory outputs (non-dispatch) mentioning aircraft above parsed budget.
- **Likely sources**: `services/consultant/recommendation_engine.py` (candidate selection) and narration context in `services/consultant/broker_advisory_layer.py`.
- **Failure mechanism**: budget not enforced in shortlist or not respected in narrated ?feasible aircraft?.
- **Remediation proposal**: budget gate before selecting max-3 aircraft; never tag over-budget aircraft as primary.

### UNJUSTIFIED_MODEL_INSERTION

- **Count**: 4
- **Generation path**: mission/advisory LLM narration path.
- **Source modules**: `services/consultant/broker_advisory_layer.py` and `services/consultant/llm_explanation_layer.py` (prompt construction), then `rag/consultant_retrieval.py` (LLM invocation).
- **Failure mechanism (observed)**: internal `[BROKER ADVISORY CONTEXT ? ?]` prompt block is returned verbatim as the final answer, inserting aircraft not present in lock/dispatch metadata.
- **Remediation proposal**: prevent prompt leakage; ensure intent_lock metadata is attached for mission turns.
