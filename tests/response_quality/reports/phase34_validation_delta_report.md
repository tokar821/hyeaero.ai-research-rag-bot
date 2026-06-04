# Phase 34 Validation Delta Report

**Run:** 2026-06-01 · `RUN_RESPONSE_QUALITY_E2E=1` · 100-query broker review set  
**Scope:** Measurement only — no production code changes in this phase.

---

## Executive Summary

```text
PASS / FAIL: FAIL (success criteria not met; stop conditions clear)

Broker Quality Score:
Before: 81.74
After:  78.55  (-3.19 pts, -3.9%)

Failure Counts:
Before → After (Δ, improvement %)
  BROKER_BAD_AIRCRAFT:         54 → 64  (+10, -18.5% regression)
  COMPARISON_NO_VERDICT:       20 → 20  (0, 0%)
  COMPARISON_INCOMPLETE:       10 → 20  (+10, -100% regression)
  BROKER_BUDGET_MISMATCH:      11 → 10  (-1, +9.1% improvement)
  UNJUSTIFIED_MODEL_INSERTION:  4 →  4  (0, 0%)

Stop conditions (hard gate): 0 hits (unchanged)
  HALLUCINATED_AIRCRAFT: 0
  CROSS_MODEL_VALUATION: 0
  MISSION_INFEASIBLE_RECOMMENDATION: 0
  VERDICT_DRIFT: 0

Remaining Risks:
- Comparison queries (20/20) still return authority-dispatch safety fallback text, not comparison v2 prose — remediation not on active E2E path.
- 28 answers remain empty; 32 return deterministic safety fallback without catalog aircraft tokens — drives BROKER_BAD_AIRCRAFT regression (+10).
- COMPARISON_INCOMPLETE doubled because fallback text lacks range/cabin/cost/verdict keywords the Phase 33 auditor expects.
- UNJUSTIFIED_MODEL_INSERTION unchanged (4) — advisory-context prompt leakage still present on buy-path samples.
- Budget gate moved 11→10 mismatches; mission/buy budget alignment still above target (≤2).

Recommended Next Phase:
- Additional remediation required (Phase 34.1) before Phase 35 Monitoring.
  Wire comparison v2 responder into authority-dispatch comparison success path.
  Fix empty LLM-bundle answers for mission/valuation dispatch paths.
  Re-run this suite after integration fixes.
```

---

## Success Criteria vs Actual

| Criterion | Target | After | Met |
| --------- | ------ | ----- | --- |
| BROKER_BAD_AIRCRAFT | < 10 | 64 | No |
| COMPARISON_NO_VERDICT | 0 | 20 | No |
| COMPARISON_INCOMPLETE | 0 | 20 | No |
| UNJUSTIFIED_MODEL_INSERTION | 0 | 4 | No |
| BROKER_BUDGET_MISMATCH | ≤ 2 | 10 | No |
| Broker Quality Score | ≥ 90 | 78.55 | No |

---

## Metric Delta Table

| Metric | Before | After | Δ | Improvement % |
| ------ | ------ | ----- | - | ------------- |
| Broker Quality Score | 81.74 | 78.55 | -3.19 | -3.9% |
| Broker recommendation accuracy | 35% | 26% | -9 pts | -25.7% |
| Comparison quality | 0% | 0% | 0 | — |
| Mission feasibility | 100% | 100% | 0 | — |
| Valuation accuracy | 100% | 100% | 0 | — |
| Answer consistency | 96% | 96% | 0 | — |
| BROKER_BAD_AIRCRAFT | 54 | 64 | +10 | -18.5% |
| COMPARISON_NO_VERDICT | 20 | 20 | 0 | 0% |
| COMPARISON_INCOMPLETE | 10 | 20 | +10 | -100% |
| BROKER_BUDGET_MISMATCH | 11 | 10 | -1 | +9.1% |
| UNJUSTIFIED_MODEL_INSERTION | 4 | 4 | 0 | 0% |

*Improvement % = (before − after) / before × 100. Negative % on failure counts means regression.*

---

## Answer-Shape Analysis (Post-Remediation Run)

| Answer shape | Count (of 100) |
| ------------ | -------------- |
| Deterministic safety fallback (`Insufficient verified data for deterministic execution`) | 32 |
| Empty final answer | 28 |
| Tier-peer alternatives (passing) | 12 |
| Advisory-context leakage (`[BROKER ADVISORY CONTEXT`) | 4 |
| Other (buy/valuation prose, etc.) | 24 |

**Comparison path:** 0 answers contain `Verified catalog comparison` or `Choose …` verdict prose from `comparison_pipeline_v2_responder.py`. All 20 comparison queries use the safety-fallback template.

---

## BROKER_BAD_AIRCRAFT by Category

| Category | Failures (after) |
| -------- | ---------------- |
| comparison | 20 |
| valuation | 20 |
| mission | 16 |
| alternative | 8 |

---

## Top Remaining Root Causes

1. **Dispatch comparison path bypasses v2 responder** — `authority_dispatch` returns `_SAFETY_FALLBACK_ANSWERS["comparison"]` even when IntentLock lists two canonical models; E2E never reaches `respond_aircraft_comparison()` output with VERDICT/cabin/cost lines.

2. **Empty answers (28)** — LLM-bundle returns without populated `answer` on mission/valuation turns; `_ensure_non_empty_answer` may not apply when E2E stub omits full query_service stream.

3. **Audit false-negative on improved comparison code** — Unit test `test_comparison_responder_structured_contrast` passes locally, but E2E answers do not include comparison renderer output.

4. **Prompt leakage (4)** — `UNJUSTIFIED_MODEL_INSERTION` unchanged; `sanitize_llm_output` not exercised when raw advisory block is returned as final answer.

5. **Budget mismatch (10)** — Slight improvement (11→10); `apply_budget_gate` filters ranker candidates but buy/mission answers still mention aircraft above stated budget in prose.

---

## Routing / Guard Regression Spot-Check

| Suite | Result |
| ----- | ------ |
| `test_deterministic_execution_guard` | Pass |
| `test_comparison_alternative_execution` | Pass |
| `test_budget_constraint_matrix` (3 cases) | **Fail** — `authority_dispatch_safety_fallback` set to `'comparison'` where tests expect `None` on successful budget-filtered compare |

No IntentLock, AKAL, or replay-specific failures observed in this spot-check. Budget-matrix failures correlate with comparison safety-fallback behavior, not routing reorder.

---

## Artifacts

- `response_quality_scorecard.json` (updated)
- `broker_review_report.md` (updated)
- `response_quality_results.json` (full per-query findings)
- This file: `phase34_validation_delta_report.md`
