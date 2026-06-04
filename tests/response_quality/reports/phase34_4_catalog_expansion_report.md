# Phase 34.4 — Comparison Catalog Expansion + Remaining BAD_AIRCRAFT Cleanup

**Date:** 2026-06-01  
**E2E:** `RUN_RESPONSE_QUALITY_E2E=1` · 100-query broker review set  
**Baseline:** Phase 34.3B (`phase34_3b_model_authority_report.md`)

---

## Executive Summary

| Criterion | Target | Result |
|-----------|--------|--------|
| `BROKER_BAD_AIRCRAFT` ≤ 3 | Yes | **0** |
| `COMPARISON_NO_VERDICT` = 0 | Yes | **0** |
| `COMPARISON_INCOMPLETE` = 0 | Yes | **0** |
| `UNJUSTIFIED_MODEL_INSERTION` = 0 | Yes | **0** |
| Broker Quality Score ≥ 96 | Yes | **100.0** |
| Routing / IntentLock / AKAL / Replay / Deterministic guard | unchanged | **Yes** |

---

## Before vs After (E2E)

| Metric | Phase 34.3B | Phase 34.4 | Δ |
|--------|-------------|------------|---|
| Broker Quality Score | 94.86 | **100.0** | +5.14 |
| Broker recommendation accuracy | 77% | **89%** | +12 pts |
| `BROKER_BAD_AIRCRAFT` | 13 | **0** | −13 |
| `COMPARISON_NO_VERDICT` | 10 | **0** | −10 |
| `COMPARISON_INCOMPLETE` | 10 | **0** | −10 |
| `UNJUSTIFIED_MODEL_INSERTION` | 0 | **0** | — |
| Answer consistency | 100% | **100%** | — |
| `BROKER_BUDGET_MISMATCH` | 10 | 10 | — (buy-path budget matrix; unchanged) |

---

## Root Causes (Phase 34.3B failures)

### Comparison (cmp-011 … cmp-020)

1. **G700** — `detect_models_from_text` emitted `G700`, but `Gulfstream G700` was absent from `AIRCRAFT_PROFILES` / `CANONICAL_COMPARISON_REGISTRY`, so `lock_comparison_aircraft` rejected the token → single-model compare → dispatch safety fallback.
2. **Longitude** — Bare `Longitude` in `G650 vs Longitude` was not detected by model regex; even when resolved via AKAL, `Citation Longitude` was missing from the comparison catalog registry.

### Alternatives (alt-019, alt-020)

- `_resolve_alternative_target` did not parse `Replacement options for Longitude` / `Similar aircraft to Longitude`.
- Recovery rewrote queries to `alternatives to Longitude` (unresolved) instead of canonical `Citation Longitude`.
- `respond_aircraft_alternative` returned generic prose without catalog aircraft tokens when target was unresolved.

### Mission (msn-017)

- Pipeline stamped `deterministic_recommendation_pipeline` recommendations, but `enforce_elimination_invariant` eliminated all ranked models (including allowlisted ones).
- `format_broker_advisory_response` fell through to operational-synthesis-only text without `Aircraft Options`.
- Recovery weak-answer detection did not re-run mission recovery on `(none)` / insufficient mission bodies.

---

## Files / Functions Modified

| File | Change |
|------|--------|
| `services/mission/aircraft_profiles.py` | Added `Gulfstream G700`, `Citation Longitude` profiles |
| `services/comparison/aircraft_registry_lock.py` | G700/Longitude spoken aliases; `resolve_aircraft_alias` before registry lock |
| `services/aircraft/aircraft_authority_service.py` | AKAL aliases `g700` → `Gulfstream G700`; G700 competitors |
| `services/routing/authority_dispatch.py` | `g700` dispatch alias; `_comparison_models` applies `resolve_aircraft_alias` |
| `rag/consultant_query_expand.py` | `G700` → `Gulfstream G700`; bare `longitude` → `Citation Longitude` |
| `services/recommendation/aircraft_positioning.py` | G700 tier = FLAGSHIP_ULR |
| `services/comparison/alternative_pipeline_responder.py` | Replacement/similar phrasing; canonical target resolution |
| `services/recommendation/replacement_hierarchy.py` | Replacement tail parsing + alias resolution |
| `services/consultant/answer_recovery.py` | Canonical alternative query; direct allowlist mission formatter; weak-answer expansion |
| `tests/test_comparison_catalog_aliases.py` | **New** — alias + compare_models length tests |
| `tests/test_authority_dispatch.py` | G700/Longitude comparison success expectations |
| `tests/response_quality/test_model_authority_guard.py` | Longitude alternative recovery expects peers |

**Not modified:** IntentLock routing order, authority dispatch ordering, AKAL cross-model profile remap policy, replay engine, deterministic execution guard.

---

## Queries Fixed

| IDs | Query pattern | Fix |
|-----|---------------|-----|
| cmp-011 … cmp-015 | G650 vs G700 | Catalog + alias → `comparison_v2.status == OK`, VERDICT |
| cmp-016 … cmp-020 | G650 vs Longitude | Longitude detection + Citation Longitude catalog |
| alt-019, alt-020 | Replacement / similar … Longitude | Target → Citation Longitude; tier-peer prose |
| msn-017 | 8 pax TEB–LAX under $10M | Allowlist mission formatter bypasses elimination wipe |

---

## Regression Results

```text
pytest tests/test_comparison_catalog_aliases.py tests/test_authority_dispatch.py
→ 22 passed

pytest tests/response_quality/test_model_authority_guard.py
         tests/response_quality/test_empty_answer_recovery.py
         tests/test_deterministic_execution_guard.py
         tests/akal/test_akal_truth_matrix.py
         tests/test_execution_replay_engine.py
→ 150 passed (full spot-check)

RUN_RESPONSE_QUALITY_E2E=1
pytest tests/response_quality/test_response_quality.py::test_broker_review_set_response_quality_report
→ 1 passed (100 queries)
```

---

## Remaining Unresolved Risks

| Risk | Notes |
|------|-------|
| `BROKER_BUDGET_MISMATCH` (10) | Buy-decision price matrix queries; intentional audit signal, not aircraft resolution |
| Elimination vs recovery | Mission recovery allowlist formatter bypasses elimination when pipeline marks all candidates eliminated; scoped to recovery only |
| Catalog drift | New shorthand tokens still require explicit alias + `AIRCRAFT_PROFILES` entry (no fuzzy matching) |
| Challenger Longitude | Not added to comparison set; bare `longitude` maps to Citation Longitude by design |

---

## Recommended Next Steps

1. Add **Challenger Longitude** to catalog if compare queries use that token explicitly.
2. Reconcile **elimination invariant** with pipeline recommendations so primary path does not emit `(none)` when ranked models exist.
3. Extend buy-path handling to reduce `BROKER_BUDGET_MISMATCH` noise without weakening budget audit.
