# Phase 54 — Regression Risk Audit

Generated: 2026-06-03

Measurement-only audit. No production logic was changed.

## Executive summary

Certification at 100% on three Phase 53 suites masks **path divergence** (`prefer_e2e=True` vs `False`), **recorder vs pytest assertion gaps** on older suites, and **centralized tier/band dictionaries** that silently affect listing, acquisition, and executive ranking when edited.

---

## High-severity risks

| File | Function / area | Impact | Severity | Recommended protection |
|------|-----------------|--------|----------|------------------------|
| `tests/e2e/broker_certification_helpers.py` | `broker_certify(prefer_e2e=True)` default | E2E retrieval can bypass layers fixes; production replay and market audit use e2e; real/listing benchmarks use layers → **certification split-brain** | **HIGH** | CI matrix: run critical suites both ways or standardize on `prefer_e2e=False` for KPI suites |
| `services/broker_decision/decision_intent_detector.py` | `detect_decision_intent` | Regex order changes reroute mission buy ↔ BUY_OR_WAIT ↔ REALISTICITY_CHECK; broke coast-to-coast in Phase 53 | **HIGH** | Golden tests per intent class; forbid reordering without `test_failure_injection` |
| `services/executive_broker/acquisition_budget_reality.py` | `_should_reject_infeasible_acquisition`, `_is_listing_assessment_query` | Listing vs acquisition boundary controls infeasible answers and market block | **HIGH** | Pairwise tests for listing ask / budget ask / “only have” probes |
| `services/executive_broker/recommendation_selector.py` | `select_executive_recommendation` | Primary selection, query-focus override, coast alt injection | **HIGH** | Lock Phase 53 txn/mission scenarios; monitor `executive_recommendation` dict not trace-only |
| `services/executive_broker/budget_reality_guard.py` | `apply_budget_reality_guard` | Demotes pinned listing models above budget (Global 7500 @ $40M) | **HIGH** | Regression test: named model + ask must keep focus model as primary |
| `services/market_intelligence/market_intelligence_engine.py` | `_band_from_catalog_tier` | When DB empty, all deal quality derives from `_ACQUISITION_TIER_MUSD` | **HIGH** | Contract test: band mid matches tier; alert on tier table edits |
| `tests/e2e/listing_validation_suite.py` | `infer_listing_verdict` | KPI definition lives in test code, not production UI | **HIGH** | Extract shared classifier or assert on `data_used.deal_quality` only |
| `services/response/response_normalizer.py` | Post-layer chain order | Order: acquisition → market → executive → truth → conversation | **HIGH** | Snapshot test on layer flags for fixed queries |

---

## Medium-severity risks

| File | Function / area | Impact | Severity | Recommended protection |
|------|-----------------|--------|----------|------------------------|
| `services/broker_reasoning/category_resolver.py` | `_ACQUISITION_TIER_MUSD`, `resolve_category` | Single dict drives benchmark ground truth, bands, mission fit | **MEDIUM** | Versioned tier table + changelog review on edit |
| `services/market_reality/market_reality_layer.py` | `apply_market_reality_layer` | Prepends/replaces answer; can drop executive model tokens | **MEDIUM** | Assert model name in final answer for listing queries |
| `services/routing/authority_dispatch.py` | `consult_authority_dispatch` | Comparison / buy / valuation routing | **MEDIUM** | Extend `production_query_replay` golden kinds |
| `services/comparison/comparison_pipeline_v2_responder.py` | Comparison V2 | INSUFFICIENT_DATA paths for unregistered models | **MEDIUM** | Registry lock tests when adding models |
| `services/catalog/alias_expansion_engine.py` | `_comparison_models` integration | Shorthand resolution before dispatch | **MEDIUM** | Alias expansion unit tests per shorthand |
| `services/client_context/recommendation_consistency.py` | `filter_models_for_consistency` | Budget/mfr filter drops query-focus models | **MEDIUM** | `pinned_models` must stay wired from selector |
| `services/broker_decision/mission_fit_scorer.py` | `score_model_fit` | Mission boosts (Europe–US, coast) affect primary | **MEDIUM** | Mission scenario golden set (coast, europe_us) |
| `tests/e2e/production_audit_helpers.py` | `replay_query` | Pass = no authority_error only; ignores primary quality | **MEDIUM** | Add optional primary-required flag per category |

---

## Low-severity risks

| File | Function / area | Impact | Severity | Recommended protection |
|------|-----------------|--------|----------|------------------------|
| `services/executive_broker/executive_broker_layer.py` | `_should_apply_executive` | Skips executive on comparisons (expected) | **LOW** | Document skip rules in benchmark gap doc |
| `services/market_reality/listing_detector.py` | `detect_listing_signal` | Tail vs listing vs none | **LOW** | Tail suite (6 cases) + listing suite |
| `runners/run_phase53_audit.py` | `_read_report_metric` | Dashboard KPI parse | **LOW** | Parse only Summary table rows |
| `services/broker_audit/broker_trust_score.py` | Trust scoring | Trust &lt; 95 on comparisons drags corpus avg | **LOW** | Category-scoped trust KPIs |

---

## Area-specific regression surfaces

### Aircraft recommendations

- `recommendation_selector._gather_candidate_rows` → `budget_matcher` → `mission_fit_scorer.rank_models_for_recommendation`
- Executive skip regex `_SKIP_RE` in `executive_broker_layer`
- **Risk:** mission queries without parsed budget never get executive primary (see Phase 54 executive consistency: 5% primary on mission replay).

### Listing validation

- Production path: `market_reality_layer` → `deal_quality` in `data_used`
- Benchmark path: `infer_listing_verdict` tier heuristics + compatible pairs
- **Risk:** benchmark can pass while production prose still uses skepticism boilerplate.

### Tail investigations

- Only 6 tail registrations in `tail_investigation_suite.py`
- Authority expects `valuation` or `tail_investigation_dispatch`
- **Risk:** new tail formats or international regs not covered.

### Comparison engine

- 100 comparison queries in production corpus; executive primary often empty (by design)
- Registry lock in `aircraft_registry_lock.py`
- **Risk:** Phenom 300, HondaJet, out-of-registry models → insufficient comparison.

### Acquisition reality checks

- `acquisition_budget_reality.apply_acquisition_budget_reality` runs twice in pipeline (normalizer + certify path)
- Listing infeasible ratio caps (`0.30` / `0.36`) are magic numbers tied to certification
- **Risk:** threshold edits move certification without market justification.
