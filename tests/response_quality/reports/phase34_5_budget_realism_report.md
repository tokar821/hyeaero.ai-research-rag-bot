# Phase 34.5 — Budget Realism & Valuation Alignment

**Date:** 2026-06-01  
**E2E:** `RUN_RESPONSE_QUALITY_E2E=1` · 100-query broker review set  
**Baseline:** Phase 34.4 (`phase34_4_catalog_expansion_report.md`)

---

## Executive Summary

| Criterion | Target | Result |
|-----------|--------|--------|
| `BROKER_BUDGET_MISMATCH` ≤ 2 | Yes | **0** |
| Broker Quality Score ≥ 99 | Yes | **100.0** |
| `BROKER_BAD_AIRCRAFT` = 0 | Yes | **0** |
| `COMPARISON_NO_VERDICT` = 0 | Yes | **0** |
| `COMPARISON_INCOMPLETE` = 0 | Yes | **0** |
| `UNJUSTIFIED_MODEL_INSERTION` = 0 | Yes | **0** |

---

## Before vs After (E2E)

| Metric | Phase 34.4 | Phase 34.5 | Δ |
|--------|------------|------------|---|
| Broker Quality Score | 99.69 | **100.0** | +0.31 |
| Broker recommendation accuracy | 89% | **100%** | +11 pts |
| `BROKER_BUDGET_MISMATCH` | 10 | **0** | −10 |
| All other broker/consistency flags | 0 | **0** | — |

---

## Step 1 — Audit Table (10 failures @ baseline)

All 10 were **buy_decision** price-matrix queries. The dollar amount in the query is the **listing ask**, not an acquisition budget cap.

| Query ID | Query (ask) | Authority models | Catalog typical price | Audit treated ask as budget cap (85%) | Failure category |
|----------|-------------|------------------|----------------------|----------------------------------------|------------------|
| buy-001 | $5M | Citation Latitude | ~$11.8M | $4.25M cap | **E — Audit false positive** |
| buy-002 | $6M | Citation Latitude | ~$11.8M | $5.1M cap | **E** |
| buy-003 | $8M | Citation Latitude | ~$11.8M | $6.8M cap | **E** |
| buy-004 | $10M | Citation Latitude | ~$11.8M | $8.5M cap | **E** |
| buy-005 | $12M | Citation Latitude | ~$11.8M | $10.2M cap | **E** |
| buy-011 | $5M (2016) | Citation Latitude | ~$11.8M | $4.25M cap | **E** |
| buy-012 | $6M (2016) | Citation Latitude | ~$11.8M | $5.1M cap | **E** |
| buy-013 | $8M (2016) | Citation Latitude | ~$11.8M | $6.8M cap | **E** |
| buy-014 | $10M (2016) | Citation Latitude | ~$11.8M | $8.5M cap | **E** |
| buy-015 | $12M (2016) | Citation Latitude | ~$11.8M | $10.2M cap | **E** |

**Category counts (baseline):**

| Category | Count |
|----------|------:|
| A. Aircraft exceeds budget | 0 |
| B. Aircraft below budget but wrong tier | 0 |
| C. Valuation inconsistency | 0 |
| D. Buy-decision market-range mismatch | 0 |
| **E. Audit false positive** | **10** |

**Note:** buy-006 … buy-010 ($15M–$30M asks) already passed because catalog price &lt; 85% of stated ask.

---

## Step 2 — First source of mismatch

| Layer | Finding |
|-------|---------|
| **IntentLock** | Correct — `canonical_models: [Citation Latitude]`, no budget constraint (buy intent). |
| **Budget gate** | Not invoked on buy-decision dispatch path (by design). |
| **Ranker** | Not used for buy_decision. |
| **Final formatter** | `respond_buy_decision` / deal killer — verdicts were conservative (`HIGH RISK` / `FAIR DEAL`) when comps empty, but **not** the audit failure driver. |
| **Broker audit** | **Root cause:** `broker_recommendation_audit.py` used `extract_money_musd(query)` as acquisition budget for *all* queries. Buy-path asks ($5M–$12M) were compared to catalog typical price (~$11.8M) with an 85% cap → false `BROKER_BUDGET_MISMATCH`. |

---

## Step 3 — Hard budget enforcement (mission)

No change required for this phase. `apply_budget_gate()` in `recommendation_engine.py` already drops candidates where `typical_market_price_usd > budget_usd * 0.85`. Mission queries with `under $XM` use `extract_acquisition_budget_musd()` in the audit (excludes buy-price phrasing).

---

## Step 4 — Buy-decision consistency (production)

| Change | Purpose |
|--------|---------|
| `respond_buy_decision` merges `build_authoritative_market_context` bands when listing comps are empty | Same low/high/mid band for Market Reality and deal killer |
| `deal_killer_engine` maps `suspiciously_low` ask → `GOOD DEAL`; `over_range_high` → `OVERPRICED` | Ask vs band alignment |
| Authority-band path skips thin-comp liquidity penalty | Avoid spurious `HIGH RISK` when comps=0 but band is verified |
| `_clean_buy_model()` strips trailing `for` from parsed model | Fixes `Citation Latitude for` parse artifact |
| Market Reality shows authority band when `authority_band` set | Visible typical range in answer |

---

## Step 5 — Audit false positives (evidence)

**Proven auditor defect (fixed):**

- Query: `Is a 2015 Citation Latitude for $5M a good deal?`
- Before: `extract_money_musd` → 5.0 → cap $4.25M &lt; catalog $11.8M → `BROKER_BUDGET_MISMATCH`
- After: `is_buy_price_query()` → true → budget check **skipped** → no mismatch
- Mission control: `Need 8 passengers TEB to LAX nonstop under $10M` still applies budget via `extract_acquisition_budget_musd()` → 10.0

**Buy-path consistency guard (audit):** flags mismatch only if answer says `GOOD DEAL` while ask &gt; 115% of catalog, or `overpriced` while ask &lt; 70% of catalog (verdict/price incoherence).

---

## Files / Functions Changed

| File | Functions / area |
|------|------------------|
| `tests/response_quality/_text_extract.py` | `is_buy_price_query`, `extract_ask_musd`, `extract_acquisition_budget_musd` |
| `tests/response_quality/broker_recommendation_audit.py` | Budget check scope + buy verdict coherence |
| `services/routing/authority_dispatch.py` | `_clean_buy_model`, `respond_buy_decision`, `_format_buy_decision_response` |
| `services/deal_killer_engine.py` | Authority-band liquidity; `suspiciously_low` → `GOOD DEAL` |
| `tests/response_quality/test_phase34_5_budget_audit.py` | **New** regression tests |

**Not modified:** IntentLock, dispatch ordering, AKAL, comparison pipeline, replay, deterministic guard, recovery authority layer.

---

## Regression Results

```text
pytest tests/response_quality/test_phase34_5_budget_audit.py
     tests/test_deal_killer_engine.py
     tests/test_authority_dispatch.py
→ 26 passed

RUN_RESPONSE_QUALITY_E2E=1
pytest tests/response_quality/test_response_quality.py::test_broker_review_set_response_quality_report
→ 100 queries, 0 findings, score 100.0
```

---

## Remaining Risks

| Risk | Notes |
|------|-------|
| Buy audit coherence heuristics | Only extreme verdict/price contradictions are flagged; nuanced copy still human-reviewed |
| Authority band vs live comps | When DB comps exist, listing comps take precedence; bands are fallback only |
| Mission budget | Still uses 85% buffer; tightening to 100% would be a separate product decision |

---

## Unresolved Mismatches

**None** on the 100-query broker review set after Phase 34.5.
