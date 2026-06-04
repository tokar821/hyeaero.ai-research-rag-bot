# Phase 54 — Executive Layer Consistency Audit

Generated: 2026-06-03

Method: measurement on production replay artifact + live layers-path sampling + 500 synthetic listing probes.

**Note:** `PHASE53_REPLAY_LIMIT=0` in environment causes `load_production_queries()` to return empty; replay analysis uses `production_query_replay_report.md` from the 500-query run (`prefer_e2e=True`).

---

## 1. Production replay (500 queries) — primary recommendation

Parsed from `production_query_replay_report.md` broker trace primary field:

| Category | Queries | Has non-empty primary | Rate |
|----------|---------|----------------------|------|
| comparison | 100 | 60 | **60%** |
| buy_decision | 100 | 87 | **87%** |
| mission | 100 | 5 | **5%** |
| alternative | 100 | 100 | **100%** |
| valuation | 100 | 100 | **100%** |

**Finding:** Mission-category production queries almost never receive an executive primary when measured via e2e replay path. This is a **hidden production risk** not covered by Phase 53 aircraft benchmark (mission scenarios use layers path).

Empty primary on comparison (40%) is often expected (`_SKIP_RE` executive for pure VS queries).

---

## 2. Layers-path sample — buy_decision (n=80)

Live `broker_certify(..., prefer_e2e=False)` on first 80 `buy_decision` production queries:

| Field | Present | Rate |
|-------|---------|------|
| `executive_recommendation.primary_recommendation` | 79 | **98.8%** |
| `deal_quality` in `data_used` | 80 | **100%** |

**Finding:** Layers path (certification path) is healthy for buy_decision primaries. Divergence from replay is **e2e vs layers**, not random flake.

---

## 3. Five hundred listing scenarios (synthetic grid)

Generated: `{model} for ${ask}M — fair price?` across `_ACQUISITION_TIER_MUSD` × ratio grid (`prefer_e2e=False`).

| Field | Present | Rate |
|-------|---------|------|
| `deal_quality` | 483 | **96.6%** |
| `market_reality.price_analysis` | 483 | **96.6%** |
| `market_reality` blob | 483 | **96.6%** |
| Inferred verdict (test classifier) | 500 | **100%** |

Verdict distribution (infer_listing_verdict):

| Verdict | Count |
|---------|-------|
| FAIR | 159 |
| OVERPRICED | 99 |
| IMPOSSIBLE | 84 |
| SUSPICIOUS | 84 |
| GOOD_DEAL | 74 |

**Finding:** Listing **verdict always exists** via classifier; production metadata (`deal_quality`) exists **96.6%** — 17 cases lacked populated deal_quality (likely non-listing dispatch or early return).

---

## 4. Mission fit

| Source | Mission metadata in `broker_reasoning.mission` |
|--------|-----------------------------------------------|
| Production replay mission (e2e) | Not measured in replay artifact |
| Real aircraft coast/europe scenarios (layers) | Present when `mission_interpreter` runs |

**Gap:** No automated assert that `passengers`, `acquisition_budget_musd`, or `range_nm` populated for all mission queries in production replay.

---

## 5. Checklist vs certification criteria

| Criterion | 500 recommendation (production replay) | 500 listing (synthetic) |
|-----------|----------------------------------------|-------------------------|
| Primary always exists | **FAIL** on mission (5%) | N/A (listing uses verdict) |
| Verdict always exists | Partial (trust/drift only) | **PASS** (100% inferred) |
| Mission fit always exists | **UNKNOWN** on replay | N/A |
| Listing verdict always exists | N/A | **PASS** (96.6% deal_quality + 100% infer) |

---

## 6. Trust score consistency (production replay)

From same 500-query report:

- Avg broker trust: **79.4**
- Trust ≥ 95: **0%**

Executive primary can exist while trust remains &lt; 95 (comparison-weighted corpus). **Certification KPI “trust &gt; 95” is not met** on production replay despite 100% pass (pass = authority + no drift only).

---

## Recommendations

1. Add `mission` category to real aircraft benchmark or require executive primary in `replay_query` when category=mission.
2. Standardize certification on the same path as production (`prefer_e2e` policy documented).
3. Store `deal_quality.verdict` in production logs and benchmark against it instead of text-only inference.
4. Report category-scoped trust, not corpus-wide average.
