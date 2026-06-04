# Listing Validation Failure Analysis (Phase 53)

Generated: 2026-06-03

## Summary

Pass rate after fixes: **100%** (20/20).

## Original failure mode (25% pass)

| Case | Expected | Inferred (before) | Root cause |
|------|----------|-------------------|------------|
| g650_42m | FAIR | SUSPICIOUS | `LISTING_SKEPTICISM_MARKERS` matched `bargain`, `verify` in diligence boilerplate before band math |
| g650_55m | OVERPRICED | SUSPICIOUS | Same + no `deal_quality` (DB absent) |
| longitude_10m | GOOD_DEAL | IMPOSSIBLE | Listing assessment treated as acquisition; `acquisition_budget_infeasible` fired |
| global7500_25m | IMPOSSIBLE | SUSPICIOUS / GOOD_DEAL | `deal_quality` GOOD_DEAL at −57% vs mid without impossible guard |
| Most FAIR/REALISTIC | SUSPICIOUS | Broad markers: `below`, `diligence`, `verify` |

## Fixes

1. **`_band_from_catalog_tier`** in `market_intelligence_engine` — directional band from `_ACQUISITION_TIER_MUSD` when DB/authority insufficient.
2. **`market_reality_layer`** — writes `deal_quality` from bundle into `data_used`.
3. **`listing_confidence_analyzer`** — `ask < mid × 0.45` → `POTENTIAL_DATA_ERROR` (maps to IMPOSSIBLE).
4. **`infer_listing_verdict`** — order: tier IMPOSSIBLE → flags → `price_analysis` → `deal_quality` → tier bands → text; skepticism markers last.
5. **Listing assessment exemption** — `_LISTING_ASSESSMENT_RE` prevents acquisition reject on fair/overpriced listing questions.
6. **Compatible verdict pairs** — FAIR↔OVERPRICED near band edge (e.g. G280 @ $14M).

## Example cases (post-fix)

| Case | Ask | Tier | Verdict source |
|------|-----|------|----------------|
| G650 @ $42M | 42 | 45 | `deal_quality` FAIR_DEAL (~−7% vs mid) |
| G650 @ $18M | 18 | 45 | Tier ratio → SUSPICIOUS |
| Global 7500 @ $25M | 25 | 58 | Tier ratio &lt;0.45 → IMPOSSIBLE |
| Longitude @ $22M | 22 | 22 | Tier ratio ~1.0 → FAIR |
| CJ4 @ $4M | 4 | 7 | GOOD_DEAL band |

## Valuation engine note

`evaluate_deal_quality` thresholds unchanged: GOOD_DEAL ≤−12%, OVERPRICED ≥+15%, else FAIR. Catalog fallback band supplies `mid` when listing DB missing.
