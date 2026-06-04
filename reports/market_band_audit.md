# Market Band Audit (Phase 53)

Generated: 2026-06-03

## Band sources (priority)

1. **Listing IQR band** — `build_market_band_from_asks` when ≥5 listings (unchanged).
2. **Authority catalog** — `expected_market_band_usd` when synced (unchanged).
3. **Catalog acquisition tier** (new fallback) — `_band_from_catalog_tier`: mid = tier×$1M, low = 0.72×mid, high = 1.28×mid, confidence MODERATE.

## Acquisition tier table (USD millions)

Used by `category_resolver._ACQUISITION_TIER_MUSD` and catalog bands:

| Model | Tier (mid) | Low (72%) | High (128%) |
|-------|------------|-------------|-------------|
| Gulfstream G280 | 12 | 8.6 | 15.4 |
| Gulfstream G650 | 45 | 32.4 | 57.6 |
| Gulfstream G700 | 65 | 46.8 | 83.2 |
| Falcon 2000 | 18 | 13.0 | 23.0 |
| Falcon 7X | 35 | 25.2 | 44.8 |
| Falcon 8X | 50 | 36.0 | 64.0 |
| Citation CJ4 | 7 | 5.0 | 9.0 |
| Citation Longitude | 22 | 15.8 | 28.2 |
| Challenger 350 | 18 | 13.0 | 23.0 |
| Challenger 650 | 28 | 20.2 | 35.8 |
| Global 7500 | 58 | 41.8 | 74.2 |
| Praetor 600 | 18 | 13.0 | 23.0 |
| Phenom 300 | 9 | 6.5 | 11.5 |
| Pilatus PC-24 | 9 | 6.5 | 11.5 |

## Verdict thresholds (`deal_quality_engine`)

| Verdict | Rule |
|---------|------|
| GOOD_DEAL | ask ≤ mid × (1 − 0.12) |
| FAIR_DEAL | between good and overpriced |
| OVERPRICED | ask ≥ mid × (1 + 0.15) |
| INSUFFICIENT_DATA | no band mid |

## Listing confidence labels (`listing_confidence_analyzer`)

| Label | Condition |
|-------|-----------|
| POTENTIAL_DATA_ERROR | ask &lt; mid × 0.45 (or &lt; low × 0.45) |
| UNUSUALLY_CHEAP | deal GOOD_DEAL |
| UNUSUALLY_EXPENSIVE | deal OVERPRICED |
| LIKELY_MARKET | else |

## Manufacturer coverage

| OEM | Models in tier table |
|-----|---------------------|
| Gulfstream | G280, G650, G700 |
| Dassault | Falcon 2000, 7X, 8X |
| Bombardier | Challenger 350/650/Longitude, Global 6500/7500, Learjet 75 |
| Textron | CJ2, CJ4, Latitude, Longitude |
| Embraer | Praetor 600, Phenom 300 |
| Pilatus | PC-24 |

## Benchmark alignment

Listing validation tier heuristic (when `deal_quality` absent) uses ratio = ask/tier:

- &lt;0.45 → IMPOSSIBLE  
- &lt;0.72 → SUSPICIOUS  
- &lt;0.92 → GOOD_DEAL  
- &gt;1.22 → OVERPRICED  
- &gt;1.18 → SUSPICIOUS  
- else → FAIR  

These align with transaction-advisor bands for certification measurement.
