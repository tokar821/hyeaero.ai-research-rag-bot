# Random Sample Audit

Generated: 2026-06-03 10:31 UTC

Seed: 53 | Sample size: 20

## txn_2019_challenger_350_17m (real aircraft)

- **Query:** 2019 Challenger 350 for $17M — good deal?
- **Expected:** Challenger 350
- **Actual primary:** Challenger 350
- **Pass:** True | metrics: `{'primary_acc': 1.0, 'alt_acc': 1.0, 'ultra_penalty': 0.0, 'path': 'layers', 'primary': 'Challenger 350'}`
- **Decision path:** `layers` | authority: `buy_decision`
- **Market logic:** deal_quality=`{'verdict': 'FAIR_DEAL', 'display_verdict': 'FAIR DEAL', 'reason': '5.6% below market median', 'position_pct': -0.05555555555555555}` | market_reality=`{'confidence': 'LIKELY_MARKET', 'reason': 'Within a plausible range vs current listing-derived band.', 'deal_verdict': 'FAIR DEAL', 'position_pct': -0.05555555555555555, 'market_mid_usd': 18000000.0, 'band_low_usd': 12960000.0, 'band_high_usd': 23040000.0}`
- **Flags:** infeas=`None` listing_infeas=`None`

## falcon8x_48m (listing)

- **Query:** Falcon 8X listed at $48M
- **Expected:** REALISTIC
- **Inferred:** FAIR
- **Pass:** True
- **Decision path:** `layers`
- **Market logic:** deal_quality=`{'verdict': 'FAIR_DEAL', 'display_verdict': 'FAIR DEAL', 'reason': '4.0% below market median', 'position_pct': -0.04}` | price_analysis=`{'confidence': 'LIKELY_MARKET', 'reason': 'Within a plausible range vs current listing-derived band.', 'deal_verdict': 'FAIR DEAL', 'position_pct': -0.04, 'market_mid_usd': 50000000.0, 'band_low_usd': 36000000.0, 'band_high_usd': 64000000.0}`

## g280_14m (listing)

- **Query:** Gulfstream G280 at $14M
- **Expected:** FAIR
- **Inferred:** OVERPRICED
- **Pass:** True
- **Decision path:** `layers`
- **Market logic:** deal_quality=`{'verdict': 'OVERPRICED', 'display_verdict': 'OVERPRICED', 'reason': '16.7% above market median', 'position_pct': 0.16666666666666666}` | price_analysis=`{'confidence': 'UNUSUALLY_EXPENSIVE', 'reason': '16.7% above market median', 'deal_verdict': 'OVERPRICED', 'position_pct': 0.16666666666666666, 'market_mid_usd': 12000000.0, 'band_low_usd': 8640000.0, 'band_high_usd': 15360000.0}`

## cheap_gulfstream (real aircraft)

- **Query:** cheap gulfstream
- **Expected:** G280
- **Actual primary:** Gulfstream G280
- **Pass:** True | metrics: `{'primary_acc': 1.0, 'alt_acc': 1.0, 'ultra_penalty': 0.0, 'path': 'layers', 'primary': 'Gulfstream G280'}`
- **Decision path:** `layers` | authority: `alternative`
- **Market logic:** deal_quality=`None` | market_reality=`{}`
- **Flags:** infeas=`None` listing_infeas=`None`

## best_jet_21m (real aircraft)

- **Query:** What is the best jet I can buy for $21M?
- **Expected:** (budget/mission)
- **Actual primary:** Challenger 350
- **Pass:** True | metrics: `{'primary_acc': 1.0, 'alt_acc': 1.0, 'ultra_penalty': 0.0, 'path': 'layers', 'primary': 'Challenger 350'}`
- **Decision path:** `layers` | authority: `buy_decision`
- **Market logic:** deal_quality=`None` | market_reality=`{}`
- **Flags:** infeas=`None` listing_infeas=`None`

## best_jet_27m (real aircraft)

- **Query:** What is the best jet I can buy for $27M?
- **Expected:** (budget/mission)
- **Actual primary:** Citation Longitude
- **Pass:** True | metrics: `{'primary_acc': 1.0, 'alt_acc': 1.0, 'ultra_penalty': 0.0, 'path': 'layers', 'primary': 'Citation Longitude'}`
- **Decision path:** `layers` | authority: `buy_decision`
- **Market logic:** deal_quality=`None` | market_reality=`{}`
- **Flags:** infeas=`None` listing_infeas=`None`

## deal_challenger_350_16m (real aircraft)

- **Query:** Is Challenger 350 at $16M a good deal?
- **Expected:** Challenger 350
- **Actual primary:** Challenger 350
- **Pass:** True | metrics: `{'primary_acc': 1.0, 'alt_acc': 1.0, 'ultra_penalty': 0.0, 'path': 'layers', 'primary': 'Challenger 350'}`
- **Decision path:** `layers` | authority: ``
- **Market logic:** deal_quality=`{'verdict': 'FAIR_DEAL', 'display_verdict': 'FAIR DEAL', 'reason': '11.1% below market median', 'position_pct': -0.1111111111111111}` | market_reality=`{'confidence': 'LIKELY_MARKET', 'reason': 'Within a plausible range vs current listing-derived band.', 'deal_verdict': 'FAIR DEAL', 'position_pct': -0.1111111111111111, 'market_mid_usd': 18000000.0, 'band_low_usd': 12960000.0, 'band_high_usd': 23040000.0}`
- **Flags:** infeas=`None` listing_infeas=`None`

## best_jet_24m (real aircraft)

- **Query:** What is the best jet I can buy for $24M?
- **Expected:** (budget/mission)
- **Actual primary:** Citation Longitude
- **Pass:** True | metrics: `{'primary_acc': 1.0, 'alt_acc': 1.0, 'ultra_penalty': 0.0, 'path': 'layers', 'primary': 'Citation Longitude'}`
- **Decision path:** `layers` | authority: `buy_decision`
- **Market logic:** deal_quality=`None` | market_reality=`{}`
- **Flags:** infeas=`None` listing_infeas=`None`

## global7500_58m (listing)

- **Query:** Global 7500 at $58M
- **Expected:** REALISTIC
- **Inferred:** FAIR
- **Pass:** True
- **Decision path:** `layers`
- **Market logic:** deal_quality=`{'verdict': 'FAIR_DEAL', 'display_verdict': 'FAIR DEAL', 'reason': '0.0% above market median', 'position_pct': 0.0}` | price_analysis=`{'confidence': 'LIKELY_MARKET', 'reason': 'Within a plausible range vs current listing-derived band.', 'deal_verdict': 'FAIR DEAL', 'position_pct': 0.0, 'market_mid_usd': 58000000.0, 'band_low_usd': 41760000.0, 'band_high_usd': 74240000.0}`

## best_jet_29m (real aircraft)

- **Query:** What is the best jet I can buy for $29M?
- **Expected:** (budget/mission)
- **Actual primary:** Challenger 650
- **Pass:** True | metrics: `{'primary_acc': 1.0, 'alt_acc': 1.0, 'ultra_penalty': 0.0, 'path': 'layers', 'primary': 'Challenger 650'}`
- **Decision path:** `layers` | authority: `buy_decision`
- **Market logic:** deal_quality=`None` | market_reality=`{}`
- **Flags:** infeas=`None` listing_infeas=`None`

## best_jet_9m (real aircraft)

- **Query:** What is the best jet I can buy for $9M?
- **Expected:** (budget/mission)
- **Actual primary:** Pilatus PC-24
- **Pass:** True | metrics: `{'primary_acc': 1.0, 'alt_acc': 1.0, 'ultra_penalty': 0.0, 'path': 'layers', 'primary': 'Pilatus PC-24'}`
- **Decision path:** `layers` | authority: `buy_decision`
- **Market logic:** deal_quality=`None` | market_reality=`{}`
- **Flags:** infeas=`None` listing_infeas=`None`

## deal_falcon_8x_45m (real aircraft)

- **Query:** Is Falcon 8X at $45M a good deal?
- **Expected:** Falcon 8X
- **Actual primary:** Falcon 8X
- **Pass:** True | metrics: `{'primary_acc': 1.0, 'alt_acc': 1.0, 'ultra_penalty': 0.0, 'path': 'layers', 'primary': 'Falcon 8X'}`
- **Decision path:** `layers` | authority: ``
- **Market logic:** deal_quality=`{'verdict': 'FAIR_DEAL', 'display_verdict': 'FAIR DEAL', 'reason': '10.0% below market median', 'position_pct': -0.1}` | market_reality=`{'confidence': 'LIKELY_MARKET', 'reason': 'Within a plausible range vs current listing-derived band.', 'deal_verdict': 'FAIR DEAL', 'position_pct': -0.1, 'market_mid_usd': 50000000.0, 'band_low_usd': 36000000.0, 'band_high_usd': 64000000.0}`
- **Flags:** infeas=`None` listing_infeas=`None`

## gs_under_14m (real aircraft)

- **Query:** I want a Gulfstream under $14M
- **Expected:** G280
- **Actual primary:** 
- **Pass:** True | metrics: `{'primary_acc': 1.0, 'alt_acc': 1.0, 'ultra_penalty': 0.0, 'path': 'layers', 'primary': ''}`
- **Decision path:** `layers` | authority: ``
- **Market logic:** deal_quality=`None` | market_reality=`{}`
- **Flags:** infeas=`None` listing_infeas=`None`

## global7500_25m (listing)

- **Query:** Global 7500 for $25M — realistic?
- **Expected:** IMPOSSIBLE
- **Inferred:** IMPOSSIBLE
- **Pass:** True
- **Decision path:** `layers`
- **Market logic:** deal_quality=`{'verdict': 'GOOD_DEAL', 'display_verdict': 'GOOD DEAL', 'reason': '56.9% below market median', 'position_pct': -0.5689655172413793}` | price_analysis=`{'confidence': 'POTENTIAL_DATA_ERROR', 'reason': 'Ask sits far below the verified listing band — verify the listing is real and complete.', 'deal_verdict': 'GOOD DEAL', 'position_pct': -0.5689655172413793, 'market_mid_usd': 58000000.0, 'band_low_usd': 41760000.0, 'band_high_usd': 74240000.0}`

## gs_under_16m (real aircraft)

- **Query:** I want a Gulfstream under $16M
- **Expected:** (budget/mission)
- **Actual primary:** 
- **Pass:** True | metrics: `{'primary_acc': 1.0, 'alt_acc': 1.0, 'ultra_penalty': 0.0, 'path': 'layers', 'primary': ''}`
- **Decision path:** `layers` | authority: ``
- **Market logic:** deal_quality=`None` | market_reality=`{}`
- **Flags:** infeas=`None` listing_infeas=`None`

## praetor_11m (listing)

- **Query:** Praetor 600 at $11M — suspicious?
- **Expected:** SUSPICIOUS
- **Inferred:** SUSPICIOUS
- **Pass:** True
- **Decision path:** `layers`
- **Market logic:** deal_quality=`{'verdict': 'GOOD_DEAL', 'display_verdict': 'GOOD DEAL', 'reason': '38.9% below market median', 'position_pct': -0.3888888888888889}` | price_analysis=`{'confidence': 'UNUSUALLY_CHEAP', 'reason': '38.9% below market median', 'deal_verdict': 'GOOD DEAL', 'position_pct': -0.3888888888888889, 'market_mid_usd': 18000000.0, 'band_low_usd': 12960000.0, 'band_high_usd': 23040000.0}`

## regional_4pax_8m (real aircraft)

- **Query:** Regional US, 4 passengers, $8M budget
- **Expected:** Citation CJ4, Phenom 300
- **Actual primary:** Citation CJ4
- **Pass:** True | metrics: `{'primary_acc': 1.0, 'alt_acc': 1.0, 'ultra_penalty': 0.0, 'path': 'layers', 'primary': 'Citation CJ4'}`
- **Decision path:** `layers` | authority: ``
- **Market logic:** deal_quality=`None` | market_reality=`{}`
- **Flags:** infeas=`None` listing_infeas=`None`

## best_jet_7m (real aircraft)

- **Query:** What is the best jet I can buy for $7M?
- **Expected:** (budget/mission)
- **Actual primary:** Citation CJ4
- **Pass:** True | metrics: `{'primary_acc': 1.0, 'alt_acc': 1.0, 'ultra_penalty': 0.0, 'path': 'layers', 'primary': 'Citation CJ4'}`
- **Decision path:** `layers` | authority: `buy_decision`
- **Market logic:** deal_quality=`None` | market_reality=`{}`
- **Flags:** infeas=`None` listing_infeas=`None`

## deal_phenom_300_8m (real aircraft)

- **Query:** Is Phenom 300 at $8M a good deal?
- **Expected:** Phenom 300
- **Actual primary:** Phenom 300
- **Pass:** True | metrics: `{'primary_acc': 1.0, 'alt_acc': 1.0, 'ultra_penalty': 0.0, 'path': 'layers', 'primary': 'Phenom 300'}`
- **Decision path:** `layers` | authority: ``
- **Market logic:** deal_quality=`{'verdict': 'FAIR_DEAL', 'display_verdict': 'FAIR DEAL', 'reason': '11.1% below market median', 'position_pct': -0.1111111111111111}` | market_reality=`{'confidence': 'LIKELY_MARKET', 'reason': 'Within a plausible range vs current listing-derived band.', 'deal_verdict': 'FAIR DEAL', 'position_pct': -0.1111111111111111, 'market_mid_usd': 9000000.0, 'band_low_usd': 6480000.0, 'band_high_usd': 11520000.0}`
- **Flags:** infeas=`None` listing_infeas=`None`

## falcon8x_50m (real aircraft)

- **Query:** Falcon 8X for $50M — market realistic?
- **Expected:** Falcon 8X
- **Actual primary:** Falcon 8X
- **Pass:** True | metrics: `{'primary_acc': 1.0, 'alt_acc': 1.0, 'ultra_penalty': 0.0, 'path': 'layers', 'primary': 'Falcon 8X'}`
- **Decision path:** `layers` | authority: ``
- **Market logic:** deal_quality=`{'verdict': 'FAIR_DEAL', 'display_verdict': 'FAIR DEAL', 'reason': '0.0% above market median', 'position_pct': 0.0}` | market_reality=`{'confidence': 'LIKELY_MARKET', 'reason': 'Within a plausible range vs current listing-derived band.', 'deal_verdict': 'FAIR DEAL', 'position_pct': 0.0, 'market_mid_usd': 50000000.0, 'band_low_usd': 36000000.0, 'band_high_usd': 64000000.0}`
- **Flags:** infeas=`None` listing_infeas=`None`

## Summary

Sample pass: **20/20**
