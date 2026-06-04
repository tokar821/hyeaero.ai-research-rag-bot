# Authority Misrouting Matrix (Phase 52)

Post-fix audit — all benchmark queries route to expected authority.

| Query | Expected authority | Actual authority | Status |
|-------|-------------------|------------------|--------|
| cheap gulfstream | alternative | alternative | OK |
| g650 for 18m | alternative | alternative | OK |
| longitude vs challenger 350 | comparison | comparison | OK |
| best jet under 20m | alternative | alternative | OK |
| g700 under 5m | alternative | alternative | OK |
| is N650GS worth investigating | valuation | valuation | OK |
| should I buy now or wait | buy_decision | buy_decision | OK |

## Phase 51 misrouting pattern (resolved)

| Query | Expected | Phase 51 actual | Frequency |
|-------|----------|-----------------|-----------|
| cheap gulfstream | alternative | (none) | 1 |
| g650 for 18m | alternative | (none) | 1 |
| best jet under 20m | alternative | (none) | 1 |
| g700 under 5m | alternative | (none) | 1 |
| tail investigation | valuation | (none) | 1 |
| buy now or wait | buy_decision | (none) | 1 |

Primary failure mode: `consult_authority_dispatch` returned `None` and legacy mission pipeline ran without setting `authority_dispatch_kind`.
