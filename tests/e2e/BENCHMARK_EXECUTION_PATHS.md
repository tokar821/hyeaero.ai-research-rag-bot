# Benchmark Execution Paths

## Paths

| Path | `broker_certify(prefer_e2e=...)` | Post-layers applied |
|------|-----------------------------------|---------------------|
| **e2e** | `True` | No — returns retrieval answer as-is |
| **layers** | `False` | Yes — acquisition, market, executive, conversation |

## Suite policy (enforced in code)

| Suite | `prefer_e2e` | Certification? |
|-------|--------------|----------------|
| `real_aircraft_benchmark.py` | `False` | Yes |
| `listing_validation_suite.py` | `False` | Yes |
| `production_query_replay_suite.py` | Category-based (see below) | Observability |
| `market_recommendation_audit.py` | `False` | Yes (semantic assert) |
| `listing_realism_suite.py` | `False` | Yes (delegates to Phase 53 inference) |
| `tail_investigation_suite.py` | `True` | Partial |
| `recommendation_accuracy_suite.py` | `False` | Yes |

## Production replay category policy

Configured in `execution_path_config.py`:

| Category | Default path |
|----------|--------------|
| mission | layers |
| buy_decision | e2e |
| comparison | e2e |
| valuation | e2e |
| alternative | e2e |
| listing | e2e |

Override: `HYEAERO_REPLAY_PREFER_E2E=0` forces layers for all replay categories.

## Parity audit

Run: `pytest tests/e2e/test_execution_path_parity.py -q`

Writes `backend/reports/phase54_execution_path_parity.md` with per-scenario divergence.

| Outcome | Meaning |
|---------|---------|
| **Critical failure** | Fails CI by default (`HYEAERO_PARITY_STRICT=1`): path tag mismatch, layers missing primary on buy/mission, layers empty |
| **Warning** | Expected divergence (e.g. layers has executive primary, e2e does not) |

Set `HYEAERO_PARITY_STRICT=0` to record warnings only without failing on critical patterns.

## Pipeline observability (`data_used`)

After every `broker_certify` call, measurement attaches:

| Key | Meaning |
|-----|---------|
| `broker_certify_path` / `execution_path` | `e2e` or `layers` |
| `broker_certify_prefer_e2e` | Requested path policy |
| `tier_source` | Catalog version, feasibility tier, or `none` |
| `market_source` | Band/listing source (incl. `catalog_acquisition_tier` fallback) |
| `executive_applied` | Whether executive layer produced a primary on layers path |
| `deal_quality_observed` | `bool` — verdict present in execution `deal_quality` |
| `tier_fallback_used` | `bool` — catalog tier used for market band fallback |

Observability **reads execution artifacts only** (see `ENGINEERING_CONTRACT.md`).

## Debugging order

1. `execution_path` / `broker_certify_path`
2. `executive_applied`
3. `tier_source` / `tier_fallback_used`
4. `market_source`
5. `deal_quality_observed`

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `HYEAERO_REPLAY_PREFER_E2E` | `1` | Global replay e2e toggle |
| `HYEAERO_MISSION_PRIMARY_MIN_PCT` | `80` | Session gate for mission primary rate |
| `PHASE53_REPLAY_LIMIT` | unset | Cap production replay count |
