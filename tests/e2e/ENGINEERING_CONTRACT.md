# Engineering Contract — Execution / Observability / CI

## Three layers (never entangle)

| Layer | Location | Rule |
|-------|----------|------|
| **Execution** | `broker_certify`, services pipeline | Produces answers and raw `data_used` |
| **Observability** | `pipeline_observability.py` | Mirrors execution only — no inference |
| **CI** | `tests/e2e/*_suite.py`, `tests/test_*` | Asserts semantic correctness |

## Observability mapping

```python
execution = build_execution_result(data_used, path=path, prefer_e2e=prefer_e2e)
attach_observability(data_used, execution)  # no inference — mirrors execution only
```

## Observability contract (required after every `broker_certify`)

```python
data_used = {
    "execution_path": "e2e | layers",
    "broker_certify_path": "e2e | layers",
    "broker_certify_prefer_e2e": bool,
    "tier_source": str,
    "market_source": str,
    "tier_fallback_used": bool,
    "executive_applied": bool,
    "deal_quality_observed": bool,
}
```

Optional when tier catalog fallback ran: `acquisition_tier_catalog_version`, `tier_fallback_checksum`.

## Debugging order

1. `execution_path` / `broker_certify_path` — which lens ran?
2. `executive_applied` — was executive layer applied?
3. `tier_source` / `tier_fallback_used` — catalog SPOF involved?
4. `market_source` — listing DB vs authority vs catalog?
5. `deal_quality_observed` — market verdict present?

## Mission KPI (CI)

- Path: **layers** only (`execution_path_config.REPLAY_LAYERS_CATEGORIES`)
- Per query: `mission_primary_present`, `mission_semantic_ok`, `executive_applied`
- Session: `mission_primary_rate` and `mission_semantic_ok_rate` ≥ 80%

## Invalid test suites

A suite is **invalid** if it passes without asserting semantic output correctness (path-only checks).

Run: `python scripts/audit_benchmark_assertions.py`

## Production API logs (observe-only)

| Log | Meaning |
|-----|---------|
| `HARDENING_ROUTING_FAILURE` | Router left `execution_path=none` without documented deferral signals — investigate routing |
| `HARDENING_ROUTING_DEFERRED` | Expected path not set, but router signals explain why (e.g. `mixed_fact_and_capability`, `no_model`) — hybrid pipeline still runs |
| `acquisition_tier_catalog_fallback` | Market band used catalog tier v1.0.0 (see `tier_fallback_used` in test `data_used`) |
| `api contract versioning skipped: render_hints unknown keys` | Fixed when `single_authority` is registered in render contract |

## Tier governance

- Source: `services/broker_reasoning/acquisition_tier_catalog.py`
- CI checksum: `tests/test_acquisition_tier_catalog.py`
- Fallback logged in execution + mirrored in `tier_fallback_used`
