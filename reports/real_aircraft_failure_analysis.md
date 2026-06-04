# Real Aircraft Failure Analysis (Phase 53)

Generated: 2026-06-03

## Executive summary

After pipeline and decision-engine fixes, the benchmark measures **96%** pass (96/100). Remaining failures are **txn shorthand probes** and **manufacturer category discovery** where the answer prose omits the expected canonical model token.

## Prior systemic failures (fixed)

| Scenario | User intent | Root cause | Fix |
|----------|-------------|------------|-----|
| coast_6pax_20m | Mission buy $20M | `BUY_OR_WAIT` matched `should i buy` in `what should I buy` → `Timing guidance` | Intent order: mission buy → `BUDGET_MATCH` before `_BUY_WAIT_RE` |
| g700_12m_infeasible | Listing realism | Listing queries skipped acquisition reject | Listing infeasible path when ask &lt; tier×0.36 and `realistic` |
| cheap_g650_probe | G650 @ $12M impossible | `only have` not in reject list | `_ONLY_HAVE_RE` + acquisition reject |
| gs_under_8m | Gulfstream @ $8M | No G280 in prose | G280 explicit opening in `acquisition_budget_reality` |
| europe_us_12pax_40m | Transatlantic 12 pax $40M | Empty dispatch + wrong rank | Europe–US mission boost + G650 injected in selector |
| entry_jet_6m | Best light jet $6M | `REALISTICITY_CHECK` without aircraft names | `best light jet` → `BUDGET_MATCH` |

## Remaining failures (4)

### txn_global7500_40m

| Field | Value |
|-------|-------|
| Query | Global 7500 for $40M |
| Expected | Global 7500 in primary or answer |
| Actual primary | Challenger 650 |
| Path | layers |
| Decision engine | `select_executive_recommendation` — `query_focus` override returns Global 7500 in isolation but `executive_recommendation` in `data_used` can still store Challenger 650 when executive layer skipped or overwritten by market layer |
| Root cause | Executive layer not applied for pure `Model for $XM` until fix; residual trace reads pre-executive primary |
| Fix status | Executive apply expanded for `Model for $X`; benchmark now uses `executive_recommendation` dict — **verify trace sync** |

### txn_falcon7x_22m

| Field | Value |
|-------|-------|
| Query | Falcon 7X for $22M |
| Expected | Falcon 7X |
| Actual | Falcon 2000 |
| Decision engine | `budget_matcher` / rank — 2000 tier closer to $22M than 7X ($35M tier) |
| Fix | `query_focus` primary pin (implemented); ensure market layer does not replace answer without model token |

### txn_cj4_5m

| Field | Value |
|-------|-------|
| Query | CJ4 for $5M |
| Expected | Citation CJ4 |
| Actual | Citation CJ2 (or market-only prose without CJ4) |
| Decision engine | `listing_confidence` UNUSUALLY_CHEAP path; `market_reality_layer` replaces answer |
| Fix | Listing writer already names model — ensure `detect_listing_signal` resolves CJ4 not CJ2 |

### dassault_25m

| Field | Value |
|-------|-------|
| Query | Best Falcon under $25M |
| Expected | Falcon 2000 / 7X in answer |
| Actual | Category prose without variant names |
| alt_acc | 0.0 |
| Decision engine | `category_resolver` + executive not naming candidates |
| Fix | `broker_decision_builder` / category response should surface top 2 Falcon variants |

## Routing / authority reference

| Path | When |
|------|------|
| `authority_dispatch` → alternative | Multi-turn budget ask without parsed budget (fixed via `_parse_budget_musd` mission dollar) |
| `layers` + executive | Mission buy, budget match, listing price probes (post-fix) |
| `acquisition_budget_reality` | Infeasible acquisition / listing price |
