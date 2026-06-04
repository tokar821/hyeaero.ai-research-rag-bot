# HyeAero Health Dashboard

Generated: 2026-06-03 09:00 UTC

## Daily KPI Summary

### real_aircraft_benchmark

Pytest: `........................................................................ [ 71%]
.............................                                            [100%]
101 passed in 48.94s`

Report metric: ****79.0%****

### listing_validation_suite

Pytest: `....................                                                     [100%]
20 passed in 10.24s`

Report metric: ****25.0%****

### tail_investigation_suite

Pytest: `......                                                                   [100%]
6 passed in 1.17s`

Report metric: ****100.0%****

### market_recommendation_audit

Pytest: `...............................................                          [100%]
47 passed in 7.88s`

Report metric: ****100.0%****

### production_query_replay_suite

Pytest: `........................................................................ [ 72%]
............................                                             [100%]
100 passed in 218.01s (0:03:38)`

Report metric: ****100.0%****

### test_alias_expansion_engine

Pytest: `...........                                                              [100%]
11 passed in 1.35s`

### KPI targets

| KPI | Target | Source |
|-----|--------|--------|
| Real Aircraft Benchmark | >90% | real_aircraft_benchmark_report.md |
| Listing Validation | >90% | listing_validation_report.md |
| Tail Accuracy | >95% | tail_investigation_report.md |
| Recommendation Drift | <3% | production_query_replay_report.md |
| Authority Error | <2% | production_query_replay_report.md |
| Broker Trust Score | >95 | production_query_replay_report.md |


## Regenerate

```bash
cd backend
PYTHONPATH=. python runners/run_phase53_audit.py
```