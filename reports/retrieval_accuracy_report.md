# Retrieval Accuracy Report (Phase 51)

Generated: 2026-06-03 08:20 UTC

## Summary

| Metric | Value |
|--------|-------|
| Scenarios | 7 |
| Passed | 7 |
| **Pass rate** | **100.0%** |

| Top1 Accuracy | 100.0% |
| Top3 Accuracy | 100.0% |
| Wrong Authority % | 0.0% |
| Aircraft Match % | 100.0% |
| Budget Match % | 100.0% |

## Scenario results

- **cheap_gulfstream** [PASS]: pool=['Gulfstream G280']
- **g650_18m** [PASS]: pool=['Gulfstream G280', 'Gulfstream G650']
- **longitude_vs_challenger** [PASS]: pool=['Citation Longitude', 'Challenger 350']
- **best_jet_20m** [PASS]: pool=['Challenger 350']
- **g700_under_5m** [PASS]: pool=['Gulfstream G280', 'Gulfstream G700']
- **tail_investigation** [PASS]: pool=[]
- **buy_now_or_wait** [PASS]: pool=['Timing guidance']

## Regenerate

```bash
cd backend
PYTHONPATH=. pytest tests/e2e/retrieval_accuracy_suite.py -q
```