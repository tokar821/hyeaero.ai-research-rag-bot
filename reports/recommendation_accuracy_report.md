# Recommendation Accuracy Report (Phase 51)

Generated: 2026-06-03 08:20 UTC

## Summary

| Metric | Value |
|--------|-------|
| Scenarios | 5 |
| Passed | 5 |
| **Pass rate** | **100.0%** |

| Primary Recommendation Accuracy | 100.0% |
| Alternative Accuracy | 100.0% |
| Budget Compliance | 100.0% |
| Mission Compliance | 100.0% |

## Scenario results

- **gulfstream_under_12m** [PASS]: {'primary_acc': 1.0, 'alt_acc': 1.0, 'budget_acc': 1.0, 'mission_acc': 1.0, 'path': 'layers', 'primary': ''}
- **coast_to_coast_6pax_20m** [PASS]: {'primary_acc': 1.0, 'alt_acc': 1.0, 'budget_acc': 1.0, 'mission_acc': 1.0, 'path': 'layers', 'primary': 'Citation Longitude'}
- **g700_under_5m** [PASS]: {'primary_acc': 1.0, 'alt_acc': 1.0, 'budget_acc': 1.0, 'mission_acc': 1.0, 'path': 'layers', 'primary': ''}
- **g650_18m** [PASS]: {'primary_acc': 1.0, 'alt_acc': 1.0, 'budget_acc': 1.0, 'mission_acc': 1.0, 'path': 'layers', 'primary': ''}
- **best_jet_15m** [PASS]: {'primary_acc': 1.0, 'alt_acc': 1.0, 'budget_acc': 1.0, 'mission_acc': 1.0, 'path': 'layers', 'primary': 'Gulfstream G280'}

## Regenerate

```bash
cd backend
PYTHONPATH=. pytest tests/e2e/recommendation_accuracy_suite.py -q
```