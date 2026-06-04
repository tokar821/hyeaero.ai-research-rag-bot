# Listing Realism Report (Phase 51/54)

Generated: 2026-06-03 11:59 UTC

## Summary

| Metric | Value |
|--------|-------|
| Scenarios | 5 |
| Passed | 5 |
| **Pass rate** | **100.0%** |

| Correctly Identified | 100.0% |

## Scenario results

- **g650_18m** [PASS]: {'correct': 1.0, 'expected': 'SUSPICIOUS', 'inferred': 'IMPOSSIBLE', 'path': 'layers'}
- **g700_12m** [PASS]: {'correct': 1.0, 'expected': 'IMPOSSIBLE', 'inferred': 'IMPOSSIBLE', 'path': 'layers'}
- **longitude_10m** [PASS]: {'correct': 1.0, 'expected': 'GOOD_DEAL', 'inferred': 'GOOD_DEAL', 'path': 'layers'}
- **falcon8x_14m** [PASS]: {'correct': 1.0, 'expected': 'IMPOSSIBLE', 'inferred': 'IMPOSSIBLE', 'path': 'layers'}
- **challenger350_7m** [PASS]: {'correct': 1.0, 'expected': 'SUSPICIOUS', 'inferred': 'IMPOSSIBLE', 'path': 'layers'}

## Regenerate

```bash
cd backend
PYTHONPATH=. pytest tests/e2e/listing_realism_suite.py -q
```