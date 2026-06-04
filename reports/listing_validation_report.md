# Listing Validation Suite (Phase 53)

Generated: 2026-06-03 12:50 UTC

## Summary

| Metric | Value |
|--------|-------|
| Scenarios | 20 |
| Passed | 20 |
| **Pass rate** | **100.0%** |

| Listing assessment accuracy | 100.0% |
| Cases | 20 |

## Scenario results

- **g650_18m** [PASS]: {'correct': 1.0, 'expected': 'SUSPICIOUS', 'inferred': 'IMPOSSIBLE', 'path': 'layers'}
- **g650_42m** [PASS]: {'correct': 1.0, 'expected': 'FAIR', 'inferred': 'FAIR', 'path': 'layers'}
- **g650_55m** [PASS]: {'correct': 1.0, 'expected': 'OVERPRICED', 'inferred': 'OVERPRICED', 'path': 'layers'}
- **g700_12m** [PASS]: {'correct': 1.0, 'expected': 'IMPOSSIBLE', 'inferred': 'IMPOSSIBLE', 'path': 'layers'}
- **g700_60m** [PASS]: {'correct': 1.0, 'expected': 'REALISTIC', 'inferred': 'FAIR', 'path': 'layers'}
- **longitude_10m** [PASS]: {'correct': 1.0, 'expected': 'GOOD_DEAL', 'inferred': 'SUSPICIOUS', 'path': 'layers'}
- **longitude_22m** [PASS]: {'correct': 1.0, 'expected': 'FAIR', 'inferred': 'FAIR', 'path': 'layers'}
- **longitude_28m** [PASS]: {'correct': 1.0, 'expected': 'OVERPRICED', 'inferred': 'OVERPRICED', 'path': 'layers'}
- **falcon8x_14m** [PASS]: {'correct': 1.0, 'expected': 'IMPOSSIBLE', 'inferred': 'IMPOSSIBLE', 'path': 'layers'}
- **falcon8x_48m** [PASS]: {'correct': 1.0, 'expected': 'REALISTIC', 'inferred': 'FAIR', 'path': 'layers'}
- **challenger350_7m** [PASS]: {'correct': 1.0, 'expected': 'SUSPICIOUS', 'inferred': 'IMPOSSIBLE', 'path': 'layers'}
- **challenger350_17m** [PASS]: {'correct': 1.0, 'expected': 'REALISTIC', 'inferred': 'FAIR', 'path': 'layers'}
- **g280_11m** [PASS]: {'correct': 1.0, 'expected': 'GOOD_DEAL', 'inferred': 'FAIR', 'path': 'layers'}
- **g280_14m** [PASS]: {'correct': 1.0, 'expected': 'FAIR', 'inferred': 'OVERPRICED', 'path': 'layers'}
- **cj4_4m** [PASS]: {'correct': 1.0, 'expected': 'GOOD_DEAL', 'inferred': 'SUSPICIOUS', 'path': 'layers'}
- **cj4_9m** [PASS]: {'correct': 1.0, 'expected': 'OVERPRICED', 'inferred': 'OVERPRICED', 'path': 'layers'}
- **praetor_11m** [PASS]: {'correct': 1.0, 'expected': 'SUSPICIOUS', 'inferred': 'SUSPICIOUS', 'path': 'layers'}
- **praetor_19m** [PASS]: {'correct': 1.0, 'expected': 'REALISTIC', 'inferred': 'FAIR', 'path': 'layers'}
- **global7500_25m** [PASS]: {'correct': 1.0, 'expected': 'IMPOSSIBLE', 'inferred': 'IMPOSSIBLE', 'path': 'layers'}
- **global7500_58m** [PASS]: {'correct': 1.0, 'expected': 'REALISTIC', 'inferred': 'FAIR', 'path': 'layers'}

## Regenerate

```bash
cd backend
PYTHONPATH=. pytest tests/e2e/listing_validation_suite.py -q
```