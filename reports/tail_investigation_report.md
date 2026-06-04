# Tail Investigation Accuracy (Phase 53)

Generated: 2026-06-03 08:56 UTC

## Summary

| Metric | Value |
|--------|-------|
| Scenarios | 6 |
| Passed | 6 |
| **Pass rate** | **100.0%** |

| Registry lookup accuracy | 100.0% |
| Listing/dispatch match | 100.0% |
| Valuation authority rate | 100.0% |
| **Tail accuracy (composite)** | **100.0%** |

## Scenario results

- **n650gs** [PASS]: {'registry_ok': 1.0, 'listing_mode_ok': 1.0, 'authority_ok': 1.0, 'answer_mentions_reg': 1.0, 'path': 'layers', 'authority': 'valuation', 'regs_found': ['N650GS']}
- **n650gs_diligence** [PASS]: {'registry_ok': 1.0, 'listing_mode_ok': 1.0, 'authority_ok': 1.0, 'answer_mentions_reg': 1.0, 'path': 'layers', 'authority': 'valuation', 'regs_found': ['N650GS']}
- **n800xx** [PASS]: {'registry_ok': 1.0, 'listing_mode_ok': 1.0, 'authority_ok': 1.0, 'answer_mentions_reg': 1.0, 'path': 'layers', 'authority': 'valuation', 'regs_found': ['N800XX']}
- **n525ab** [PASS]: {'registry_ok': 1.0, 'listing_mode_ok': 1.0, 'authority_ok': 1.0, 'answer_mentions_reg': 1.0, 'path': 'layers', 'authority': 'valuation', 'regs_found': ['N525AB']}
- **n200qs** [PASS]: {'registry_ok': 1.0, 'listing_mode_ok': 1.0, 'authority_ok': 1.0, 'answer_mentions_reg': 1.0, 'path': 'layers', 'authority': 'valuation', 'regs_found': ['N200QS']}
- **n44pj** [PASS]: {'registry_ok': 1.0, 'listing_mode_ok': 1.0, 'authority_ok': 1.0, 'answer_mentions_reg': 1.0, 'path': 'layers', 'authority': 'valuation', 'regs_found': ['N44PJ']}

## Regenerate

```bash
cd backend
PYTHONPATH=. pytest tests/e2e/tail_investigation_suite.py -q
```