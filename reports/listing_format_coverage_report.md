# Listing Format Coverage (Phase 54)

Generated: 2026-06-03 12:22 UTC

## Summary

| Metric | Value |
|--------|-------|
| Scenarios | 13 |
| Passed | 13 |
| **Pass rate** | **100.0%** |

| Format coverage pass rate | 100.0% |
| Cases | 13 |

## Scenario results

- **partial_cj4** [PASS]: {'inferred': 'SUSPICIOUS', 'has_deal_quality': True, 'has_market_reality': True, 'path': 'layers'}
- **unicode_dash** [PASS]: {'inferred': 'FAIR', 'has_deal_quality': True, 'has_market_reality': True, 'path': 'layers'}
- **leading_ask** [PASS]: {'inferred': 'FAIR', 'has_deal_quality': True, 'has_market_reality': True, 'path': 'layers'}
- **ask_range_dash** [PASS]: {'inferred': 'FAIR', 'has_deal_quality': True, 'has_market_reality': True, 'path': 'layers'}
- **multi_model** [PASS]: {'inferred': 'FAIR', 'has_deal_quality': True, 'has_market_reality': True, 'path': 'layers'}
- **ask_no_dollar_sign** [PASS]: {'inferred': 'FAIR', 'has_deal_quality': True, 'has_market_reality': True, 'path': 'layers'}
- **ask_word_million** [PASS]: {'inferred': 'OVERPRICED', 'has_deal_quality': False, 'has_market_reality': False, 'path': 'layers'}
- **comma_thousands** [PASS]: {'inferred': 'FAIR', 'has_deal_quality': True, 'has_market_reality': True, 'path': 'layers'}
- **ambiguous_no_price** [PASS]: {'inferred': 'FAIR', 'has_deal_quality': False, 'has_market_reality': True, 'path': 'layers'}
- **malformed_price** [PASS]: {'inferred': 'REALISTIC', 'has_deal_quality': False, 'has_market_reality': False, 'path': 'layers'}
- **partial_global** [PASS]: {'inferred': 'FAIR', 'has_deal_quality': True, 'has_market_reality': True, 'path': 'layers'}
- **multi_partial_models** [PASS]: {'inferred': 'IMPOSSIBLE', 'has_deal_quality': True, 'has_market_reality': True, 'path': 'layers'}
- **word_forty_million** [PASS]: {'inferred': 'GOOD_DEAL', 'has_deal_quality': False, 'has_market_reality': True, 'path': 'layers'}

## Regenerate

```bash
cd backend
PYTHONPATH=. pytest tests/e2e/listing_format_coverage_suite.py -q
```