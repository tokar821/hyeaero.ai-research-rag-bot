# Failure Injection Validation

Generated: 2026-06-03 10:31 UTC

## Method

Temporary defects injected via **pytest monkeypatch** in `tests/verification/test_failure_injection.py` (not committed to production code).

| Injection | Target | Probe case | Expected |
|-----------|--------|------------|----------|
| G280 primary for all | `select_executive_recommendation` | `g700_65m` (expects G700) | `_evaluate` → **fail** |
| Always SUSPICIOUS | `infer_listing_verdict` | `g650_42m` (expects FAIR) | `_compatible` → **fail** |
| Always IMPOSSIBLE | `infer_listing_verdict` | `cj4_4m` (expects GOOD_DEAL) | `_compatible` → **fail** |

## Results

```
2 failed, 3 passed in 4.67s
```

Exit code: 1 (0 = all injection tests passed, meaning defects were detected)

## Before / after

| State | Real aircraft `g700_65m` | Listing `g650_42m` | Listing `cj4_4m` |
|-------|------------------------|--------------------|--------------------|
| **Before (baseline)** | PASS | PASS (FAIR) | PASS (GOOD_DEAL) |
| **After injection** | FAIL (G280 primary) | FAIL (SUSPICIOUS) | FAIL (IMPOSSIBLE) |

## Conclusion

Benchmarks **can fail** when defects are injected — pass criteria are not tautological.
