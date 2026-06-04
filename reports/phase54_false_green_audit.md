# Phase 54 — False Certification Detection Audit

Generated: 2026-06-03

Reviews whether benchmarks **fail** when quality fails.

---

## Summary

| Suite | Fails on bad primary? | Fails on bad listing? | Fails on missing verdict? | False-green risk |
|-------|----------------------|----------------------|---------------------------|------------------|
| `real_aircraft_benchmark.py` | **Yes** | N/A | N/A | Low (post-Phase 53) |
| `listing_validation_suite.py` | N/A | **Yes** | **Yes** | Low |
| `market_recommendation_audit.py` | **No assert** | N/A | N/A | **HIGH** |
| `production_query_replay_suite.py` | **No** | N/A | N/A | **HIGH** |
| `listing_realism_suite.py` | N/A | **No** | **No** | **HIGH** (stale) |
| `tail_investigation_suite.py` | N/A | N/A | Partial | Medium |
| `recommendation_accuracy_suite.py` | **Yes** (6 cases) | N/A | N/A | Low count |
| `retrieval_accuracy_suite.py` | Partial | N/A | N/A | Medium |

**Proof tests exist:** `tests/verification/test_failure_injection.py` monkeypatches selector and `infer_listing_verdict` and confirms benchmarks fail.

---

## 1. `real_aircraft_benchmark.py`

| Check | Status |
|-------|--------|
| `assert passed` on recorder | **Yes** (line 146) |
| Wrong primary fails | **Yes** — `primary_acc` from expected model in answer/executive |
| Missing verdict / infeasible | **Yes** — `expect_infeasible` branch |
| Path-only pass | **No** — path + passed |

**Caveats (residual false green):**

- `prefer_e2e=False` only — production e2e not certified.
- Relaxed pass for listing-style txn queries (`primary_acc` + budget only).
- Coast mission passes without alt when primary matches.
- Compatible inference not used (strict `_evaluate`).

---

## 2. `listing_validation_suite.py`

| Check | Status |
|-------|--------|
| `assert correct >= 1.0` | **Yes** |
| Wrong verdict fails | **Yes** — unless in `_compatible` pairs (FAIR↔OVERPRICED, etc.) |
| Missing deal_quality | **No** — tier fallback can pass without production `deal_quality` |

**Caveats:**

- `_compatible()` allows FAIR/OVERPRICED/SUSPICIOUS/GOOD_DEAL adjacency — intentional but reduces strictness.
- Inference in test code, not exported production function.

---

## 3. `market_recommendation_audit.py`

| Check | Status |
|-------|--------|
| Pytest assert on `bias_ok` | **MISSING** |
| Records `bias_ok` to report only | Bias failure visible in markdown, **pytest still green** |

**False green:** G280 over-selection on $18M+ budget scenarios does not fail CI.

---

## 4. `production_query_replay_suite.py`

| Check | Status |
|-------|--------|
| `passed = not authority_error` | Authority only |
| Wrong primary | **Does not fail** |
| Low trust | **Does not fail** |
| Drift | Fails if `recommendation_consistency_audit_v2` flags |

**False green:** 500/500 pass with avg trust 79.4 and many empty buy primaries.

---

## 5. `listing_realism_suite.py` (Phase 51 legacy)

| Check | Status |
|-------|--------|
| Assert | **`assert path in (e2e, layers)` only** |
| Records `passed` in recorder | **Not asserted in pytest** |

**False green:** Defective listing inference would still show pytest green.

---

## 6. `tail_investigation_suite.py`

| Check | Status |
|-------|--------|
| Registry / listing mode | **Asserted** |
| Answer mentions reg | Recorded, **not asserted** |
| Valuation authority | Asserted when expected |

---

## 7. `recommendation_accuracy_suite.py` / `retrieval_accuracy_suite.py`

Small scenario counts (6–7). **Assert passed** on quality but do not block release alone.

---

## 8. Failure injection verification

`tests/verification/test_failure_injection.py`:

- Injects wrong executive primary → `real_aircraft` **fails**
- Injects always SUSPICIOUS / IMPOSSIBLE inference → listing **fails**

**Recommendation:** Run this module in CI on every PR touching selector or inference.

---

## 9. Phase 53 summary document false green (historical)

`phase53_production_reality_summary.md` claimed 100% while reports showed 79%/25% because:

1. Summary was hand-edited.
2. Pytest did not assert recorder pass bit.

**Status:** Summary now auto-generated; benchmarks assert pass. **Residual risk:** other suites listed above.

---

## Certification integrity scorecard

| Requirement | Met? |
|-------------|------|
| Tests fail when recommendation quality fails | **Partial** — only real_aircraft + recommendation_accuracy |
| Tests fail when listing quality fails | **Yes** on listing_validation; **No** on listing_realism |
| Tests fail when primary wrong | **Partial** — not on replay/market audit |
| Tests fail when verdict missing | **No** — tier fallback masks missing `deal_quality` |
