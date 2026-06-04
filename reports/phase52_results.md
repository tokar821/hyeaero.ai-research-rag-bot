# Phase 52 Results

Generated: 2026-06-03

## Before / After

| Metric | Phase 51 baseline | Phase 52 after | Target |
|--------|-------------------|----------------|--------|
| Retrieval accuracy pass rate | 14.3% (1/7) | **100%** (7/7) | >80% |
| Recommendation accuracy pass rate | 40% (2/5) | **100%** (5/5) | >80% |
| Wrong authority rate (retrieval) | 85.7% | **0%** | — |
| Broker Certification V2 | 96.8% (184/190) | **96.8%** (185/191) | >95% |

## Code changed (decision engine only)

- `services/routing/authority_dispatch.py` — category discovery, buy/wait, tail investigation routing
- `services/broker_decision/mission_fit_scorer.py` — mission + budget ranking (new)
- `services/broker_decision/budget_matcher.py` — mission-fit scoring, super-midsize stretch band
- `services/broker_decision/decision_intent_detector.py` — category vs realisticity intent separation
- `services/executive_broker/recommendation_selector.py` — query-focus models, mission-ranked primary
- `services/broker_reasoning/category_resolver.py` — super-midsize class, Gulfstream entry tier
- `services/broker_reasoning/mission_interpreter.py` — coast-to-coast range inference
- `services/broker_reasoning/broker_reasoning_layer.py` — super-midsize guidance listing
- `services/comparison/alternative_pipeline_responder.py` — cheap gulfstream / best-jet patterns

No new presentation, humanizer, conversation, or executive formatting layers.

## Certification V2 (unchanged failures)

Six pre-existing failures remain (consistency threads + comparison shorthand):

- `cessna_14m_8turn`, `budget_12m_g650_probe`, `mission_coast_20m`, `regional_8m_thread`
- `cj4_vs_phenom300`, `longitude_vs_falcon2000`

Pass rate **185/191 = 96.8%** — above 95% target. No regression in budget reality, listing realism, tail, or comparison groups A–F.

## Remaining accuracy blockers

1. **Multi-turn consistency** — thread memory does not always keep G280 / Praetor / budget references across 8-turn scripts.
2. **Comparison model resolution** — CJ4/Phenom and Longitude/Falcon 2000 shorthand still miss in comparison prose.
3. **E2E retrieval path** — benchmarks often run `layers` path; full `e2e` retrieval may differ when bundle returns non-substantive answers.

## Regenerate

```bash
cd backend
PYTHONPATH=. pytest tests/e2e/retrieval_accuracy_suite.py tests/e2e/recommendation_accuracy_suite.py tests/e2e/test_broker_certification_v2.py -q
PYTHONPATH=. python runners/run_phase52_audit.py
```
