# Aviation Mission Benchmark

Production evaluation framework for HyeAero consultant mission intelligence.

## Dataset

`aviation_mission_suite.json` — 24 real buyer/advisor cases across:

- Transatlantic missions
- Asia nonstop
- Fractional vs ownership
- Runway flexibility
- Operating cost
- Mountain airports
- Cabin comparison
- Range realism
- Westbound winter constraints
- Turn isolation (memory leak detection)

Each case includes **golden expectations** (`expected_any_models`, `forbidden_any_models`, routes, constraints).

## Scoring dimensions (0–1 each)

| Dimension | Measures |
|-----------|----------|
| `mission_understanding` | Category, pax, ownership, constraints |
| `entity_accuracy` | Places / regions in profile |
| `route_accuracy` | Correct routes, no contamination |
| `aircraft_realism` | Ranker vs forbidden / expected models |
| `operational_reasoning` | Tradeoffs, practical range language |
| `hallucination_rate` | Fake model denial (1 = clean) |
| `contamination_rate` | Template bleed / turn leak (1 = clean) |

## Automated failures

A case **fails** when any of these fire:

- `previous_turn_leak` — prior route/pax in current turn profile
- `invalid_routes` — UI garbage or forbidden patterns
- `impossible_aircraft_recommended` — ULR on regional hop, etc.
- `ignores_mission_constraints` — missing nonstop, pax, runway, etc.

## Reports

`generate_benchmark_report()` produces:

- **contamination_report** — turn leaks, invalid routes, template hits
- **realism_score** — mean aircraft realism
- **aircraft_diversity_score** — top-model dominance
- **recommendation_precision** — golden expected model hit rate

## Run

```bash
cd backend
python scripts/run_aviation_mission_benchmark.py
python scripts/run_aviation_mission_benchmark.py --category runway_flexibility --fail-fast
python scripts/run_aviation_mission_benchmark.py --out evals/results/run_$(date +%Y%m%d).json
```

Default mode `intelligence` runs **without API**: turn extraction, memory merge, mission ranker, structured formatter.

Use `--mode full` with a wired `get_response` callable for end-to-end LLM evaluation.

## Aviation QA / auto-improvement (v2)

`evals/aviation_qa/` adds an **evaluator agent** that critiques responses (does not answer users).

### Evaluator JSON (per case)

```json
{
  "route_realism": "PASS",
  "aircraft_realism": "FAIL",
  "hallucination_risk": 0.12,
  "repetition_score": 0.35,
  "humanness_score": 0.62,
  "operational_realism": 0.58,
  "tone_broker_score": 0.71,
  "fake_confidence_risk": 0.18,
  "brochure_language_risk": 0.22,
  "missing_tradeoffs": false,
  "main_failure": "Unrealistic aircraft recommended for mission: Praetor 600",
  "trust_score": 0.41,
  "passed": false
}
```

### What it detects

- Unrealistic / forbidden aircraft in shortlist
- Robotic phrasing & cross-suite repetition (`starts the conversation`, `on my list`, …)
- Diagnostic-console tone (`Mission Summary`, `Best Fit Aircraft`)
- Brochure / fake-certainty language
- Weak operational reasoning & missing tradeoffs
- Mission validation bypass (fallback ranking on impossible missions)

### Run QA loop

```bash
cd backend
python scripts/run_aviation_qa.py
python scripts/run_aviation_qa.py --category westbound_winter_constraints
python scripts/run_aviation_qa.py --case asia_003 --out evals/results/qa.json
```

Output includes `improvement_plan` with **failure source** and **targeted fix suggestions** (routing, hard elimination, phrase guard, formatter).

Suite-level `qa_defaults.forbidden_phrases` and per-case `qa` blocks in `aviation_mission_suite.json`.

## Tests

```bash
pytest tests/test_aviation_benchmark.py tests/test_aviation_qa.py -q
```

## Example golden case

**Input:** Miami Caribbean missions with runway flexibility  

**Expected:** PC-12, CJ3+, King Air 350 class — **NOT** G650  

Encoded in `runway_001` / `runway_003` with `forbidden_any_models: ["Gulfstream G650", "Falcon 8X", "Global 7500"]`.
