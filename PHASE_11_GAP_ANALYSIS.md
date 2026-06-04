# Phase 11 — Aircraft Consultant Authority Completion  
## Gap Analysis (Audit Only — No Implementation)

**Date:** 2026-05-29  
**Scope:** Why COMPARISON, BUY_DECISION, ALTERNATIVE, and MISSION paths still produce generic or mission-kernel fallback prose instead of broker-grade consultant output.  
**Constraint:** Phases 5–10 are stable and out of scope for modification in this audit.

---

## Executive Summary

Phase 10 fixed **entity contamination** (stale tail / Phly bleed). Remaining quality failures are primarily **authority routing gaps**, not retrieval leakage:

| Symptom | Primary root cause |
|--------|---------------------|
| Comparison → "OPERATIONAL SYNTHESIS" / "VIABLE WITH COMPROMISES" | Unified **comparison responder never wins** in default production; legacy path runs **mission ranking pipeline** instead of Comparison v2 |
| Buy decision → generic commentary | **No dedicated buy-decision execution path**; QRI misclassifies as `payload_range_analysis`; deal killer / structured verdict require listing rows that are often absent |
| Alternatives → weak / mixed categories | **Model detection failure** blocks unified ALTERNATIVE path; legacy LLM free-forms without tier-peer hierarchy |
| Mission → no clarifying questions | Clarification logic exists but is **soft** (LLM prompt only) and often bypassed when pipeline still runs |
| Missing GOOD FIT / CONDITIONAL FIT / NOT A FIT | Verdict is **prompt-only** on legacy LLM path; mission kernel uses **different verdict vocabulary** (`VIABLE WITH COMPROMISES`) |

**Production default:** `UNIFIED_INTENT_ENFORCE_*` flags are **off** and `UNIFIED_INTENT_ROLLOUT_PERCENT` defaults to **0**. Unified handlers (comparison, alternative, fact, capability) are **shadow/observe-only** unless explicitly enabled. Most user traffic executes the **legacy consultant LLM pipeline** in `rag/consultant_retrieval.py`.

---

## 1. Fallback Template Origins

| User-visible phrase | Source module | Function | Trigger |
|---------------------|---------------|----------|---------|
| `Some details below are directional rather than catalog-verified…` | `services/broker/graceful_degradation.py` | `degraded_low_confidence_prefix()` → `apply_graceful_degradation_to_answer()` | `confidence < 0.55` on degraded/fallback answers |
| `OPERATIONAL SYNTHESIS (AUTHORITATIVE)` | `services/mission/mission_authority_kernel.py` | `KERNEL_BLOCK_MARKER`; `render_kernel_synthesis()` | Mission authority kernel built during pre-LLM / orchestration pipeline |
| `[BROKER ADVISORY — OPERATIONAL SYNTHESIS FIRST]` | `services/consultant/llm_explanation_layer.py` | `build_pipeline_authority_block()` | Ranked pipeline returns **zero feasible** recs but `mission_understanding_authority` exists |
| `* VIABLE WITH COMPROMISES:` | `services/mission/mission_authority_kernel.py` | `render_kernel_verdict()` | Kernel advisory render when no PRIMARY RECOMMENDATION |
| `GOOD FIT` / `CONDITIONAL FIT` / `NOT A FIT` | `rag/query_service.py` (`CONSULTANT_SYSTEM_PROMPT`); `services/consultant/broker_response_renderer.py` | Prompt instruction; deterministic renderer | **Legacy path:** LLM may ignore. **Orchestrator path:** renderer can append |
| `GOOD DEAL` / `OVERPRICED` / `RISKY` | `services/deal_killer_engine.py` | `run_deal_killer_engine()` | Only when deal killer runs **and** injects into system prompt via `format_deal_killer_for_system_prompt()` |

**Important:** `rag/consultant_retrieval.py` does **not** call `response_formatter.py`, `broker_response_renderer.py`, or `apply_graceful_degradation_to_answer()` on the main LLM answer path. Fallback templates reach users via **pre-LLM authority blocks** fed to the LLM, or via separate orchestrator entry points—not via post-LLM formatting on the primary consultant route.

---

## 2. Production Execution Topology

```
User query
  → rag/consultant_retrieval.run_consultant_retrieval_bundle()
      → Phase 10 entity scope / intent persistence
      → classify_query_recommendation_intent()          [QRI — legacy advisory classifier]
      → classify_unified_intent()                       [Unified router — Phase 5]
      → evaluate_rollout()                              [Phase 7 — default 0%]
      → evaluate_pipeline_gate() + execute_unified_handler()
            IF enforce_* AND rollout.enabled:
                EARLY RETURN (comparison / alternative / fact / capability responders)
            ELSE:
                FALL THROUGH → legacy pipeline below
      → should_run_pre_llm_pipeline() → run_pre_llm_recommendation()
            → run_recommendation_pipeline()             [Mission ranker + kernel]
            → build_pipeline_authority_block()          [May inject OPERATIONAL SYNTHESIS]
      → Phly / Tavily / Pinecone / market SQL
      → deal_killer (optional, listing-dependent)
      → CONSULTANT_SYSTEM_PROMPT + context → LLM answer
```

**Gate condition for unified early return** (`consultant_retrieval.py` ~747–823):

```python
_unified_route is not None
AND _rollout_enabled                    # UNIFIED_INTENT_ROLLOUT_PERCENT > 0 + session bucket
AND (UNIFIED_INTENT_ENFORCE_FACT | CAPABILITY | COMPARISON | ALTERNATIVE)
AND _gate.enforce                       # path-specific enforce flag
```

With default env, **all comparison/alternative queries fall through** to mission pipeline + LLM even when `execution_path=COMPARISON` in shadow telemetry.

---

## 3. Path-by-Path Execution Traces

Traces captured via live classification (`classify_unified_intent`, `classify_query_recommendation_intent`, `should_run_pre_llm_pipeline`) in the current codebase.

### 3.1 COMPARISON

#### Example: `Compare G650 vs Falcon 8X`

| Stage | Result |
|-------|--------|
| Unified intent | `OTHER` |
| Unified `execution_path` | `COMPARISON` |
| Secondary intent | `aircraft_comparison_likely` |
| QRI | `aircraft_comparison` (`requires_ranked_pipeline=True`) |
| `should_run_pre_llm_pipeline` | **True** |
| Unified handler invoked? | **No** (enforce off / rollout 0%) |
| Responder that actually runs | `run_recommendation_pipeline()` → mission ranker → `build_pipeline_authority_block()` → **LLM** |
| Intended responder (when enforced) | `services/comparison/comparison_pipeline_v2_responder.respond_aircraft_comparison()` |

**Exact chain (default production):**

1. `consultant_retrieval.py` → unified shadow logs `execution_path=comparison`
2. Gate `enforce=False` → no early return
3. `pre_llm_recommendation.py` → QRI=`aircraft_comparison` → `run_recommendation_pipeline(..., query_intent=aircraft_comparison)`
4. Pipeline builds `mission_authority_kernel` with `OPERATIONAL SYNTHESIS (AUTHORITATIVE)` block
5. `llm_explanation_layer.build_pipeline_authority_block()` injects synthesis-first instructions
6. LLM narrates kernel + generic comparison → user sees "OPERATIONAL SYNTHESIS" / "VIABLE WITH COMPROMISES"

#### Example: `Longitude vs Praetor 600`

Same as above. Both models detected; unified path=`COMPARISON`, legacy runs mission ranker.

#### Example: `CJ3+ vs Phenom 300`

| Stage | Result |
|-------|--------|
| Unified `execution_path` | `COMPARISON` (via `is_explicit_comparison_query`) |
| Unified `model` | **None** (detection gap on `CJ3+`) |
| QRI | `aircraft_comparison` |
| Enforced comparison responder | Would call `respond_aircraft_comparison()` → likely **INSUFFICIENT_DATA** if models not locked |
| Legacy path | Still runs mission pipeline with partial model detection |

#### Example: `G280 vs Challenger 3500`

| Stage | Result |
|-------|--------|
| Unified `model` | `Challenger 350` (**alias collapse** — 3500 → 350) |
| Risk | Wrong-aircraft comparison in both unified and legacy paths |

**Why comparison responder does not win**

1. **Enforcement disabled by default** — correct router classification is observe-only.
2. **QRI maps comparison to ranked pipeline** — `_RANKED_PIPELINE_INTENTS` includes `AIRCRAFT_COMPARISON`, so legacy intentionally runs mission ranker instead of Comparison v2.
3. **Architectural mismatch** — Comparison v2 (`comparison_pipeline_v2_responder.py`) is wired only to unified gate; legacy uses `run_recommendation_pipeline()`.
4. **Mission kernel vocabulary** — `render_kernel_verdict()` emits `VIABLE WITH COMPROMISES`, not side-by-side spec contrast.

---

### 3.2 BUY_DECISION

#### Examples: `2016 Latitude $10M good deal?`, `2018 Praetor 600 $17M good deal?`, `2015 CJ3+ $7M good deal?`

| Stage | Result |
|-------|--------|
| Unified intent | `OTHER` |
| Unified `execution_path` | **NONE** |
| QRI | **`payload_range_analysis`** (confidence ~0.42 fallback) |
| `requires_ranked_pipeline` | **False** |
| `should_run_pre_llm_pipeline` | **False** |
| Deal killer | Runs only if `phly_rows` or `consultant_primary_listing_for_deal_review` present |
| `aircraft_decision_engine` | **Not triggered** — `consultant_query_requests_aircraft_decision()` does not match `"good deal"` |

**Exact chain (typical hypothetical listing question without DB row):**

1. QRI scores all intents low; fallback at `query_recommendation_intent.py:325–331` assigns `payload_range_analysis`
2. `build_intent_authority_note()` tells LLM "Do NOT produce purchase-style ranked shortlist"
3. No pre-LLM pipeline; no deal killer payload (no phly/listing row)
4. LLM answers from general knowledge + `CONSULTANT_SYSTEM_PROMPT` → **generic commentary**
5. Verdict lines (`GOOD DEAL`, red flags table) are prompt suggestions only — **no deterministic formatter** on this path

**Why buy-decision responder does not win**

1. **No unified `BUY_DECISION` execution path exists** (Phase 5 paths: FACT, MARKET_FACT, CAPABILITY, COMPARISON, ALTERNATIVE only).
2. **QRI has no `good deal` / `overpriced` / `fair price` scoring** — queries fall through to `payload_range_analysis`.
3. **`deal_killer_engine`** is the closest deterministic buy judge but requires merged phly/listing identity + ask/time; hypothetical "2016 Latitude $10M" without ingest returns `None`.
4. **`aircraft_decision_engine`** trigger phrases omit `"good deal"` (only `"worth buying"`, `"should i buy"`, etc.).
5. **`response_mode_router`** may tag deal-analysis mode, but that affects prompt suffix—not structured output schema.

---

### 3.3 ALTERNATIVE

#### Example: `Alternatives to Praetor 600`

| Stage | Result |
|-------|--------|
| Unified `execution_path` | **ALTERNATIVE** |
| `_resolve_alternative_target` | `Praetor 600` |
| QRI | **`payload_range_analysis`** (misclassified) |
| Enforced handler | `respond_aircraft_alternative()` → tier-peer list from `replacement_hierarchy.py` |
| Default production | **Fall through** → LLM without tier-peer authority |

#### Example: `Alternatives to Longitude` ⚠️ Critical gap

| Stage | Result |
|-------|--------|
| `is_alternative_execution_query` | **True** |
| `detect_models_from_text("Alternatives to Longitude")` | **`[]`** |
| `_resolve_alternative_target` | **None** |
| Unified `execution_path` | **NONE** (gate requires resolved target — `unified_intent_router.py:456–458`) |
| QRI | `payload_range_analysis` |
| Legacy | LLM free-form → category-mixed suggestions |

**Candidate generation (when unified alternative DOES run)**

| Step | Source |
|------|--------|
| Target resolution | `alternative_pipeline_responder._resolve_alternative_target()` → `replacement_hierarchy.extract_replacement_target()` |
| Peer list | `realistic_replacement_candidates()` → `_TIER_REPLACEMENT_POOL` by `aircraft_position_tier()` |
| Filtering | `is_prestige_collapse()`, `violates_class_sanity()`, `tier_distance() <= 2` |
| Output cap | Top 4 peers, 3-sentence guard in `_guard_answer()` |

**Why alternatives are weak in production**

1. Enforce off → tier-peer responder never runs for most sessions.
2. **"Alternatives to {model}" fails** when `detect_models_from_text` misses shorthand (`Longitude`, `G550`, `CJ3+`).
3. QRI mislabels alternatives as `payload_range_analysis` → no pipeline authority; LLM improvises.
4. Legacy path has **no replacement_hierarchy injection** into LLM context (only unified handler embeds peers).

---

### 3.4 MISSION

#### Example: `What aircraft should I buy for LA to Miami?`

| Stage | Result |
|-------|--------|
| Unified `execution_path` | NONE (`aircraft_mission_likely` secondary) |
| QRI | `acquisition_recommendation` |
| Pre-LLM pipeline | **Runs** full ranker |
| Clarification | Depends on `mission_validation.validate_mission_state_consistency()` |

#### Example: `8 passengers LA to Miami nonstop under $10M`

| Stage | Result |
|-------|--------|
| QRI | `mission_feasibility` |
| Pre-LLM pipeline | **Runs** |
| Route in query | Present → `needs_route_clarification` typically **false** |

#### Example: Open-ended `What jet should I buy?` (no pax/route/budget)

| Stage | Result |
|-------|--------|
| QRI | Often `shortlist_ranking` or `acquisition_recommendation` |
| `needs_route_clarification` | Set when `query_requires_route_for_advisory()` + `route_truly_missing()` (`mission_validation.py:210–225`) |
| Enforcement | `build_pipeline_authority_block()` returns **CLARIFICATION ONLY** block when validation flag set |
| Bypass risk | If QRI confidence low → `payload_range_analysis` → **no pipeline** → LLM may recommend models without asking |

**Why mission clarifications are inconsistent**

1. Clarification is **conditional** on `query_requires_route_for_advisory()` — not all mission asks trigger it.
2. **`clarification_questions_asked` budget** — after one ask, pipeline proceeds with incomplete mission.
3. **LLM system prompt** also says "ask 1–2 questions" but conflicts with pre-LLM pipeline that may already inject a shortlist.
4. No unified MISSION execution path — mission logic split across QRI, `run_pipeline`, kernel, and LLM narration.

---

## 4. Gap Report by Path

### COMPARISON

| | |
|--|--|
| **Current behavior** | Shadow classifies `execution_path=COMPARISON`; legacy runs mission ranker + kernel synthesis; LLM emits operational bands and `VIABLE WITH COMPROMISES` |
| **Expected behavior** | Deterministic side-by-side spec contrast (`comparison_pipeline_v2_responder`), no mission shortlist, no kernel verdict |
| **Root cause** | (1) Unified enforce/rollout off. (2) QRI includes `AIRCRAFT_COMPARISON` in `_RANKED_PIPELINE_INTENTS`. (3) No legacy dispatch to Comparison v2. (4) Kernel verdict vocabulary overrides HyeAero fit labels |
| **Recommended fix** | Phase 11: **Authority dispatch layer** (not new router): when `execution_path=COMPARISON` OR QRI=`aircraft_comparison` with ≥2 models, call `respond_aircraft_comparison()` **before** LLM regardless of enforce flags; remove comparison from ranked pipeline intents; post-process or block kernel injection for comparison turns; fix CJ3+/3500 alias locks |

**Key files:** `rag/consultant_retrieval.py`, `services/consultant/pre_llm_recommendation.py`, `services/recommendation/query_recommendation_intent.py`, `services/comparison/comparison_pipeline_v2_responder.py`, `services/mission/mission_authority_kernel.py`, `services/consultant/llm_explanation_layer.py`

---

### BUY_DECISION

| | |
|--|--|
| **Current behavior** | QRI=`payload_range_analysis`; no pipeline; deal killer silent without listing row; LLM generic prose |
| **Expected behavior** | Structured block: Aircraft / Year / Ask → Market Reality → Red Flags → Verdict (`GOOD DEAL` / `FAIR DEAL` / `OVERPRICED` / `RISKY`) |
| **Root cause** | (1) No buy-decision execution path. (2) QRI lacks deal/valuation intent. (3) `deal_killer_engine` gated on phly/listing presence. (4) `aircraft_decision_engine` misspells trigger (`good deal` not matched). (5) No deterministic response formatter on consultant_retrieval LLM path |
| **Recommended fix** | Add QRI intent `LISTING_VALUATION` or `BUY_DECISION`; route to `deal_killer_engine` with **synthetic market slice** when no tail row (model+year+ask from query parse); add **`respond_buy_decision()`** deterministic formatter; extend `consultant_query_requests_aircraft_decision()` / deal killer to parse "2016 Latitude $10M good deal" |

**Key files:** `services/recommendation/query_recommendation_intent.py`, `services/deal_killer_engine.py`, `services/aircraft_decision_engine.py`, `rag/consultant_retrieval.py`, `services/response_mode_router/triggers.py`

---

### ALTERNATIVE

| | |
|--|--|
| **Current behavior** | Unified ALTERNATIVE only when target resolves; "Alternatives to Longitude" → path NONE; legacy LLM mixed categories |
| **Expected behavior** | Tier-peer replacements from `replacement_hierarchy` (e.g., Longitude → Praetor 600, Challenger 650, G280) |
| **Root cause** | (1) Enforce off. (2) `_resolve_alternative_target` depends on `detect_models_from_text` — fails on common shorthand. (3) Unified gate requires target before assigning ALTERNATIVE path. (4) QRI fallback to `payload_range_analysis` |
| **Recommended fix** | Expand model detection for alternative targets (`Longitude`, `G550`, `CJ3+`); assign `execution_path=ALTERNATIVE` when `is_alternative_execution_query` even if target extracted via catalog alias resolver on trailing token; legacy dispatch to `respond_aircraft_alternative()`; add QRI intent `AIRCRAFT_ALTERNATIVES` |

**Key files:** `services/comparison/alternative_pipeline_responder.py`, `services/routing/unified_intent_router.py` (`_resolve_execution_path`), `services/recommendation/replacement_hierarchy.py`, `services/consultant/recommendation_engine.py` (`detect_models_from_text`)

---

### MISSION

| | |
|--|--|
| **Current behavior** | Pre-LLM pipeline runs when QRI ∈ ranked intents; may inject shortlist before mission complete; clarification only on route-missing heuristic |
| **Expected behavior** | Ask pax / route / budget / nonstop before naming aircraft on open-ended acquisition asks |
| **Root cause** | (1) QRI routes incomplete missions to `acquisition_recommendation` / `shortlist_ranking`. (2) Clarification flag easily bypassed. (3) LLM prompt and pipeline authority conflict. (4) No hard gate blocking shortlist when mandatory fields missing |
| **Recommended fix** | **Mission completeness gate** before `run_recommendation_pipeline()`: if acquisition intent and missing (pax OR route OR budget), return clarifying question response without LLM shortlist; unify `needs_route_clarification` with pax/budget checks; deterministic clarifying template |

**Key files:** `services/state/mission_validation.py`, `services/consultant/pre_llm_recommendation.py`, `services/recommendation/clarification_decision.py`, `services/consultant/llm_explanation_layer.py`

---

## 5. HyeAero Verdict Enforcement Gap

| Verdict type | Where defined | Enforced on legacy consultant path? |
|--------------|---------------|-------------------------------------|
| `✅ GOOD FIT` / `⚠️ CONDITIONAL FIT` / `❌ NOT A FIT` | `rag/query_service.py` CONSULTANT_SYSTEM_PROMPT | **Prompt only** — LLM may omit |
| `GOOD DEAL` / `OVERPRICED` / `RISKY` | deal_killer + prompt | **Only if deal_killer runs** |
| `VIABLE WITH COMPROMISES` | mission_authority_kernel | **Injected via pre-LLM authority** — often dominates comparison/mission turns |
| `PRIMARY RECOMMENDATION` / broker verdict enum | `services/broker/broker_verdicts.py` | Used in orchestrator/renderer paths, **not** consultant_retrieval LLM |

**Root cause:** Consultant retrieval ends in **raw LLM generation** without `broker_response_renderer.py` or `response_formatter.py` post-processing. Kernel synthesis uses a **different verdict taxonomy** than HyeAero client rules.

**Recommended fix:** Post-LLM verdict validator for advisory turns OR deterministic responder-first for comparison/buy/alternative/mission-complete paths; align kernel `render_kernel_verdict()` vocabulary with HyeAero fit labels when kernel is used.

---

## 6. Module Reference Map

| Concern | Primary modules |
|---------|-----------------|
| Entry / routing fork | `rag/consultant_retrieval.py` |
| Unified classification | `services/routing/unified_intent_router.py` |
| Unified gate / handlers | `services/routing/unified_pipeline_gate.py` |
| Rollout gating | `services/routing/unified_rollout_controller.py` |
| Legacy intent | `services/recommendation/query_recommendation_intent.py` |
| Pre-LLM pipeline | `services/consultant/pre_llm_recommendation.py`, `services/recommendation/recommendation_pipeline.py` |
| Mission kernel / synthesis | `services/mission/mission_authority_kernel.py`, `services/consultant/llm_explanation_layer.py` |
| Comparison v2 | `services/comparison/comparison_pipeline_v2_responder.py` |
| Alternatives | `services/comparison/alternative_pipeline_responder.py`, `services/recommendation/replacement_hierarchy.py` |
| Buy / deal | `services/deal_killer_engine.py`, `services/aircraft_decision_engine.py` |
| Degraded prefix | `services/broker/graceful_degradation.py` |
| LLM prompt / verdict rules | `rag/query_service.py` |
| Mission clarification | `services/state/mission_validation.py` |

---

## 7. Recommended Phase 11 Fix Plan (Implementation Deferred)

Priority order — **authority dispatch**, not new routing phases:

### P0 — Authority dispatch (legacy path)

1. **Comparison:** If QRI=`aircraft_comparison` OR unified `execution_path=COMPARISON`, invoke `respond_aircraft_comparison()` and **return** (skip mission pipeline + LLM) unless insufficient data.
2. **Alternative:** If `is_alternative_execution_query`, resolve target with enhanced alias parser; invoke `respond_aircraft_alternative()` and return.
3. **Buy decision:** New QRI intent + `respond_buy_decision()` using deal killer + market SQL; parse year/model/ask from query text.

### P1 — Classification fixes

4. Add QRI scoring for `good deal`, `overpriced`, `fair price`, `worth it at $X`.
5. Remove `AIRCRAFT_COMPARISON` from `_RANKED_PIPELINE_INTENTS` OR branch inside `run_pre_llm_recommendation` to Comparison v2.
6. Fix model detection for `Longitude`, `CJ3+`, `Challenger 3500` in alternative/comparison locks.

### P2 — Mission completeness

7. Hard gate: no `run_recommendation_pipeline` when acquisition intent and missing mandatory mission fields.
8. Single clarifying question template (pax, route, budget, nonstop).

### P3 — Verdict enforcement

9. Post-LLM verdict linter or migrate advisory turns to deterministic formatters.
10. Map kernel `VIABLE WITH COMPROMISES` → `CONDITIONAL FIT` when kernel must remain for multi-leg missions.

### P4 — Enable unified enforce (ops)

11. After P0 dispatch proves parity, enable `UNIFIED_INTENT_ENFORCE_COMPARISON` / `ALTERNATIVE` with rollout percent — redundant if P0 legacy dispatch is complete.

---

## 8. Success Criteria Mapping

| Test sequence | Current expected failure mode | Phase 11 target |
|---------------|------------------------------|-----------------|
| N988NW → Praetor → G650 → Longitude → CJ3+ | ✅ Phase 10 — no listing bleed | Maintain |
| G650 vs Falcon 8X | Mission synthesis + LLM | Verified catalog comparison |
| 2016 Latitude $10M good deal | Generic LLM | Structured deal killer verdict |
| Alternatives to Longitude | LLM mixed / path NONE | Tier-peer list (Praetor 600, Challenger 650, G280) |
| Open buy without mission | May shortlist anyway | Clarifying question first |
| All advisory turns | Missing fit verdict | Exactly one GOOD FIT / CONDITIONAL FIT / NOT A FIT |

---

## 9. Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| Legacy dispatch to comparison/alternative responders | Low — responders already tested in `test_comparison_alternative_execution.py` | Feature flag `CONSULTANT_AUTHORITY_DISPATCH=1` |
| Removing comparison from ranked pipeline | Medium — may affect eval runners expecting pipeline trace | Golden eval split: comparison cases assert v2 output |
| Buy decision without listing row | Medium — market comps may be thin | Graceful "FAIR DEAL — limited comp data" template |
| Mission hard gate | Medium — power users with implicit context | Allow inherit when conversation memory has pax+route+budget |
| Verdict post-processor | Low | Only append if missing; never duplicate |

---

*End of Phase 11 Gap Analysis — audit only, no code changes applied.*
