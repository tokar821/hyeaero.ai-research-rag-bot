# Model Insertion Audit (UNJUSTIFIED_MODEL_INSERTION) ? Phase 34

Total flagged: **4**

## Call path (observed)

`rag.consultant_retrieval.run_consultant_retrieval_bundle()`
? `services.consultant.llm_explanation_layer.build_pipeline_authority_block()`
? `services.consultant.broker_advisory_layer.BrokerAdvisoryContext.to_llm_block()`
? prompt injected into system prompt
? **LLM failure/echo** ? prompt block returned as final `answer`

## Evidence

### `msn-002`

- **Query**: Need 8 passengers TEB to LAX nonstop
- **Authority models**: (none)
- **IntentLock captured**: EMPTY {}

```
[BROKER ADVISORY CONTEXT — narrate only; do not invent aircraft or feasibility]

YOUR ROLE: Top-tier aircraft acquisition consultant. Concise, factual, decisive.
You may be slightly critical. No marketing language. No generic AI phrasing.
You MUST NOT add, remove, or swap aircraft. You MUST NOT invent range/feasibility.

Mission: 8 passengers; nonstop required
Passengers: 8
Constraints: nonstop required

FEASIBLE AIR
```

### `msn-007`

- **Query**: Need 8 passengers TEB to LAX nonstop under $15M
- **Authority models**: (none)
- **IntentLock captured**: EMPTY {}

```
[BROKER ADVISORY CONTEXT — narrate only; do not invent aircraft or feasibility]

YOUR ROLE: Top-tier aircraft acquisition consultant. Concise, factual, decisive.
You may be slightly critical. No marketing language. No generic AI phrasing.
You MUST NOT add, remove, or swap aircraft. You MUST NOT invent range/feasibility.

Mission: 8 passengers; nonstop required
Passengers: 8
Constraints: nonstop required

FEASIBLE AIR
```

### `msn-012`

- **Query**: Need 8 passengers TEB to LAX nonstop under $25M
- **Authority models**: (none)
- **IntentLock captured**: EMPTY {}

```
[BROKER ADVISORY CONTEXT — narrate only; do not invent aircraft or feasibility]

YOUR ROLE: Top-tier aircraft acquisition consultant. Concise, factual, decisive.
You may be slightly critical. No marketing language. No generic AI phrasing.
You MUST NOT add, remove, or swap aircraft. You MUST NOT invent range/feasibility.

Mission: 8 passengers; nonstop required
Passengers: 8
Constraints: nonstop required

FEASIBLE AIR
```

### `msn-017`

- **Query**: Need 8 passengers TEB to LAX nonstop under $10M
- **Authority models**: (none)
- **IntentLock captured**: EMPTY {}

```
[BROKER ADVISORY CONTEXT — narrate only; do not invent aircraft or feasibility]

YOUR ROLE: Top-tier aircraft acquisition consultant. Concise, factual, decisive.
You may be slightly critical. No marketing language. No generic AI phrasing.
You MUST NOT add, remove, or swap aircraft. You MUST NOT invent range/feasibility.

Mission: 8 passengers; nonstop required
Passengers: 8
Constraints: nonstop required

FEASIBLE AIR
```
