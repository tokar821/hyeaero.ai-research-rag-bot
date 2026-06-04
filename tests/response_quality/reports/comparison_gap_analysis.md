# Comparison Gap Analysis (Phase 34)

Review set comparisons: 20 queries (`cmp-001` ? `cmp-020`).

## Findings

- **No deterministic verdict clause** in all 20 comparison answers (audit requires ?Choose X if?, otherwise Y? or `INSUFFICIENT_DATA`).

## Evidence (per-query)

| Query ID | Query | Dispatch kind | Authority models | Findings | Preview |
|---|---|---|---|---|---|
| `cmp-001` | G650 vs Falcon 8X | comparison | Gulfstream G650, Falcon 8X | COMPARISON_NO_VERDICT | Verified catalog comparison: - Gulfstream G650: large-cabin class; practical range 5720 nm; seats 14; operating cost ban |
| `cmp-002` | Compare G650 and Falcon 8X | comparison | Gulfstream G650, Falcon 8X | COMPARISON_NO_VERDICT | Verified catalog comparison: - Gulfstream G650: large-cabin class; practical range 5720 nm; seats 14; operating cost ban |
| `cmp-003` | How does G650 compare to Falcon 8X? | comparison | Gulfstream G650, Falcon 8X | COMPARISON_NO_VERDICT | Verified catalog comparison: - Gulfstream G650: large-cabin class; practical range 5720 nm; seats 14; operating cost ban |
| `cmp-004` | G650 versus Falcon 8X for charter operations | comparison | Gulfstream G650, Falcon 8X | COMPARISON_NO_VERDICT | Verified catalog comparison: - Gulfstream G650: large-cabin class; practical range 5720 nm; seats 14; operating cost ban |
| `cmp-005` | Which is better, G650 or Falcon 8X? | comparison | Gulfstream G650, Falcon 8X | COMPARISON_NO_VERDICT | Verified catalog comparison: - Gulfstream G650: large-cabin class; practical range 5720 nm; seats 14; operating cost ban |
| `cmp-006` | G650 vs Global 7500 | comparison | Gulfstream G650, Global 7500 | COMPARISON_NO_VERDICT | Verified catalog comparison: - Gulfstream G650: large-cabin class; practical range 5720 nm; seats 14; operating cost ban |
| `cmp-007` | Compare G650 and Global 7500 | comparison | Gulfstream G650, Global 7500 | COMPARISON_NO_VERDICT | Verified catalog comparison: - Gulfstream G650: large-cabin class; practical range 5720 nm; seats 14; operating cost ban |
| `cmp-008` | How does G650 compare to Global 7500? | comparison | Gulfstream G650, Global 7500 | COMPARISON_NO_VERDICT | Verified catalog comparison: - Gulfstream G650: large-cabin class; practical range 5720 nm; seats 14; operating cost ban |
| `cmp-009` | G650 versus Global 7500 for charter operations | comparison | Gulfstream G650, Global 7500 | COMPARISON_NO_VERDICT | Verified catalog comparison: - Gulfstream G650: large-cabin class; practical range 5720 nm; seats 14; operating cost ban |
| `cmp-010` | Which is better, G650 or Global 7500? | comparison | Gulfstream G650, Global 7500 | COMPARISON_NO_VERDICT | Verified catalog comparison: - Gulfstream G650: large-cabin class; practical range 5720 nm; seats 14; operating cost ban |
| `cmp-011` | G650 vs G700 | comparison | Gulfstream G650 | BROKER_BAD_AIRCRAFT, COMPARISON_INCOMPLETE, COMPARISON_NO_VERDICT | Insufficient verified data for deterministic execution.  Verified catalog comparison requires two recognized aircraft mo |
| `cmp-012` | Compare G650 and G700 | comparison | Gulfstream G650 | BROKER_BAD_AIRCRAFT, COMPARISON_INCOMPLETE, COMPARISON_NO_VERDICT | Insufficient verified data for deterministic execution.  Verified catalog comparison requires two recognized aircraft mo |
| `cmp-013` | How does G650 compare to G700? | comparison | Gulfstream G650 | BROKER_BAD_AIRCRAFT, COMPARISON_INCOMPLETE, COMPARISON_NO_VERDICT | Insufficient verified data for deterministic execution.  Verified catalog comparison requires two recognized aircraft mo |
| `cmp-014` | G650 versus G700 for charter operations | comparison | Gulfstream G650 | BROKER_BAD_AIRCRAFT, COMPARISON_INCOMPLETE, COMPARISON_NO_VERDICT | Insufficient verified data for deterministic execution.  Verified catalog comparison requires two recognized aircraft mo |
| `cmp-015` | Which is better, G650 or G700? | comparison | Gulfstream G650 | BROKER_BAD_AIRCRAFT, COMPARISON_INCOMPLETE, COMPARISON_NO_VERDICT | Insufficient verified data for deterministic execution.  Verified catalog comparison requires two recognized aircraft mo |
| `cmp-016` | G650 vs Longitude | comparison | Gulfstream G650 | BROKER_BAD_AIRCRAFT, COMPARISON_INCOMPLETE, COMPARISON_NO_VERDICT | Insufficient verified data for deterministic execution.  Verified catalog comparison requires two recognized aircraft mo |
| `cmp-017` | Compare G650 and Longitude | comparison | Gulfstream G650 | BROKER_BAD_AIRCRAFT, COMPARISON_INCOMPLETE, COMPARISON_NO_VERDICT | Insufficient verified data for deterministic execution.  Verified catalog comparison requires two recognized aircraft mo |
| `cmp-018` | How does G650 compare to Longitude? | comparison | Gulfstream G650 | BROKER_BAD_AIRCRAFT, COMPARISON_INCOMPLETE, COMPARISON_NO_VERDICT | Insufficient verified data for deterministic execution.  Verified catalog comparison requires two recognized aircraft mo |
| `cmp-019` | G650 versus Longitude for charter operations | comparison | Gulfstream G650 | BROKER_BAD_AIRCRAFT, COMPARISON_INCOMPLETE, COMPARISON_NO_VERDICT | Insufficient verified data for deterministic execution.  Verified catalog comparison requires two recognized aircraft mo |
| `cmp-020` | Which is better, G650 or Longitude? | comparison | Gulfstream G650 | BROKER_BAD_AIRCRAFT, COMPARISON_INCOMPLETE, COMPARISON_NO_VERDICT | Insufficient verified data for deterministic execution.  Verified catalog comparison requires two recognized aircraft mo |

## Source

- `services/comparison/comparison_pipeline_v2_responder.py::_format_structured_contrast()` does not emit verdict text.