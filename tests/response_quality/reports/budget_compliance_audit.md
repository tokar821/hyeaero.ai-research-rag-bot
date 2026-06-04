# Budget Compliance Audit (BROKER_BUDGET_MISMATCH) ? Phase 34

Total flagged: **11**

| Query ID | Query | Budget (parsed) | Preview |
|---|---|---:|---|
| `msn-017` | Need 8 passengers TEB to LAX nonstop under $10M | 10M | [BROKER ADVISORY CONTEXT — narrate only; do not invent aircraft or feasibility]  YOUR ROLE: Top-tier aircraft acquisition consultant. Concis |
| `buy-001` | Is a 2015 Citation Latitude for $5M a good deal? | 5M | Aircraft: Citation Latitude for Year: 2015 Ask: $5.0M  Market Reality: - Limited synced comp data for this model slice — price verdict uses  |
| `buy-002` | 2015 Citation Latitude at $6M — fair price? | 6M | Aircraft: Citation Latitude at Year: 2015 Ask: $6.0M  Market Reality: - Limited synced comp data for this model slice — price verdict uses a |
| `buy-003` | Is this 2015 Citation Latitude for $8M overpriced? | 8M | Aircraft: Is this 2015 Citation Latitude for Ask: $8.0M  Market Reality: - Limited synced comp data for this model slice — price verdict use |
| `buy-004` | Should I buy a 2015 Citation Latitude listed at $10M? | 10M | Aircraft: Citation Latitude listed at Year: 2015 Ask: $10.0M  Market Reality: - Limited synced comp data for this model slice — price verdic |
| `buy-005` | 2015 Citation Latitude $12M good buy? | 12M | Aircraft: Citation Latitude Year: 2015 Ask: $12.0M  Market Reality: - Limited synced comp data for this model slice — price verdict uses ask |
| `buy-011` | Is a 2016 Citation Latitude for $5M a good deal? | 5M | Aircraft: Citation Latitude for Year: 2016 Ask: $5.0M  Market Reality: - Limited synced comp data for this model slice — price verdict uses  |
| `buy-012` | 2016 Citation Latitude at $6M — fair price? | 6M | Aircraft: Citation Latitude at Year: 2016 Ask: $6.0M  Market Reality: - Limited synced comp data for this model slice — price verdict uses a |
| `buy-013` | Is this 2016 Citation Latitude for $8M overpriced? | 8M | Aircraft: Is this 2016 Citation Latitude for Ask: $8.0M  Market Reality: - Limited synced comp data for this model slice — price verdict use |
| `buy-014` | Should I buy a 2016 Citation Latitude listed at $10M? | 10M | Aircraft: Citation Latitude listed at Year: 2016 Ask: $10.0M  Market Reality: - Limited synced comp data for this model slice — price verdic |
| `buy-015` | 2016 Citation Latitude $12M good buy? | 12M | Aircraft: Citation Latitude Year: 2016 Ask: $12.0M  Market Reality: - Limited synced comp data for this model slice — price verdict uses ask |

## Source modules to inspect

- `services/consultant/recommendation_engine.py` (budget gating in shortlist)
- `services/consultant/broker_advisory_layer.py` (what counts as ?feasible aircraft?)