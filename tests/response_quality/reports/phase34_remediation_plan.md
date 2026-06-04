# Phase 34 ? Remediation Plan (Proposed, Not Implemented)

Ranked proposals derived from Phase 33 findings. No code changes made in this phase.

| Severity | ID | Issue | Affected modules | Fix complexity | Regression risk |
|---|---|---|---|---|---|
| CRITICAL | CRIT-01 | Empty answer / prompt leakage on advisory path | rag/consultant_retrieval.py; services/consultant/* | Medium | Medium |
| HIGH | HIGH-01 | Comparison renderer lacks verdict | services/comparison/comparison_pipeline_v2_responder.py | Low | Low |
| HIGH | HIGH-02 | Comparison missing explicit cabin/cost delta | services/comparison/comparison_pipeline_v2_responder.py | Low?Medium | Low |
| MEDIUM | MED-01 | Budget mismatches in mission shortlist | services/consultant/recommendation_engine.py; broker_advisory_layer.py | Medium | Medium |
| LOW | LOW-01 | IntentLock provenance missing in mission records (drives insertion flags) | rag/consultant_retrieval.py | Low | Low |