"""Query routing services."""

from services.routing.unified_intent_execution import (
    should_enforce_alternative_path,
    should_enforce_capability_path,
    should_enforce_comparison_path,
    should_enforce_fact_path,
)
from services.routing.unified_intent_router import (
    PROMOTION_CONTRACT_VERSION,
    UnifiedExecutionPath,
    UnifiedIntent,
    UnifiedIntentRoute,
    UnifiedSecondaryIntent,
    build_unified_intent_shadow,
    classify_unified_intent,
    get_secondary_intent_promotion_contract,
    validate_unified_intent_route_invariants,
)
from services.routing.unified_pipeline_gate import (
    UnifiedPipelineGateDecision,
    evaluate_pipeline_gate,
    execute_unified_pipeline_handler,
)

__all__ = [
    "PROMOTION_CONTRACT_VERSION",
    "UnifiedExecutionPath",
    "UnifiedIntent",
    "UnifiedIntentRoute",
    "UnifiedPipelineGateDecision",
    "UnifiedSecondaryIntent",
    "build_unified_intent_shadow",
    "classify_unified_intent",
    "evaluate_pipeline_gate",
    "execute_unified_pipeline_handler",
    "get_secondary_intent_promotion_contract",
    "should_enforce_alternative_path",
    "should_enforce_capability_path",
    "should_enforce_comparison_path",
    "should_enforce_fact_path",
    "validate_unified_intent_route_invariants",
]
