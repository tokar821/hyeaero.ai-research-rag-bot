"""Phase 36 — cross-pipeline identity and market consistency."""

from services.consistency.consistency_injection_layer import (
    inject_consistency,
    prepare_buy_decision_state,
    prepare_comparison_consistency,
    prepare_valuation_state,
    render_buy_decision_answer,
    render_valuation_answer,
)
from services.consistency.cross_model_identity import (
    CanonicalAircraftIdentity,
    resolve_canonical_identity,
    resolve_comparison_identities,
)
from services.consistency.pipeline_agreement_checker import (
    AgreementFlag,
    PipelineAgreementReport,
    check_pipeline_agreement,
)
from services.consistency.unified_broker_state import UnifiedBrokerState

__all__ = [
    "AgreementFlag",
    "CanonicalAircraftIdentity",
    "PipelineAgreementReport",
    "UnifiedBrokerState",
    "check_pipeline_agreement",
    "inject_consistency",
    "prepare_buy_decision_state",
    "prepare_comparison_consistency",
    "prepare_valuation_state",
    "render_buy_decision_answer",
    "render_valuation_answer",
    "resolve_canonical_identity",
    "resolve_comparison_identities",
]
