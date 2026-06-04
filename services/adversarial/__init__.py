"""Phase 38 — adversarial query normalization and conflict detection."""

from services.adversarial.adversarial_preprocessor import (
    CleanNormalizedQuery,
    check_comparison_safety,
    get_pipeline_query,
    preprocess_adversarial_query,
    to_unified_adversarial_metadata,
    try_adversarial_buy_block,
)
from services.adversarial.budget_conflict_normalizer import PriceSignalKind, classify_price_signals
from services.adversarial.budget_conflict_normalizer import BudgetConflictState, normalize_budget_conflicts
from services.adversarial.intent_sanitizer import sanitize_intents
from services.adversarial.model_adversary_resolver import AdversaryResolvedModel, resolve_adversary_models
from services.adversarial.query_conflict_detector import (
    ConflictSeverity,
    ConflictType,
    QueryConflictReport,
    detect_query_conflicts,
)

__all__ = [
    "AdversaryResolvedModel",
    "BudgetConflictState",
    "CleanNormalizedQuery",
    "ConflictSeverity",
    "ConflictType",
    "QueryConflictReport",
    "check_comparison_safety",
    "detect_query_conflicts",
    "get_pipeline_query",
    "normalize_budget_conflicts",
    "PriceSignalKind",
    "classify_price_signals",
    "preprocess_adversarial_query",
    "resolve_adversary_models",
    "sanitize_intents",
    "to_unified_adversarial_metadata",
    "try_adversarial_buy_block",
]
