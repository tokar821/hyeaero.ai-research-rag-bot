"""Phase 40 — broker reasoning expansion (interpretation layer only)."""

from services.broker_reasoning.broker_reasoning_layer import (
    apply_broker_reasoning_layer,
    get_broker_reasoning_buy_parse,
    get_broker_reasoning_compare_models,
    is_acquisition_budget_query,
    render_acquisition_guidance,
    append_multi_intent_overlays,
)

__all__ = [
    "apply_broker_reasoning_layer",
    "append_multi_intent_overlays",
    "get_broker_reasoning_buy_parse",
    "get_broker_reasoning_compare_models",
    "is_acquisition_budget_query",
    "render_acquisition_guidance",
]
