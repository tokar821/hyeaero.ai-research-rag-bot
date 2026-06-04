"""Phase 47 — comparison presentation guard tests."""

from __future__ import annotations

from services.conversation.broker_conversation_layer import apply_broker_conversation_layer


def test_comparison_guard_rewrites_insufficient_when_models_known():
    du = {
        "broker_reasoning": {"compare_models": ["Gulfstream G650", "Gulfstream G700"]},
        "authority_dispatch_kind": "comparison",
    }
    raw = (
        "Insufficient verified data for deterministic execution.\n\n"
        "Verified catalog comparison requires two recognized aircraft models."
    )
    out = apply_broker_conversation_layer(raw, query="compare G650 vs G700", data_used=du)
    assert "Insufficient verified" not in out
    assert "Gulfstream G650" in out
    assert "Gulfstream G700" in out

