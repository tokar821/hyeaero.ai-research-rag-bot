"""Phase 47 — broker humanizer tests (presentation only)."""

from __future__ import annotations

from services.conversation.broker_humanizer import humanize_broker_language


def test_humanize_key_risk():
    out = humanize_broker_language("Key risk: Deferred maintenance.")
    assert "Key risk:" not in out
    assert out.lower().startswith("the biggest risk here is")


def test_humanize_what_i_would_do():
    out = humanize_broker_language("What I would do: verify logbooks.")
    assert "What I would do:" not in out
    assert "If I were spending my own money" in out


def test_supporting_market_context_header():
    out = humanize_broker_language("Supporting market context:\n• Median $22M")
    assert "Supporting market context" not in out
    assert "market context" in out.lower()

