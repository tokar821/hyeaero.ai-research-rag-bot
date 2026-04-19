"""Tests for buyer psychology advisory engine."""

from __future__ import annotations

import pytest

from services import buyer_psychology_engine as bpe


def test_g650_cheaper_interior_profile():
    q = "I want something like G650 but cheaper. Show interior."
    out = bpe.run_buyer_psychology_engine(
        latest_query=q,
        conversation_history=None,
        detected_aircraft_interest=["G650"],
        budget_hint=None,
        mission_hint=None,
        phly_rows=None,
    )
    assert "ASPIRATIONAL" in out["buyer_type"]
    assert "LIFESTYLE" in out["buyer_type"] or "LIFESTYLE" in out.get("buyer_type", "")
    assert out["confidence"] >= 0.7
    assert out["strategy"] in ("MEDIUM", "LARGE")
    assert "premium" in out["detected_signals"]["preference"].lower() or "visual" in out[
        "detected_signals"
    ]["preference"].lower()
    assert out["response_guidance"]


def test_explicit_budget_small_gap():
    q = "I have $45M and want a Gulfstream G650 if it fits."
    out = bpe.run_buyer_psychology_engine(
        latest_query=q,
        conversation_history=None,
        detected_aircraft_interest=["Gulfstream G650"],
        phly_rows=None,
    )
    assert out["parsed_budget_millions_usd"] == 45.0
    assert out["strategy"] == "SMALL"
    assert "compatible" in out["gap_analysis"].lower() or "broadly" in out["gap_analysis"].lower()


def test_value_buyer_challenger():
    q = "Best deal under $8M for a super midsize — what are hidden costs?"
    out = bpe.run_buyer_psychology_engine(
        latest_query=q,
        conversation_history=None,
        detected_aircraft_interest=["Challenger 350"],
        phly_rows=None,
    )
    assert "VALUE" in out["buyer_type"]
    assert "cautious" in out["detected_signals"]["risk"].lower() or "deal" in out["detected_signals"][
        "risk"
    ].lower()


def test_toggle_off(monkeypatch):
    monkeypatch.setenv("CONSULTANT_BUYER_PSYCHOLOGY", "0")
    assert bpe.consultant_buyer_psychology_enabled() is False


def test_system_prompt_block_truncation():
    huge = {"buyer_type": "X", "confidence": 0.5, "detected_signals": {}, "gap_analysis": "y" * 3000, "strategy": "M", "strategy_detail": "", "response_guidance": "z" * 3000}
    s = bpe.format_buyer_psychology_for_system_prompt(huge, max_chars=200)
    assert len(s) <= 200
    assert s.endswith("...")
