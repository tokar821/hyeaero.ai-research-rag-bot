"""Phase 47 — acceptance tests for broker-like final answer quality."""

from __future__ import annotations

import re

import pytest

from services.adversarial.adversarial_preprocessor import preprocess_adversarial_query
from services.broker_reasoning.broker_reasoning_layer import apply_broker_reasoning_layer
from services.broker_decision.broker_decision_layer import apply_broker_decision_synthesis
from services.client_context.client_context_layer import apply_client_context_turn, personalize_client_answer
from services.conversation.broker_conversation_layer import apply_broker_conversation_layer
from services.executive_broker.executive_broker_layer import apply_executive_broker_layer
from services.intent_collapse.intent_collapse_engine import apply_intent_collapse
from services.market_reality.market_reality_layer import apply_market_reality_layer
from services.truth_compression.truth_compression_layer import apply_truth_compression


def _final_answer(query: str, *, state: dict | None = None, history: list | None = None) -> tuple[dict, str]:
    du: dict = {}
    clean = preprocess_adversarial_query(query, data_used=du)
    apply_client_context_turn(query, data_used=du, client_conversation_state=state or {}, history=history)
    apply_intent_collapse(query, data_used=du, normalized_query=clean.normalized_query)
    apply_broker_reasoning_layer(clean.normalized_query, data_used=du)

    # Use a deterministic placeholder raw answer; the layers under test should rewrite tone/structure.
    raw = "INSUFFICIENT_DATA: Insufficient verified aircraft data for deterministic execution."
    ans = apply_broker_decision_synthesis(raw, query=query, data_used=du)
    ans = personalize_client_answer(ans, query=query, data_used=du)
    ans = apply_market_reality_layer(ans, query=query, data_used=du)
    ans = apply_executive_broker_layer(ans, query=query, data_used=du)
    ans = apply_truth_compression(ans, query=query, data_used=du)
    ans = apply_broker_conversation_layer(ans, query=query, data_used=du)
    return du, ans


def test_can_i_buy_g700_for_5m_begins_no():
    _, out = _final_answer("Can I realistically buy a G700 for $5M?")
    first = re.split(r"\n\s*\n", out.strip())[0].strip()
    assert first.lower().startswith("no.")


def test_budget_12m_gulfstream_never_recommends_g700():
    # this is a single-turn requirement
    _, out = _final_answer("I like Gulfstreams but my budget is only around $12M. What would you buy?")
    assert "G700" not in out
    assert "g700" not in out.lower()


def test_compare_g650_vs_g700_compares_without_insufficient():
    du = {"broker_reasoning": {"compare_models": ["Gulfstream G650", "Gulfstream G700"]}}
    out = apply_broker_conversation_layer(
        "Insufficient verified data for deterministic execution.\n\nVerified catalog comparison requires two recognized aircraft models.",
        query="Compare G650 vs G700 and tell me which is the better buy.",
        data_used=du,
    )
    assert "Insufficient verified" not in out
    assert "Gulfstream G650" in out
    assert "Gulfstream G700" in out


def test_g650_shopping_budget_range_broker_voice():
    _, out = _final_answer(
        "I've been looking at G650s for a few weeks. My budget is $15M-20M. What would you do?"
    )
    low = out.lower()
    assert "if i were buying today" in low or "i'd focus on" in low or low.startswith("i would")
    assert "checklist" not in low


def test_tail_investigation_no_speculation_requests_info():
    _, out = _final_answer("Is N719GF worth investigating?")
    low = out.lower()
    assert "cannot tell you whether it is worth buying" in low
    assert "send me the listing" in low
    assert "engine program" in low

