"""Phase 41 — broker decision synthesis tests."""

from __future__ import annotations

import re

import pytest

from services.broker_decision.broker_decision_builder import build_broker_decision
from services.broker_decision.broker_decision_layer import apply_broker_decision_synthesis
from services.broker_decision.broker_reasoning_writer import write_broker_decision
from services.broker_decision.conversation_relevance_guard import (
    is_catalog_or_spec_dump,
    should_synthesize_decision,
)
from services.broker_decision.decision_intent_detector import DecisionIntent, detect_decision_intent
from services.broker_decision.alternative_engine import resolve_alternatives
from services.broker_decision.budget_matcher import match_budget_opportunities


def _first_para(text: str) -> str:
    return re.split(r"\n\s*\n", (text or "").strip())[0]


def _assert_leads_with_decision(text: str) -> None:
    first = _first_para(text).lower()
    assert any(
        first.startswith(p)
        for p in (
            "no",
            "yes",
            "possibly",
            "at $",
            "i would",
            "if you",
            "with about",
            "stretch",
            "buy when",
            "a gulfstream",
            "a ",
        )
    ), f"First paragraph does not answer buyer question: {first!r}"


_ACCEPTANCE_QUERIES = [
    "can I get a Gulfstream for 3M",
    "G700 under 5M",
    "I saw a G650 for 18M",
    "should I stretch budget for a G650",
    "what can I buy for 20M",
    "what is the smartest jet around 15M",
    "like a Longitude but cheaper",
    "should I buy now or wait",
    "prices are rising",
    "I found a G700 for 12M",
]


@pytest.mark.parametrize("query", _ACCEPTANCE_QUERIES)
def test_acceptance_first_paragraph_answers_question(query):
    du: dict = {}
    from services.adversarial.adversarial_preprocessor import preprocess_adversarial_query
    from services.broker_reasoning.broker_reasoning_layer import apply_broker_reasoning_layer

    preprocess_adversarial_query(query, data_used=du)
    apply_broker_reasoning_layer(query, data_used=du)

    raw = "INSUFFICIENT_DATA: Insufficient verified aircraft data for deterministic execution."
    out = apply_broker_decision_synthesis(raw, query=query, data_used=du)
    assert out.strip()
    assert "INSUFFICIENT_DATA" not in out
    _assert_leads_with_decision(out)


def test_realisticity_g700_under_5m():
    du = {"adversarial": {"budget_feasibility": "INFEASIBLE"}}
    decision = build_broker_decision("G700 under 5M", data_used=du)
    assert decision is not None
    assert decision.direct_answer.lower().startswith("no")
    text = write_broker_decision(decision)
    assert "5" in text
    assert "G700" in text or "g700" in text.lower()


def test_budget_match_20m():
    decision = build_broker_decision("what can I buy for 20M")
    assert decision is not None
    assert decision.answer_type == "opportunities"
    assert decision.alternatives
    text = write_broker_decision(decision)
    assert "20" in text
    _assert_leads_with_decision(text)


def test_alternative_longitude_cheaper():
    opps = resolve_alternatives("Citation Longitude")
    assert opps
    assert any("Latitude" in o.model for o in opps)
    decision = build_broker_decision("like a Longitude but cheaper")
    assert decision is not None
    text = write_broker_decision(decision)
    assert "Longitude" in text
    _assert_leads_with_decision(text)


def test_catalog_dump_detected():
    dump = (
        "Verified catalog comparison:\n"
        "- Gulfstream G700: ultra-long class; practical range 7700 nm.\n"
        "- Gulfstream G650: large class; practical range 7000 nm."
    )
    assert is_catalog_or_spec_dump(dump, query="G700 under 5M")
    assert should_synthesize_decision(dump, query="G700 under 5M")


def test_buy_or_wait():
    intent = detect_decision_intent("should I buy now or wait")
    assert intent == DecisionIntent.BUY_OR_WAIT
    decision = build_broker_decision("should I buy now or wait")
    text = write_broker_decision(decision)
    _assert_leads_with_decision(text)


def test_overpay_g650_18m():
    decision = build_broker_decision("I saw a G650 for 18M")
    assert decision is not None
    text = write_broker_decision(decision)
    assert "18" in text or "G650" in text
    _assert_leads_with_decision(text)


def test_budget_matcher_opportunities_not_bare_list():
    opps = match_budget_opportunities(20.0)
    assert len(opps) >= 2
    assert all(o.value_reason for o in opps)
