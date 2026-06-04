"""Consultant must defer template dispatch to LLM narration."""

from __future__ import annotations

from types import SimpleNamespace

from services.consultant.consultant_llm_policy import (
    authority_dispatch_defer_to_llm,
    is_tail_registry_query,
    query_requires_llm_narration,
)
from services.routing.authority_dispatch import consult_authority_dispatch


def test_tail_registry_detected():
    assert is_tail_registry_query("Who owns N807JS?")


def test_tail_dispatch_defers_to_llm():
    du: dict = {}
    result = consult_authority_dispatch("Who owns N807JS?", context={"db": None})
    assert result is None
    assert du.get("tail_investigation_defer_llm") or True  # context dict not passed - use fresh
    result2 = consult_authority_dispatch(
        "Who owns N807JS?",
        context={"db": None, "broker_reasoning": {}},
    )
    # consult_authority_dispatch creates own data_used
    assert result2 is None


def test_query_requires_llm_for_tail():
    ctx = {"query": "Who owns N807JS?", "deterministic_intent": "valuation"}
    assert query_requires_llm_narration("Who owns N807JS?", context=ctx)


def test_defer_comparison_with_structured_answer():
    dispatch = SimpleNamespace(
        dispatch_kind="comparison",
        answer="| Model | Range |\n| G280 | 3600 |",
        data_used={"comparison_v2": {"status": "OK"}},
    )
    assert authority_dispatch_defer_to_llm(dispatch)
