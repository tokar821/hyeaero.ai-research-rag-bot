"""Regression tests for manual chat report issues (Jun 2026)."""

from __future__ import annotations

from services.broker_execution.intent_answer_contract import build_intent_answer_contract_suffix
from services.broker_execution.output_governance import is_llm_primary_output
from services.broker_execution.response_mode_classifier import ResponseMode, classify_response_mode
from services.response.response_normalizer import normalize_consultant_response
from rag.aviation_engines.geo import mission_endpoints_from_text


def test_llm_primary_skips_overview_scaffold():
    raw = (
        "The Challenger 350 is the better pick for NYC to LA.\n"
        "For London you will need a fuel stop with either aircraft."
    )
    out = normalize_consultant_response(
        {
            "answer": raw,
            "data_used": {"llm_executed": 1, "consultant_llm_draft": raw},
        },
        context={"query": "Challenger 350 vs Praetor 500 NYC LA"},
    )
    assert "Overview" not in out.answer_text
    assert "Challenger 350" in out.answer_text


def test_mission_endpoints_new_york_to_tokyo():
    ep = mission_endpoints_from_text("9 passengers from New York to Tokyo nonstop")
    assert ep is not None
    c0, c1, nm = ep
    assert c0 in ("KTEB", "KJFK", "KEWR")
    assert c1 in ("RJTT", "RJAA")
    assert nm > 5000


def test_intent_contract_tail_ownership_shape():
    suffix = build_intent_answer_contract_suffix(
        "Who is the registered owner of N525AB?",
        data_used={"tail_investigation_dispatch": "N525AB"},
        response_mode=ResponseMode.FACT_ONLY,
    )
    assert "2–5 short lines" in suffix
    assert "Forbidden section headers" in suffix


def test_classify_sale_status_fact_only():
    mode = classify_response_mode(
        "Is N650GS currently on the market?",
        data_used={"tail_registration": "N650GS"},
    )
    assert mode == ResponseMode.FACT_ONLY


def test_is_llm_primary_output():
    assert is_llm_primary_output({"llm_executed": 1})
    assert not is_llm_primary_output({})
