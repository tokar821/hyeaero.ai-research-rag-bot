"""Output governance — single writer / LLM-primary gating."""

from __future__ import annotations

import re

import pytest

from services.broker_execution.output_governance import (
    apply_governed_client_answer,
    enforce_final_render_contract,
    is_llm_primary_output,
    resolve_output_governance,
)
from services.broker_execution.response_mode_classifier import ResponseMode


def test_llm_primary_skips_template_layers():
    du = {"llm_executed": True, "consultant_llm_draft": 1}
    plan = resolve_output_governance("Who owns N807JS?", du)
    assert plan.llm_primary
    assert plan.response_mode == ResponseMode.FACT_ONLY
    assert not plan.executive
    assert not plan.data_first
    assert not plan.market_reality
    assert not plan.broker_decision


def test_deterministic_tail_allows_data_first_not_executive():
    du = {}
    plan = resolve_output_governance("Who owns N807JS?", du)
    assert not plan.llm_primary
    assert plan.response_mode == ResponseMode.FACT_ONLY
    assert not plan.executive
    assert plan.data_first


def test_final_contract_strips_forbidden_and_registry_blocks():
    raw = (
        "[REGISTRY FACTS]\nOwner: Acme LLC\n\n"
        "If I were buying today I would focus on records.\n\n"
        "• Owner: Acme LLC\n"
        "execution_path: tail_dispatch\n"
    )
    du = {
        "tail_facts": [
            {"kind": "ownership", "label": "Owner", "value": "Acme LLC"},
            {"kind": "aircraft_model", "label": "Aircraft", "value": "G550"},
        ],
        "tail_registration": "N807JS",
    }
    out = enforce_final_render_contract(raw, query="Who owns N807JS?", data_used=du)
    assert "if i were buying" not in out.lower()
    assert "execution_path" not in out.lower()
    assert "[registry" not in out.lower()
    assert "acme" in out.lower()


def test_governed_pipeline_preserves_llm_prose():
    du = {"llm_executed": True}
    prose = "Owner: Acme Aviation LLC."
    out = apply_governed_client_answer(prose, query="Who owns N807JS?", data_used=du)
    assert "Acme" in out
    assert du.get("output_governance_applied") == 1
    assert du.get("model_authority_skipped_llm_primary") == 1


def test_is_llm_primary_output():
    assert is_llm_primary_output({"llm_executed": True})
    assert is_llm_primary_output({"consultant_llm_draft": 1})
    assert not is_llm_primary_output({})


@pytest.mark.parametrize(
    "phrase",
    [
        "If I were buying today",
        "operational synthesis",
        "send me the listing package",
    ],
)
def test_hygiene_strips_forbidden_on_llm_path(phrase):
    du = {"llm_executed": True}
    raw = f"• Owner: Test Co\n\n{phrase}, I would verify records."
    out = apply_governed_client_answer(raw, query="Who owns N123AB?", data_used=du)
    assert phrase.lower() not in out.lower()
