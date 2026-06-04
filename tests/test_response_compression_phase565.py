"""Phase 56.5 — response compression and fact-mode enforcement."""

from __future__ import annotations

import re

import pytest

from services.broker_execution.response_compression_layer import apply_response_compression_layer
from services.broker_execution.response_mode_classifier import ResponseMode, classify_response_mode
from tests.e2e.broker_certification_helpers import broker_certify

_FORBIDDEN_TAIL = re.compile(
    r"(?is)before\s+treating|send\s+me\s+(?:the\s+)?listing\s+package|engine\s+program"
)
_FORBIDDEN_COMP = re.compile(
    r"(?is)if\s+i\s+were\s+buying|i\s+would\s+focus\s+on"
)


def test_classify_fact_only():
    assert classify_response_mode("Who owns N807JS?") == ResponseMode.FACT_ONLY


def test_classify_comparison():
    assert classify_response_mode("G280 vs Longitude") == ResponseMode.COMPARISON


def test_fact_only_max_lines(monkeypatch):
    monkeypatch.setenv("RESPONSE_COMPRESSION_MODE", "replace")
    verbose = (
        "Registry facts for N807JS:\n"
        "• Registration: N807JS\n"
        "I can verify ownership and basic registry facts on N807JS.\n"
        "Send me the listing package.\n"
        "Before treating this tail as a buy, I need year and total time.\n"
        "• Aircraft: Citation Excel\n"
        "• Owner: HRL Ventures LLC\n"
    )
    du = {
        "tail_registration": "N807JS",
        "tail_selected_facts": [
            {"kind": "aircraft_model", "label": "Aircraft", "value": "Citation Excel"},
            {"kind": "ownership", "label": "Owner", "value": "HRL Ventures LLC"},
            {"kind": "registry_status", "label": "Status", "value": "For Sale"},
            {"kind": "year", "label": "Year", "value": "2003"},
        ],
    }
    out = apply_response_compression_layer(verbose, query="Who owns N807JS?", data_used=du)
    assert out.count("\n") + 1 <= 5
    assert not _FORBIDDEN_TAIL.search(out)
    assert "Citation Excel" in out
    assert du.get("response_mode") == "FACT_ONLY"


def test_comparison_table_format(monkeypatch):
    monkeypatch.setenv("RESPONSE_COMPRESSION_MODE", "replace")
    raw = (
        "If I were buying today, I'd focus on the Citation Longitude.\n\n"
        "G280 has strong range. Longitude has a larger cabin."
    )
    out = apply_response_compression_layer(raw, query="G280 vs Longitude", data_used={})
    assert "| Feature |" in out
    assert "Verdict:" in out
    assert not _FORBIDDEN_COMP.search(out)


@pytest.mark.slow
def test_broker_certify_tail_llm_powered():
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY required for LLM consultant path")
    answer, du, path = broker_certify("Who owns N807JS?", prefer_e2e=False)
    assert path == "layers"
    assert du.get("broker_certify_llm_raw") or du.get("llm_executed") or du.get("consultant_llm_draft")
    assert not _FORBIDDEN_TAIL.search(answer)
    assert "807JS" in answer.upper() or "excel" in answer.lower() or "owner" in answer.lower()


@pytest.mark.slow
def test_broker_certify_comparison_llm_powered():
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY required for LLM consultant path")
    answer, du, _ = broker_certify("G280 vs Longitude", prefer_e2e=False)
    assert du.get("broker_certify_llm_raw") or du.get("llm_executed")
    assert not _FORBIDDEN_COMP.search(answer)
    low = answer.lower()
    assert "range" in low or "cabin" in low or "g280" in low


@pytest.mark.slow
def test_mission_feasibility_fail_compressed():
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY required for LLM consultant path")
    answer, du, _ = broker_certify("8 passengers New York to Tokyo nonstop", prefer_e2e=False)
    assert "FAIL" in answer or "feasibility" in answer.lower() or "cannot" in answer.lower()
    assert "operational synthesis" not in answer.lower()


@pytest.mark.slow
def test_compression_metrics_present():
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY required for LLM consultant path")
    _, du, _ = broker_certify("Who owns N807JS?", prefer_e2e=False)
    assert "response_compression_score" in du
    assert "compression_low" in du
    assert "ideal_tokens_by_mode" in du
