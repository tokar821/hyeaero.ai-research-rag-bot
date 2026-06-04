"""Phase 56 — tail fact utilization certification."""

from __future__ import annotations

import re

import pytest

from tests.e2e.broker_certification_helpers import broker_certify

TAIL_CASES = [
    "Who owns N807JS?",
    "Show me N650GS",
    "Tell me about N525AB",
]

_LISTING_PKG_RE = re.compile(r"(?is)send\s+me\s+(?:the\s+)?listing\s+package")


@pytest.mark.slow
@pytest.mark.parametrize("query", TAIL_CASES)
def test_tail_fact_utilization(query: str):
    import os

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY required for LLM consultant path")
    answer, du, path = broker_certify(query, prefer_e2e=False)
    assert path == "layers"
    flow = du.get("fact_flow") or {}
    retrieved = int(flow.get("retrieved_entities") or 0)
    selected = int(flow.get("selected_facts") or 0)
    rendered = int(flow.get("rendered_facts") or 0)

    assert du.get("broker_certify_llm_raw") or du.get("llm_executed"), "expected LLM-powered path"
    if retrieved > 0 or selected > 0:
        assert rendered > 0 or du.get("llm_executed"), (
            f"retrieved={retrieved} selected={selected} but rendered=0; answer={answer[:300]}"
        )
        assert not _LISTING_PKG_RE.search(answer), (
            "listing-package fallback must not appear when ownership facts exist"
        )

    assert du.get("fact_flow"), "fact_flow observability missing"
