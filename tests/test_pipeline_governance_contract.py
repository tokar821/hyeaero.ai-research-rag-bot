"""
Engineering contract: production LLM path uses hygiene-only layers.
"""

import re

from services.broker_execution.output_governance import (
    apply_governed_client_answer,
    resolve_output_governance,
)
from services.broker_execution.response_mode_classifier import ResponseMode
from rag.response_safety import enforce_consultant_quality
from services.consultant.intelligence_engine import run_consultant_intelligence_layer


_FORBIDDEN = re.compile(
    r"(?is)\b(?:if\s+i\s+were\s+buying|send\s+me\s+the\s+listing\s+package|operational\s+synthesis)\b"
)


def test_tail_ownership_llm_contract_chain():
    query = "Who owns N807JS?"
    du = {
        "llm_executed": True,
        "consultant_llm_draft": 1,
        "tail_facts": [
            {"kind": "ownership", "label": "Owner", "value": "Acme Aviation LLC"},
        ],
    }
    plan = resolve_output_governance(query, du)
    assert plan.response_mode == ResponseMode.FACT_ONLY
    assert plan.llm_primary
    assert not plan.executive

    raw = "Owner: Acme Aviation LLC."
    intel = run_consultant_intelligence_layer(answer=raw, query=query, data_used=du)
    assert intel.data_used_patch.get("consultant_intelligence_llm_primary_hygiene_only") == 1

    safe = enforce_consultant_quality(intel.answer, query=query, data_used=du)
    final = apply_governed_client_answer(safe, query=query, data_used=du)

    assert not _FORBIDDEN.search(final)
    assert "Acme" in final
    assert du.get("output_governance_applied") == 1
