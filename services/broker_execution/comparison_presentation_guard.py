"""
Phase 56 — comparison responses: facts first; forbid acquisition voice without buy intent.
Does not modify executive_broker_layer.
"""

from __future__ import annotations

import re
from typing import Optional

from services.broker_execution.broker_execution_category import (
    BrokerExecutionCategory,
    classify_broker_execution_category,
    comparison_requests_recommendation,
)

_FORBIDDEN_COMPARISON_RE = re.compile(
    r"(?is)\b(?:if\s+i\s+were\s+buying\s+today|i\s+would\s+buy|i\s+would\s+focus\s+on|"
    r"i'd\s+focus\s+on|primary\s+recommendation)\b"
)
_FACT_LINE_RE = re.compile(
    r"(?is)\b(?:range|cabin|operating\s+cost|liquidity|passengers?|nm\b|nautical)"
)


def apply_comparison_presentation_guard(
    answer: str,
    *,
    query: str = "",
    data_used: Optional[dict] = None,
) -> str:
    du = data_used if isinstance(data_used, dict) else {}
    cat = classify_broker_execution_category(query, data_used=du)
    if cat != BrokerExecutionCategory.COMPARISON:
        return (answer or "").strip()
    if comparison_requests_recommendation(query):
        return (answer or "").strip()

    body = (answer or "").strip()
    if not body:
        return body

    if not _FORBIDDEN_COMPARISON_RE.search(body):
        return _reorder_facts_before_opinion(body)

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    kept = [p for p in paragraphs if not _FORBIDDEN_COMPARISON_RE.search(p)]
    if not kept:
        kept = [p for p in paragraphs if _FACT_LINE_RE.search(p)]
    if not kept:
        return _reorder_facts_before_opinion(body)

    du["comparison_acquisition_voice_stripped"] = True
    return _reorder_facts_before_opinion("\n\n".join(kept).strip())


def _reorder_facts_before_opinion(text: str) -> str:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) < 2:
        return text.strip()
    facts = [p for p in paragraphs if _FACT_LINE_RE.search(p) and not _FORBIDDEN_COMPARISON_RE.search(p)]
    opinions = [p for p in paragraphs if p not in facts]
    if not facts:
        return text.strip()
    return "\n\n".join(facts + opinions).strip()


__all__ = ["apply_comparison_presentation_guard"]
