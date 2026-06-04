"""
Phase 56 — enforce facts-before-opinion ordering for tail, listing, and comparison.
"""

from __future__ import annotations

from typing import Optional

from services.broker_execution.broker_execution_category import (
    BrokerExecutionCategory,
    classify_broker_execution_category,
    tail_registry_prepend_required,
)
from services.broker_execution.comparison_presentation_guard import apply_comparison_presentation_guard
from services.broker_execution.listing_parse_audit import strip_redundant_listing_questions
from services.broker_execution.tail_fact_loader import ensure_tail_facts_for_query
from services.broker_execution.tail_fact_renderer import (
    prepend_tail_facts_to_answer,
    select_tail_facts,
)
from services.market_reality.listing_detector import ListingMode, detect_listing_signal

_TAIL_CATEGORIES = frozenset(
    {
        "tail_lookup",
        "tail_ownership",
        "registry_lookup",
        "tail_history",
        "tail_investigation",
    }
)


def is_data_first_required(category: BrokerExecutionCategory) -> bool:
    from services.broker_execution.broker_execution_category import data_first_required

    return data_first_required(category)


def apply_data_first_layer(
    answer: str,
    *,
    query: str = "",
    data_used: Optional[dict] = None,
) -> str:
    du = data_used if isinstance(data_used, dict) else {}
    try:
        from services.broker_execution.output_governance import is_llm_primary_output

        if is_llm_primary_output(du):
            du["data_first_layer_skipped_llm_primary"] = 1
            return (answer or "").strip()
    except Exception:
        pass
    cat = classify_broker_execution_category(query, data_used=du)
    du["data_first_required"] = is_data_first_required(cat)

    if not du["data_first_required"]:
        return (answer or "").strip()

    body = (answer or "").strip()

    if cat.value in _TAIL_CATEGORIES or (
        detect_listing_signal(query).mode == ListingMode.TAIL_INVESTIGATION
    ):
        ensure_tail_facts_for_query(query, du)
        reg = str(du.get("tail_registration") or "").upper()
        if not reg:
            try:
                from services.broker_execution.tail_fact_loader import _extract_registration

                reg = _extract_registration(query) or ""
            except Exception:
                reg = ""
        facts = select_tail_facts(du, reg) if reg else []
        du["tail_selected_facts"] = facts
        if facts and tail_registry_prepend_required(query, du):
            from services.market_reality.tail_broker_rewriter import rewrite_tail_investigation

            body = prepend_tail_facts_to_answer(
                body, facts=facts, registration=reg, data_used=du
            )
            body = rewrite_tail_investigation(
                body, registration=reg, facts_available=True, data_used=du
            )
        elif not facts:
            du["tail_fallback_used"] = True

    if cat == BrokerExecutionCategory.LISTING:
        body = strip_redundant_listing_questions(body, data_used=du)

    if cat == BrokerExecutionCategory.COMPARISON:
        body = apply_comparison_presentation_guard(body, query=query, data_used=du)

    return body.strip()


__all__ = ["apply_data_first_layer", "is_data_first_required"]
