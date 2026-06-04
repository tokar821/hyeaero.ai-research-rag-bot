"""
Phase 39 — broker conversation layer (response generation only).

Final pass on client-facing answers: no internal terminology, broker tone, clean markdown.
Does not alter routing, market math, adversarial classification, or UnifiedBrokerState fields.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from services.broker.broker_language import sanitize_broker_language
from services.conversation.broker_fallbacks import (
    apply_broker_fallbacks,
    contains_internal_language,
)
from services.conversation.comparison_presentation_guard import guard_comparison_presentation
from services.conversation.broker_style import apply_broker_style
from services.conversation.broker_humanizer import humanize_broker_language
from services.conversation.output_cleaner import clean_broker_output

logger = logging.getLogger(__name__)


def _detect_intent_from_context(
    data_used: Optional[Dict[str, Any]],
    query: str,
) -> str:
    du = data_used or {}
    dispatch = str(du.get("authority_dispatch_kind") or "").strip().lower()
    if dispatch:
        return dispatch
    qri = str(du.get("query_recommendation_intent") or "").strip().lower()
    if qri == "aircraft_comparison":
        return "comparison"
    if du.get("deal_killer") or du.get("buy_decision_dispatch"):
        return "buy_decision"
    if du.get("comparison_v2"):
        return "comparison"
    if du.get("alternative_execution"):
        return "alternative"
    q = (query or "").lower()
    if "good deal" in q or "overpriced" in q or "worth it" in q:
        return "buy_decision"
    if " vs " in q or " versus " in q or q.startswith("compare "):
        return "comparison"
    return "other"


def apply_broker_conversation_layer(
    answer: str,
    *,
    query: str = "",
    intent_type: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Transform a raw consultant answer into broker-facing prose.

    Safe to call multiple times; idempotent for already-clean text.
    """
    raw = (answer or "").strip()
    if not raw:
        fallback = apply_broker_fallbacks("", query=query)
        return fallback

    du = data_used if isinstance(data_used, dict) else {}
    try:
        from services.broker_execution.output_governance import is_llm_primary_output

        if is_llm_primary_output(du):
            du["broker_conversation_layer_hygiene_only"] = 1
            return sanitize_broker_language(clean_broker_output(raw, use_unicode_bullets=True))
    except Exception:
        pass

    try:
        from services.executive_broker.acquisition_budget_reality import prepend_budget_reality_opening

        raw = prepend_budget_reality_opening(raw, data_used=du)
    except Exception:
        pass

    intent = (intent_type or _detect_intent_from_context(data_used, query)).strip().lower()

    if intent == "comparison":
        raw = guard_comparison_presentation(raw, query=query, data_used=data_used)

    styled = apply_broker_style(raw, intent_type=intent)
    translated = apply_broker_fallbacks(styled, query=query)
    cleaned = clean_broker_output(translated, use_unicode_bullets=True)
    cleaned = humanize_broker_language(cleaned, data_used=data_used)
    final = sanitize_broker_language(cleaned)

    if contains_internal_language(final):
        logger.debug(
            "broker conversation layer: residual internal language after pass query=%r",
            (query or "")[:80],
        )
        final = apply_broker_fallbacks(final, query=query)

    try:
        from services.broker_scoring.broker_quality_score import attach_broker_quality_score

        attach_broker_quality_score(final, query=query, data_used=data_used)
    except Exception as exc:
        logger.debug("broker quality score skipped: %s", exc)

    try:
        from services.broker_audit.broker_trace import attach_broker_trace
        from services.broker_audit.broker_trust_score import attach_broker_trust_score

        attach_broker_trace(final, query=query, data_used=data_used)
        attach_broker_trust_score(final, query=query, data_used=data_used)
    except Exception as exc:
        logger.debug("broker audit attach skipped: %s", exc)

    return final.strip()


__all__ = ["apply_broker_conversation_layer"]
