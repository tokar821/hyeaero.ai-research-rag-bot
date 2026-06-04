"""
Phase 41 — broker decision synthesis orchestrator.

Runs after dispatch / recovery, before conversation rendering.
Does not alter routing, market math, or valuation formulas.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from services.broker_decision.broker_decision_builder import build_broker_decision
from services.broker_decision.broker_reasoning_writer import write_broker_decision
from services.broker_decision.conversation_relevance_guard import should_synthesize_decision

logger = logging.getLogger(__name__)


def apply_broker_decision_synthesis(
    answer: str,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Synthesize acquisition-advisor prose when the buyer question is not answered
    by catalog/spec output.
    """
    raw = (answer or "").strip()
    q = (query or "").strip()
    du = data_used if isinstance(data_used, dict) else {}

    if not q:
        return raw

    decision = build_broker_decision(q, data_used=du, raw_answer=raw)
    if decision is None:
        return raw

    try:
        from services.client_context.broker_context_builder import BrokerConversationContext
        from services.client_context.recommendation_consistency import (
            apply_consistency_to_broker_decision,
        )

        ctx = BrokerConversationContext.from_dict(
            du.get("broker_conversation_context") or du.get("client_context")
        )
        if ctx.remembered_budget_musd is not None or ctx.remembered_targets:
            ddict = apply_consistency_to_broker_decision(decision.to_dict(), ctx)
            if ddict.get("alternatives"):
                decision.alternatives = list(ddict["alternatives"])
    except Exception:
        pass

    if not should_synthesize_decision(raw, query=q):
        du["broker_decision"] = decision.to_dict()
        return raw

    synthesized = write_broker_decision(decision, raw_answer=raw, preserve_supporting=True)

    du["broker_decision"] = decision.to_dict()
    du["broker_decision_synthesis_applied"] = 1

    if not synthesized.strip():
        return raw

    logger.debug(
        "broker decision synthesis: intent=%s query=%r",
        decision.decision_intent,
        q[:80],
    )
    return synthesized


__all__ = ["apply_broker_decision_synthesis"]
