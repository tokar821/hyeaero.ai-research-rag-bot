"""Inject conversation memory into broker answers."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from services.client_context.broker_context_builder import BrokerConversationContext


def personalize_answer(
    answer: str,
    *,
    query: str = "",
    context: Optional[BrokerConversationContext] = None,
    profile_dict: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Prepend or weave client-aware framing when memory exists.

    Does not change underlying market facts — only framing.
    """
    body = (answer or "").strip()
    if not body or context is None:
        return body

    prefix_parts: list[str] = []
    budget = context.remembered_budget_musd
    mfrs = context.preferred_manufacturers
    targets = context.remembered_targets
    stage = context.stage

    # Avoid double-personalization.
    if re.search(r"(?i)based on the \$|you(?:'|')ve been discussing|you mentioned", body[:200]):
        return body

    if budget is not None and not re.search(
        rf"(?i)\$?\s*{re.escape(str(int(budget)))}\s*m", body[:400]
    ):
        if re.search(r"(?i)\b(?:latitude|praetor|challenger|g650|g700|longitude)\b", body):
            prefix_parts.append(
                f"Based on the ${budget:.0f}M budget you've been discussing, "
            )
        elif re.search(r"(?i)\b(?:consider|focus|start with|would look)\b", body):
            prefix_parts.append(
                f"At your ${budget:.0f}M level, "
            )

    if mfrs and not budget and re.search(r"(?i)gulfstream|dassault|falcon|citation", body):
        if "Gulfstream" in mfrs and re.search(r"(?i)gulfstream|g650|g700", body):
            prefix_parts.append("Given your Gulfstream interest in this thread, ")
        elif mfrs[0] in body or any(m.lower() in body.lower() for m in mfrs):
            prefix_parts.append(f"Given your {mfrs[0]} focus in this thread, ")

    if context.active_aircraft and stage in ("ACTIVE_SHOPPING", "NEGOTIATING"):
        ac = context.active_aircraft
        if ac.lower() in (query or "").lower() or ac.lower() in body.lower():
            if not prefix_parts and re.search(r"(?i)should i buy|buy now|timing", query):
                prefix_parts.append(f"On the {ac} you're evaluating, ")

    if targets and re.search(r"(?i)cheaper|alternative|instead", query):
        pair = context.active_constraints.get("comparison_pair") or targets[:2]
        if isinstance(pair, list) and len(pair) >= 2:
            if not re.search(r"(?i)longitude|praetor|latitude", body[:120]):
                prefix_parts.append(
                    f"Sticking with your earlier {pair[0]} vs {pair[1]} comparison, "
                )

    if not prefix_parts:
        return body

    prefix = "".join(prefix_parts)
    first = body.split("\n\n", 1)[0]
    rest = body[len(first) :].lstrip()

    if first.endswith("."):
        personalized_first = prefix.rstrip() + first[0].lower() + first[1:] if first else prefix + first
    else:
        personalized_first = prefix + first

    if rest:
        return f"{personalized_first}\n\n{rest}".strip()
    return personalized_first.strip()


def enrich_query_context_for_reasoning(
    data_used: Dict[str, Any],
    context: BrokerConversationContext,
) -> None:
    """Stamp remembered constraints for downstream layers (read-only hints)."""
    data_used["broker_conversation_context"] = context.to_dict()
    if context.remembered_budget_musd is not None:
        br = data_used.setdefault("broker_reasoning", {})
        if isinstance(br, dict):
            mission = br.setdefault("mission", {})
            if isinstance(mission, dict):
                mission["acquisition_budget_musd"] = context.remembered_budget_musd
                mission["from_client_memory"] = True


__all__ = ["enrich_query_context_for_reasoning", "personalize_answer"]
