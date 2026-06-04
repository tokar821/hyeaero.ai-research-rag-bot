"""Tail / registration investigation — known facts only, no speculation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_tail_investigation_brief(
    registration: str,
    *,
    model: Optional[str] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Return broker prose for a specific tail — only states what is in context or DB paths.
    """
    reg = (registration or "").strip().upper()
    du = data_used if isinstance(data_used, dict) else {}
    lines: List[str] = []

    try:
        from services.broker_execution.tail_fact_loader import ensure_tail_facts_for_query
        from services.broker_execution.tail_fact_renderer import (
            render_tail_facts_block,
            select_tail_facts,
        )

        ensure_tail_facts_for_query(f"Tell me about {reg}", du)
        facts = select_tail_facts(du, reg)
        if facts:
            du["tail_selected_facts"] = facts
            block = render_tail_facts_block(facts, registration=reg)
            if block:
                lines.append(block)
                lines.append("")
    except Exception:
        pass

    if not lines:
        lines.append(
            f"On {reg}, I would not speculate beyond what we can verify on the record and the listing package."
        )

    dk = du.get("deal_killer") or {}
    if isinstance(dk, dict) and str(dk.get("tail") or "").upper() == reg:
        if dk.get("model"):
            lines.append(f"• Aircraft tied to this inquiry: {dk.get('model')}.")
        if dk.get("ask_usd"):
            try:
                lines.append(f"• Referenced ask: ${float(dk['ask_usd'])/1e6:.1f}M.")
            except (TypeError, ValueError):
                pass

    if model:
        lines.append(f"• Model context from your message: {model}.")

    phly = du.get("phly_rows") or du.get("phly_authority")
    if phly:
        lines.append("• Synced listing or registry snippets are in context — I would reconcile those against the seller's spec sheet.")

    try:
        from services.broker_execution.tail_depth_mode import TailDepthMode, classify_tail_depth_mode

        depth, _ = classify_tail_depth_mode(f"Tell me about {reg}")
    except Exception:
        depth = None

    if depth == TailDepthMode.ACQUISITION:
        lines.append(
            "\nBefore treating this tail as a buy, I need:"
            "\n• Year and total time"
            "\n• Engine program status"
            "\n• Maintenance / damage history"
            "\n• The listing link or broker package"
        )
    elif depth == TailDepthMode.DETAIL:
        lines.append(
            "\nFor a full profile I would combine registry, listing, program status, and recent market comps on this serial."
        )
    else:
        du["tail_investigation_no_acquisition_scaffold"] = 1

    if not model and not dk:
        lines.append(
            "\nI do not have enough verified tail-specific data loaded for this registration alone — "
            "send the listing or logbook summary and I will give a direct deal read."
        )

    return "\n".join(lines).strip()


__all__ = ["build_tail_investigation_brief"]
