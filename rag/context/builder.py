"""Assemble consultant LLM context: intent-filtered sections and token budget."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from rag.context.intent_context_policy import (
    assemble_filtered_context,
    build_section_bodies,
    consultant_context_token_budget,
    effective_context_char_cap,
    estimate_tokens,
    section_mask_for_intent,
)
from rag.intent.schemas import IntentClassification


def build_consultant_llm_context(
    *,
    phly_authority: str,
    market_block: str,
    tavily_block: str,
    rag_results: List[Dict[str, Any]],
    max_context_chars: int,
    intent_classification: Optional[IntentClassification] = None,
    max_context_tokens: Optional[int] = None,
) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Context-builder: SQL / authority split into AIRCRAFT_SPECS, OPERATIONAL_DATA,
    MARKET_DATA, REGISTRY_DATA; filtered by Ask Consultant intent; capped by token budget.

    When ``intent_classification`` is None, uses the legacy flat join (no section headers)
    for backward compatibility.
    """
    tok = max_context_tokens
    if tok is None:
        env_t = (os.getenv("CONSULTANT_CONTEXT_MAX_TOKENS") or "").strip()
        if env_t.isdigit():
            tok = int(env_t)
        else:
            tok = consultant_context_token_budget(None)

    cap = effective_context_char_cap(max_context_chars=max_context_chars, max_context_tokens=tok)

    if intent_classification is None:
        parts: List[str] = []
        total = 0
        sep = 20

        def append_block(text: str) -> None:
            nonlocal total
            chunk = (text or "").strip()
            if not chunk:
                return
            if total + len(chunk) + sep > cap:
                chunk = chunk[: max(0, cap - total - sep)]
            if not chunk.strip():
                return
            parts.append(chunk)
            total += len(chunk) + sep

        append_block(phly_authority)
        append_block(market_block)
        append_block(tavily_block)

        for r in rag_results:
            text = (r.get("full_context") or r.get("chunk_text") or "").strip()
            if not text:
                continue
            if total + len(text) + sep > cap:
                text = text[: max(0, cap - total - sep)]
            if text:
                parts.append(text)
                total += len(text) + sep
            if total >= cap:
                break

        context = "\n\n---\n\n".join(parts) if parts else ""
        meta = {
            "sections_included": [],
            "context_tokens_est": estimate_tokens(context),
            "context_char_budget": cap,
            "legacy_flat": True,
        }
        return context, parts, meta

    mask = section_mask_for_intent(intent_classification)
    bodies = build_section_bodies(
        phly_authority=phly_authority,
        market_block=market_block,
        tavily_block=tavily_block,
        rag_results=rag_results,
        mask=mask,
    )
    context, pieces, meta = assemble_filtered_context(
        section_bodies=bodies, mask=mask, max_chars=cap
    )
    meta["legacy_flat"] = False
    return context, pieces, meta
