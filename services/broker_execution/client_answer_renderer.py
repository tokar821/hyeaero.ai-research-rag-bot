"""
Single client-answer renderer — the only post-LLM mutation path for LLM-primary turns.

Architecture:
  Retriever (optional) → Reasoning (facts) → LLM draft → render_client_answer
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from services.broker_execution.output_governance import (
    enforce_final_render_contract,
    is_llm_primary_output,
    resolve_output_governance,
)


_REGISTRY_FIELD_RE = re.compile(
    r"(?im)^\s*(?:•\s*)?(?:aircraft|owner|registration|status|year|serial)\s*:\s*.+$"
)


def _normalize_block(block: str) -> str:
    return re.sub(r"\s+", " ", (block or "").strip().lower())


def collapse_duplicate_registry_blocks(answer: str) -> str:
    """Remove repeated registry fact blocks (common when LLM + facts both present)."""
    text = (answer or "").strip()
    if not text:
        return text
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) < 2:
        return text
    kept: list[str] = []
    seen_norms: set[str] = set()
    seen_registry_keys: set[str] = set()
    for para in paragraphs:
        norm = _normalize_block(para)
        if not norm:
            continue
        field_lines = _REGISTRY_FIELD_RE.findall(para)
        if len(field_lines) >= 2:
            reg_key = _normalize_block("\n".join(sorted(field_lines[:8])))
            if reg_key in seen_registry_keys:
                continue
            seen_registry_keys.add(reg_key)
        if norm in seen_norms:
            continue
        seen_norms.add(norm)
        kept.append(para)
    return "\n\n".join(kept).strip()


def render_client_answer(
    answer: str,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Final render step for client-visible text.

    LLM-primary: hygiene + dedupe + contract only (no broker/market/tail template layers).
    Non-LLM: delegates to full ``apply_governed_client_answer``.
    """
    q = (query or "").strip()
    du = data_used if isinstance(data_used, dict) else {}
    body = (answer or "").strip()

    if not is_llm_primary_output(du):
        from services.broker_execution.output_governance import apply_governed_client_answer

        return apply_governed_client_answer(body, query=q, data_used=du)

    plan = resolve_output_governance(q, du)
    du["client_answer_renderer"] = "llm_primary_single_pass"

    try:
        from services.broker_execution.output_governance import _apply_conversation_hygiene

        body = _apply_conversation_hygiene(body, query=q, data_used=du)
    except Exception:
        pass

    body = collapse_duplicate_registry_blocks(body)

    try:
        from services.broker_execution.response_compression_layer import (
            apply_response_compression_layer,
        )

        body = apply_response_compression_layer(body, query=q, data_used=du)
    except Exception:
        pass

    if plan.final_contract:
        body = enforce_final_render_contract(body, query=q, data_used=du)

    du["output_governance_applied"] = 1
    du["model_authority_skipped_llm_primary"] = 1
    du["client_answer_renderer_applied"] = 1
    return body.strip()


__all__ = ["collapse_duplicate_registry_blocks", "render_client_answer"]
