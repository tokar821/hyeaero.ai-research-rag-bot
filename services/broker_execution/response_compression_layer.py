"""
Phase 56.5 — response compression + fact-mode enforcement (formatting only).

Runs last in the broker post-pipeline. Does not alter ranking, retrieval, executive,
mission feasibility, or existing observability keys.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from services.broker_execution.response_compression_formatters import (
    _BROKER_TEMPLATE_RE,
    _FORBIDDEN_ALL_RE,
    format_analysis,
    format_comparison,
    format_fact_only,
    format_listing,
    format_mission,
)
from services.broker_execution.response_deduplication import (
    collapse_repeated_aircraft_mentions,
    deduplicate_lines,
)
from services.broker_execution.response_mode_classifier import (
    IDEAL_TOKENS_BY_MODE,
    MAX_TOKENS_BY_MODE,
    ResponseMode,
    classify_response_mode,
)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def truncate_to_token_budget(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text.strip()
    words = text.split()
    keep = max(1, int(max_tokens * 0.75))
    clipped = " ".join(words[:keep]).strip()
    if clipped and not clipped.endswith("."):
        clipped += "…"
    return clipped


def _apply_hygiene_only(
    answer: str,
    *,
    query: str,
    data_used: dict,
    mode: ResponseMode,
) -> str:
    """Dedupe and strip template boilerplate without replacing LLM narrative."""
    body = (answer or "").strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    kept = [
        p
        for p in paragraphs
        if not _FORBIDDEN_ALL_RE.search(p) and not _BROKER_TEMPLATE_RE.search(p.split("\n", 1)[0])
    ]
    body = deduplicate_lines("\n\n".join(kept).strip() if kept else body)
    aircraft = _aircraft_names_from_data(data_used, query or "")
    body = collapse_repeated_aircraft_mentions(body, aircraft)
    max_tok = MAX_TOKENS_BY_MODE.get(mode, 500)
    return truncate_to_token_budget(body, max_tok)


def attach_compression_metrics(
    answer: str,
    *,
    mode: ResponseMode,
    data_used: dict,
    pre_tokens: int,
) -> None:
    post_tokens = estimate_tokens(answer)
    ideal = IDEAL_TOKENS_BY_MODE.get(mode, 400)
    score = 1.0 - (post_tokens / max(ideal, 1))
    score = max(0.0, min(1.0, score))
    data_used["response_mode"] = mode.value
    data_used["response_compression_pre_tokens"] = pre_tokens
    data_used["response_compression_post_tokens"] = post_tokens
    data_used["ideal_tokens_by_mode"] = ideal
    data_used["response_compression_score"] = round(score, 3)
    data_used["compression_low"] = score < 0.4


def _compression_mode() -> str:
    import os

    return (os.getenv("RESPONSE_COMPRESSION_MODE") or "hygiene").strip().lower()


def apply_response_compression_layer(
    answer: str,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Enforce minimum-sufficient answers by response mode.

    ``hygiene`` (default): dedupe and strip boilerplate only — preserves LLM prose.
    ``replace``: rewrite into mode templates (certification / layers path).
    ``off``: no-op.
    """
    raw = (answer or "").strip()
    q = (query or "").strip()
    du = data_used if isinstance(data_used, dict) else {}
    if not raw:
        return raw

    mode_setting = _compression_mode()
    if mode_setting == "off":
        return raw

    pre_tokens = estimate_tokens(raw)
    mode = classify_response_mode(q, data_used=du)

    if du.get("llm_executed") or du.get("consultant_llm_draft"):
        du["response_compression_preserved_llm"] = 1

    if mode_setting == "hygiene" or du.get("response_compression_preserved_llm"):
        compressed = _apply_hygiene_only(raw, query=q, data_used=du, mode=mode)
        du["response_compression_layer_applied"] = 1
        du["response_compression_mode"] = "hygiene"
        attach_compression_metrics(compressed, mode=mode, data_used=du, pre_tokens=pre_tokens)
        return compressed

    if mode == ResponseMode.FACT_ONLY:
        compressed = format_fact_only(raw, query=q, data_used=du)
    elif mode == ResponseMode.COMPARISON:
        compressed = format_comparison(raw, query=q, data_used=du)
    elif mode == ResponseMode.LISTING:
        compressed = format_listing(raw, query=q, data_used=du)
    elif mode == ResponseMode.MISSION:
        compressed = format_mission(raw, query=q, data_used=du)
    else:
        compressed = format_analysis(raw, query=q, data_used=du)

    compressed = deduplicate_lines(compressed)
    aircraft = _aircraft_names_from_data(du, q)
    compressed = collapse_repeated_aircraft_mentions(compressed, aircraft)

    max_tok = MAX_TOKENS_BY_MODE.get(mode, 500)
    compressed = truncate_to_token_budget(compressed, max_tok)

    du["response_compression_layer_applied"] = 1
    attach_compression_metrics(compressed, mode=mode, data_used=du, pre_tokens=pre_tokens)
    return compressed or raw


def _aircraft_names_from_data(data_used: dict, query: str) -> list[str]:
    names: list[str] = []
    br = data_used.get("broker_reasoning") or {}
    if isinstance(br, dict):
        comp = br.get("comparison") or {}
        if isinstance(comp, dict):
            names.extend(str(m) for m in comp.get("models") or [])
    rec = data_used.get("executive_recommendation") or {}
    if isinstance(rec, dict) and rec.get("primary_recommendation"):
        names.append(str(rec["primary_recommendation"]))
    try:
        from services.broker_reasoning.comparison_soft_resolution import soft_resolve_comparison

        res = soft_resolve_comparison(query)
        names.extend(list(res.models))
    except Exception:
        pass
    return names


__all__ = [
    "apply_response_compression_layer",
    "attach_compression_metrics",
    "estimate_tokens",
]
