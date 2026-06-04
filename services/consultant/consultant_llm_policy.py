"""
Consultant LLM policy — when structured/dispatch paths must defer to LLM narration.

Presentation and routing only; does not change ranking, retrieval, or tier logic.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

_TAIL_REGISTRY_RE = re.compile(
    r"(?is)\b(?:who\s+owns|who\s+is\s+the\s+owner|owner\s+of|ownership\s+of|"
    r"show\s+me|tell\s+me\s+about|registry|registration\s+record|for\s+sale\s+status)\b"
)
_TAIL_TOKEN_RE = re.compile(r"\bN[A-Z0-9]{3,6}\b")


def consultant_llm_narration_enabled() -> bool:
    return os.getenv("CONSULTANT_FORCE_LLM", "1").strip().lower() not in ("0", "false", "no")


def consultant_narrate_structured_dispatch() -> bool:
    return os.getenv("CONSULTANT_LLM_NARRATE_STRUCTURED", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def is_tail_registry_query(query: str) -> bool:
    """Only owner/sale registry-card turns defer to structured registry handling."""
    q = (query or "").strip()
    if not q or not _TAIL_TOKEN_RE.search(q.upper()):
        return False
    try:
        from services.broker_execution.tail_depth_mode import TailDepthMode, classify_tail_depth_mode

        depth, _ = classify_tail_depth_mode(q)
        return depth in (TailDepthMode.OWNER, TailDepthMode.SALE_STATUS)
    except Exception:
        return bool(_TAIL_REGISTRY_RE.search(q))


def authority_dispatch_defer_to_llm(dispatch_result: Any) -> bool:
    """
    When True, caller merges dispatch ``data_used`` / facts but must run the LLM draft.
    """
    if not consultant_llm_narration_enabled() or dispatch_result is None:
        return False
    kind = str(getattr(dispatch_result, "dispatch_kind", "") or "").lower()
    du = getattr(dispatch_result, "data_used", None) or {}
    if kind == "valuation" or du.get("tail_investigation_defer_llm"):
        return True
    if du.get("broker_reasoning_acquisition_guidance") or du.get(
        "broker_reasoning_alternative_guidance"
    ):
        return True
    if consultant_narrate_structured_dispatch() and kind in (
        "comparison",
        "alternative",
    ):
        cv2 = du.get("comparison_v2")
        if isinstance(cv2, dict) and str(cv2.get("status") or "").upper() == "OK":
            return True
        if str(getattr(dispatch_result, "answer", "") or "").strip():
            return True
    return False


def query_requires_llm_narration(query: str, *, context: Optional[dict] = None) -> bool:
    """Execution-boundary: do not bypass LLM for these turns when policy is on."""
    if not consultant_llm_narration_enabled():
        return False
    if is_tail_registry_query(query):
        return True
    ctx = context if isinstance(context, dict) else {}
    intent = str(ctx.get("deterministic_intent") or "").lower()
    if consultant_narrate_structured_dispatch() and intent in ("comparison",):
        return True
    auth = ctx.get("authority_dispatch_result")
    if auth is not None and authority_dispatch_defer_to_llm(auth):
        return True
    return False


def structured_dispatch_llm_block(dispatch_result: Any) -> str:
    """Context block for LLM narration of structured/dispatch output."""
    answer = str(getattr(dispatch_result, "answer", "") or "").strip()
    kind = str(getattr(dispatch_result, "dispatch_kind", "") or "")
    if not answer:
        return ""
    return (
        f"[AUTHORITATIVE STRUCTURED FACTS — {kind} — narrate professionally]\n"
        "Write a concise, expert broker answer in natural language. "
        "Use only facts below; do not invent data. "
        "Do not use template phrases ('If I were buying today', 'Send me the listing package', "
        "'Before treating this tail'). Lead with the direct answer.\n\n"
        f"{answer}"
    ).strip()


__all__ = [
    "authority_dispatch_defer_to_llm",
    "consultant_llm_narration_enabled",
    "consultant_narrate_structured_dispatch",
    "is_tail_registry_query",
    "query_requires_llm_narration",
    "structured_dispatch_llm_block",
]
