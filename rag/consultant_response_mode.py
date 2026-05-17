"""
Consultant response modes — thin compatibility layer over :mod:`services.response_mode_router`.

Legacy ``ConsultantResponseMode`` values remain for ``data_used`` consumers that expect
``visual_mode`` / ``advisory_mode``; routing logic lives in the response mode router.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Literal, Optional, TypedDict

from services.response_mode_router import (
    ResponseMode,
    route_response_mode,
    response_mode_prompt_suffix as _router_prompt_suffix,
    router_result_json,
)
from services.response_mode_router.schemas import ResponseModeRouterResult
from services.response_mode_router.triggers import DEAL_RE

Verbosity = Literal["minimal", "short", "detailed"]


class ConsultantResponseMode(str, Enum):
    """Legacy enum — mapped from :class:`ResponseMode`."""

    VISUAL_MODE = "visual_mode"
    ADVISORY_MODE = "advisory_mode"
    COMPARISON_MODE = "comparison_mode"
    DEAL_ANALYSIS_MODE = "deal_analysis_mode"
    CONVERSATION_MODE = "conversation_mode"
    TAIL_SPECIFIC = "tail_specific"
    INVALID_SANITY = "invalid_sanity"


class ConsultantResponseRouterResult(TypedDict):
    mode: str
    reason: str
    visual_priority: bool
    verbosity: Verbosity


_LEGACY_FROM_ROUTER: dict[str, ConsultantResponseMode] = {
    ResponseMode.IMAGE_SHOWCASE.value: ConsultantResponseMode.VISUAL_MODE,
    ResponseMode.ADVISORY.value: ConsultantResponseMode.ADVISORY_MODE,
    ResponseMode.FOLLOWUP_CONTINUATION.value: ConsultantResponseMode.ADVISORY_MODE,
    ResponseMode.COMPARISON_MODE.value: ConsultantResponseMode.COMPARISON_MODE,
    ResponseMode.EDUCATIONAL_MODE.value: ConsultantResponseMode.ADVISORY_MODE,
    ResponseMode.TAIL_SPECIFIC.value: ConsultantResponseMode.TAIL_SPECIFIC,
    ResponseMode.INVALID_SANITY.value: ConsultantResponseMode.INVALID_SANITY,
}

_ROUTER_FROM_LEGACY: dict[str, ResponseMode] = {
    ConsultantResponseMode.VISUAL_MODE.value: ResponseMode.IMAGE_SHOWCASE,
    ConsultantResponseMode.ADVISORY_MODE.value: ResponseMode.ADVISORY,
    ConsultantResponseMode.COMPARISON_MODE.value: ResponseMode.COMPARISON_MODE,
    ConsultantResponseMode.DEAL_ANALYSIS_MODE.value: ResponseMode.ADVISORY,
    ConsultantResponseMode.CONVERSATION_MODE.value: ResponseMode.ADVISORY,
    ConsultantResponseMode.TAIL_SPECIFIC.value: ResponseMode.TAIL_SPECIFIC,
    ConsultantResponseMode.INVALID_SANITY.value: ResponseMode.INVALID_SANITY,
}


def _to_legacy_router_result(
    r: ResponseModeRouterResult,
    *,
    query: str = "",
) -> ConsultantResponseRouterResult:
    legacy_mode = _LEGACY_FROM_ROUTER.get(r["mode"], ConsultantResponseMode.ADVISORY_MODE)
    if r["mode"] == ResponseMode.ADVISORY.value and DEAL_RE.search(query or ""):
        legacy_mode = ConsultantResponseMode.DEAL_ANALYSIS_MODE
    out: ConsultantResponseRouterResult = {
        "mode": legacy_mode.value,
        "reason": r["reason"],
        "visual_priority": r["visual_priority"],
        "verbosity": r["verbosity"],
    }
    return out


def route_consultant_response_mode(
    *,
    query: str,
    fine_intent: str,
    has_tail: bool,
    has_visual_intent: bool,
    suspicious_model_note: Optional[str],
    user_wants_gallery: bool = False,
    refinement_type: str = "none",
    standalone_confidence: float = 1.0,
    persistence_routing: str = "",
    memory_response_mode: str = "",
    has_conversation_anchor: bool = False,
) -> ConsultantResponseRouterResult:
    """Classify user intent for answer generation (legacy-shaped result)."""
    r = route_response_mode(
        query=query,
        fine_intent=fine_intent,
        has_tail=has_tail,
        has_visual_intent=has_visual_intent,
        user_wants_gallery=user_wants_gallery,
        suspicious_model_note=suspicious_model_note,
        refinement_type=refinement_type,
        standalone_confidence=standalone_confidence,
        persistence_routing=persistence_routing,
        memory_response_mode=memory_response_mode,
        has_conversation_anchor=has_conversation_anchor,
    )
    return _to_legacy_router_result(r, query=query)


def classify_consultant_response_mode(
    *,
    query: str,
    fine_intent: str,
    has_tail: bool,
    has_visual_intent: bool,
    suspicious_model_note: Optional[str],
    **kwargs: object,
) -> ConsultantResponseMode:
    r = route_consultant_response_mode(
        query=query,
        fine_intent=fine_intent,
        has_tail=has_tail,
        has_visual_intent=has_visual_intent,
        suspicious_model_note=suspicious_model_note,
        **kwargs,  # type: ignore[arg-type]
    )
    return ConsultantResponseMode(r["mode"])


def consultant_response_router_json(result: ConsultantResponseRouterResult) -> str:
    payload = {k: v for k, v in dict(result).items() if not str(k).startswith("_")}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def response_mode_prompt_suffix(mode: ConsultantResponseMode) -> str:
    """System-prompt suffix — delegates to canonical router prompts."""
    canonical = _ROUTER_FROM_LEGACY.get(mode.value, ResponseMode.ADVISORY)
    return _router_prompt_suffix(canonical)


def canonical_mode_from_legacy(mode: ConsultantResponseMode) -> ResponseMode:
    return _ROUTER_FROM_LEGACY.get(mode.value, ResponseMode.ADVISORY)


__all__ = [
    "ConsultantResponseMode",
    "ConsultantResponseRouterResult",
    "Verbosity",
    "canonical_mode_from_legacy",
    "classify_consultant_response_mode",
    "consultant_response_router_json",
    "response_mode_prompt_suffix",
    "route_consultant_response_mode",
]
