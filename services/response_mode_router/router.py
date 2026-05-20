"""
Response Mode Router — classify each turn into a specialized answer shape.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from .schemas import ResponseMode, ResponseModeRouterResult, Verbosity
from .triggers import (
    ADVISORY_RE,
    COMPARISON_RE,
    CONVERSATION_ONLY_RE,
    DEAL_RE,
    EDUCATIONAL_RE,
    FOLLOWUP_RE,
    IMAGE_SHOWCASE_RE,
    VS_MODEL_RE,
)

_REFINEMENT_FOLLOWUP = frozenset(
    {
        "size_upgrade",
        "style_shift",
        "size_or_budget_down",
        "budget_shift",
        "lifestyle_inference",
        "sleeping_configuration",
        "view_change",
        "ambiguous_followup",
    }
)

_PERSISTENCE_FOLLOWUP_ROUTES = frozenset(
    {
        "inherit_context",
        "refinement_continuation",
        "image_showcase_continuation",
    }
)


def route_response_mode(
    *,
    query: str,
    fine_intent: str = "",
    has_tail: bool = False,
    has_visual_intent: bool = False,
    user_wants_gallery: bool = False,
    suspicious_model_note: Optional[str] = None,
    refinement_type: str = "none",
    standalone_confidence: float = 1.0,
    persistence_routing: str = "",
    memory_response_mode: str = "",
    has_conversation_anchor: bool = False,
) -> ResponseModeRouterResult:
    q = (query or "").strip()
    ql = q.lower()
    fi = (fine_intent or "").strip().lower()
    ref = (refinement_type or "none").strip().lower()
    pr = (persistence_routing or "").strip().lower()
    mem_rm = (memory_response_mode or "").strip().lower()

    def _out(
        mode: ResponseMode,
        reason: str,
        *,
        visual_priority: bool = False,
        verbosity: Verbosity = "short",
        inherit_context: bool = False,
        forbid_urls: bool = False,
        max_sentences: int = 8,
    ) -> ResponseModeRouterResult:
        if mode == ResponseMode.IMAGE_SHOWCASE:
            verbosity = "minimal"
            visual_priority = True
            forbid_urls = True
            max_sentences = 2
        elif mode == ResponseMode.FOLLOWUP_CONTINUATION:
            inherit_context = True
            verbosity = "short"
            max_sentences = 6
        elif mode == ResponseMode.COMPARISON_MODE:
            verbosity = "short"
            max_sentences = 8
        elif mode == ResponseMode.EDUCATIONAL_MODE:
            verbosity = "short"
            max_sentences = 10
        elif mode == ResponseMode.ADVISORY:
            verbosity = "short"
            max_sentences = 12

        return {
            "mode": mode.value,
            "reason": reason,
            "visual_priority": visual_priority,
            "verbosity": verbosity,
            "inherit_context": inherit_context,
            "forbid_urls_in_text": forbid_urls,
            "max_sentences_hint": max_sentences,
        }

    if (suspicious_model_note or "").strip():
        return _out(ResponseMode.INVALID_SANITY, "suspicious_or_nonexistent_aircraft_model")

    try:
        from services.intent_persistence.pivot import is_visual_budget_shopping_pivot

        if is_visual_budget_shopping_pivot(q):
            return _out(
                ResponseMode.IMAGE_SHOWCASE,
                "budget_cabin_shopping_pivot",
                inherit_context=False,
            )
    except Exception:
        pass

    if len(q) < 80 and CONVERSATION_ONLY_RE.search(q) and not re.search(
        r"\b(jet|aircraft|n\d|citation|gulfstream)\b", ql, re.I
    ):
        return _out(ResponseMode.ADVISORY, "brief_social_turn", verbosity="short", max_sentences=3)

    is_followup = bool(
        has_conversation_anchor
        and (
            ref in _REFINEMENT_FOLLOWUP
            or pr in _PERSISTENCE_FOLLOWUP_ROUTES
            or float(standalone_confidence) < 0.52
            or (FOLLOWUP_RE.search(q) and len(q) < 140)
        )
    )
    is_visual = bool(
        user_wants_gallery
        or has_visual_intent
        or IMAGE_SHOWCASE_RE.search(q)
        or mem_rm in ("image_showcase", "visual_only", "short_caption")
    )
    is_comparison = bool(
        (fi == "aircraft_comparison" or COMPARISON_RE.search(q) or VS_MODEL_RE.search(q))
        and not (is_followup and ref in _REFINEMENT_FOLLOWUP)
        and not IMAGE_SHOWCASE_RE.search(q)
    )
    is_educational = bool(EDUCATIONAL_RE.search(q) and not is_comparison)
    is_advisory = bool(ADVISORY_RE.search(q) or DEAL_RE.search(q) or fi in ("aircraft_recommendation", "aviation_mission"))

    if has_tail and is_visual:
        return _out(
            ResponseMode.TAIL_SPECIFIC,
            "tail_locked_visual_gallery",
            visual_priority=True,
            forbid_urls=True,
            max_sentences=2,
            inherit_context=True,
        )

    if is_comparison and not (is_followup and not COMPARISON_RE.search(q)):
        return _out(ResponseMode.COMPARISON_MODE, "explicit_comparison")

    if is_educational and not is_visual:
        return _out(ResponseMode.EDUCATIONAL_MODE, "explain_or_how_why")

    # Visual beats generic follow-up when user names a view facet (cockpit, interior, show me).
    if is_visual and (IMAGE_SHOWCASE_RE.search(q) or user_wants_gallery or ref == "view_change"):
        return _out(
            ResponseMode.IMAGE_SHOWCASE,
            "visual_gallery_primary",
            inherit_context=is_followup or has_conversation_anchor,
        )

    if is_followup and not is_educational:
        mode = ResponseMode.FOLLOWUP_CONTINUATION
        if is_visual or mem_rm in ("image_showcase", "short_caption"):
            return _out(
                ResponseMode.IMAGE_SHOWCASE,
                "followup_with_visual_continuation",
                inherit_context=True,
            )
        return _out(mode, f"contextual_refinement:{ref or pr or 'deictic'}")

    if is_visual:
        return _out(
            ResponseMode.IMAGE_SHOWCASE,
            "visual_intent",
            inherit_context=has_conversation_anchor,
        )

    if is_advisory:
        reason = "deal_or_recommendation" if DEAL_RE.search(q) else "advisory_decision_support"
        return _out(ResponseMode.ADVISORY, reason)

    if has_tail:
        return _out(ResponseMode.TAIL_SPECIFIC, "registration_lookup", inherit_context=True)

    if fi in ("aircraft_comparison",):
        return _out(ResponseMode.COMPARISON_MODE, f"fine_intent:{fi}")

    return _out(ResponseMode.ADVISORY, "default_luxury_consultant")


def router_result_json(result: ResponseModeRouterResult) -> str:
    return json.dumps(dict(result), ensure_ascii=False, sort_keys=True)
