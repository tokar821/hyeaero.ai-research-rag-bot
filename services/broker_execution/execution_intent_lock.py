"""
Execution intent lock — routing must obey classified intent.

Stamps ``data_used`` flags so downstream layers do not run acquisition/market/broker overlays
on registry-only turns.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, Optional

from services.broker_execution.tail_depth_mode import (
    TailDepthMode,
    classify_tail_depth_mode,
    llm_required_depths,
    registry_template_depths,
)


class ExecutionProfile(str, Enum):
    TAIL_OWNER = "tail_owner"
    TAIL_SALE_STATUS = "tail_sale_status"
    TAIL_SUMMARY = "tail_summary"
    TAIL_DETAIL = "tail_detail"
    TAIL_ACQUISITION = "tail_acquisition"
    TAIL_ENGINE_PROGRAM = "tail_engine_program"
    TAIL_IMAGES = "tail_images"
    MISSION = "mission"
    COMPARISON = "comparison"
    LISTING = "listing"
    GENERAL = "general"


_MISSION_RE = re.compile(
    r"(?is)\b(?:\d+\s*(?:pax|passengers?)\b|nonstop|coast.?to.?coast|"
    r"(?:from|between)\s+.+?\s+to\s+.+|under\s+\$\d|budget\s+of\s+\$)\b"
)
_COMPARISON_RE = re.compile(r"(?is)\b(?:\bvs\.?\b|versus|compare\s+)\b")
_LISTING_RE = re.compile(
    r"(?is)\b(?:listed\s+at|asking\s+\$|\d{4}\s+\w+\s+listed|is\s+this\s+(?:price|ask)\s+normal)\b"
)


def resolve_execution_profile(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> ExecutionProfile:
    q = (query or "").strip()
    du = data_used if isinstance(data_used, dict) else {}

    depth, reg = classify_tail_depth_mode(q)
    if reg:
        if depth == TailDepthMode.OWNER:
            return ExecutionProfile.TAIL_OWNER
        if depth == TailDepthMode.SALE_STATUS:
            return ExecutionProfile.TAIL_SALE_STATUS
        if depth == TailDepthMode.DETAIL:
            return ExecutionProfile.TAIL_DETAIL
        if depth in (TailDepthMode.ACQUISITION, TailDepthMode.ACQUISITION_RISKS):
            return ExecutionProfile.TAIL_ACQUISITION
        if depth == TailDepthMode.ENGINE_PROGRAM:
            return ExecutionProfile.TAIL_ENGINE_PROGRAM
        if depth == TailDepthMode.IMAGES:
            return ExecutionProfile.TAIL_IMAGES
        if depth == TailDepthMode.COMPARISON:
            return ExecutionProfile.COMPARISON
        if depth in (TailDepthMode.MARKET_PRICE,):
            return ExecutionProfile.LISTING
        if depth == TailDepthMode.SUMMARY:
            return ExecutionProfile.TAIL_SUMMARY
        if depth in llm_required_depths():
            return ExecutionProfile.GENERAL

    if du.get("deterministic_pre_llm_executed") or du.get("pipeline_llm_facts"):
        return ExecutionProfile.MISSION
    if _COMPARISON_RE.search(q):
        return ExecutionProfile.COMPARISON
    if _LISTING_RE.search(q):
        return ExecutionProfile.LISTING
    if _MISSION_RE.search(q):
        return ExecutionProfile.MISSION

    qri = str(du.get("query_recommendation_intent") or "").strip().lower()
    if qri in ("aircraft_recommendation", "mission_feasibility", "shortlist_ranking"):
        return ExecutionProfile.MISSION
    if qri == "aircraft_comparison":
        return ExecutionProfile.COMPARISON

    return ExecutionProfile.GENERAL


def attach_execution_intent_lock(
    data_used: Optional[Dict[str, Any]],
    query: str,
) -> ExecutionProfile:
    du = data_used if isinstance(data_used, dict) else {}
    profile = resolve_execution_profile(query, du)
    depth, reg = classify_tail_depth_mode(query)

    du["execution_profile"] = profile.value
    du["tail_depth_mode"] = depth.value if depth != TailDepthMode.NONE else None
    if reg:
        du["tail_registration"] = reg

    reg_card = registry_template_depths()
    depth_enum = depth if depth != TailDepthMode.NONE else None
    try:
        if depth_name := str(du.get("tail_depth_mode") or ""):
            depth_enum = TailDepthMode(depth_name)
    except ValueError:
        pass

    du["suppress_broker_reasoning_overlay"] = int(
        profile in (ExecutionProfile.TAIL_OWNER, ExecutionProfile.TAIL_SALE_STATUS)
        or (depth_enum in reg_card if depth_enum else False)
    )
    du["suppress_market_authority_block"] = int(
        profile in (ExecutionProfile.TAIL_OWNER, ExecutionProfile.TAIL_SALE_STATUS)
    )
    du["suppress_acquisition_tail_rewrite"] = int(
        profile in (ExecutionProfile.TAIL_OWNER, ExecutionProfile.TAIL_SALE_STATUS)
    )
    du["question_first_required"] = int(
        profile in (ExecutionProfile.TAIL_OWNER, ExecutionProfile.TAIL_SALE_STATUS)
    )
    du["tail_llm_narration_required"] = int(
        profile
        in (
            ExecutionProfile.TAIL_ACQUISITION,
            ExecutionProfile.TAIL_DETAIL,
            ExecutionProfile.TAIL_ENGINE_PROGRAM,
            ExecutionProfile.GENERAL,
            ExecutionProfile.COMPARISON,
            ExecutionProfile.LISTING,
        )
        or (depth_enum in llm_required_depths() if depth_enum else False)
    )
    du["mission_reasoning_required"] = int(profile == ExecutionProfile.MISSION)
    du["suppress_executive_broker_layer"] = int(
        profile
        in (
            ExecutionProfile.MISSION,
            ExecutionProfile.COMPARISON,
            ExecutionProfile.TAIL_OWNER,
            ExecutionProfile.TAIL_SALE_STATUS,
        )
    )
    du["suppress_template_post_layers"] = int(
        profile in (ExecutionProfile.MISSION, ExecutionProfile.COMPARISON, ExecutionProfile.TAIL_OWNER, ExecutionProfile.TAIL_SALE_STATUS)
    )

    return profile


def should_skip_broker_reasoning_layer(data_used: Optional[Dict[str, Any]]) -> bool:
    du = data_used if isinstance(data_used, dict) else {}
    return bool(du.get("suppress_broker_reasoning_overlay"))


def should_skip_market_block(data_used: Optional[Dict[str, Any]]) -> bool:
    du = data_used if isinstance(data_used, dict) else {}
    return bool(du.get("suppress_market_authority_block"))


__all__ = [
    "ExecutionProfile",
    "attach_execution_intent_lock",
    "resolve_execution_profile",
    "should_skip_broker_reasoning_layer",
    "should_skip_market_block",
]
