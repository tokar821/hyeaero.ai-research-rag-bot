"""
Phase 55 — mission profile must exist before aircraft recommendation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from services.broker_execution.broker_execution_category import (
    BrokerExecutionCategory,
    classify_broker_execution_category,
)
from services.broker_reasoning.mission_interpreter import interpret_mission


def _mission_analysis_required(query: str, *, data_used: Optional[dict] = None) -> bool:
    """Hard gate only for mission-category turns (not acquisition or tail)."""
    cat = classify_broker_execution_category(query, data_used=data_used)
    return cat == BrokerExecutionCategory.MISSION


def check_mission_profile_ready(
    query: str,
    *,
    data_used: Optional[dict] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Return (ready, mission_profile_dict).

    Requires route and passenger count before recommendation path proceeds.
    """
    if not _mission_analysis_required(query, data_used=data_used):
        return True, {}

    interp = interpret_mission(query or "")
    profile = interp.to_dict()
    profile["nonstop"] = bool(re.search(r"(?is)\bnonstop\b", query or ""))

    missing = []
    if interp.passengers is None:
        missing.append("passengers")
    if interp.route is None:
        missing.append("route")

    profile["missing_required_fields"] = missing
    data_used = data_used if isinstance(data_used, dict) else {}
    data_used["mission_profile"] = profile
    data_used["mission_profile_complete"] = len(missing) == 0

    return len(missing) == 0, profile


def mission_profile_clarification_answer(query: str, *, data_used: Optional[dict] = None) -> str:
    interp = interpret_mission(query or "")
    parts = [
        "Before I recommend an aircraft, I need a complete mission profile.",
    ]
    if interp.passengers is None:
        parts.append("How many passengers do you typically carry?")
    if interp.route is None:
        parts.append("What is the primary city pair or typical route (e.g. Boston to Denver)?")
    if interp.follow_up_questions:
        parts.extend(interp.follow_up_questions[:1])
    return "\n\n".join(parts).strip()


__all__ = [
    "check_mission_profile_ready",
    "mission_profile_clarification_answer",
]
