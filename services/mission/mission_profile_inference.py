"""
Mission profile inference — utilization style and operational philosophy without extra questions.

Inferred signals influence elimination bands and ranking weights; they do not trigger
clarification when corridor + route are already sufficient.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.mission.models import MissionProfile, PriorityLevel


class UtilizationStyle:
    EXECUTIVE_SHUTTLE = "executive_shuttle"
    FAMILY_LEISURE = "family_leisure"
    OWNER_FLOWN = "owner_flown"
    BOARD_TRANSPORT = "board_transport"
    CHARTER_OFFSET = "charter_offset"
    MIXED_CORPORATE = "mixed_corporate"
    UNKNOWN = "unknown"


@dataclass
class InferredMissionProfile:
    utilization_style: str = UtilizationStyle.UNKNOWN
    cost_sensitive: bool = False
    dispatch_priority: bool = False
    airport_access_priority: bool = False
    nonstop_preference: bool = False
    tech_stop_tolerance: str = "unknown"  # none | limited | flexible
    cabin_priority_inferred: bool = False
    runway_priority_inferred: bool = False
    confidence: float = 0.0
    signals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "utilization_style": self.utilization_style,
            "cost_sensitive": self.cost_sensitive,
            "dispatch_priority": self.dispatch_priority,
            "airport_access_priority": self.airport_access_priority,
            "nonstop_preference": self.nonstop_preference,
            "tech_stop_tolerance": self.tech_stop_tolerance,
            "cabin_priority_inferred": self.cabin_priority_inferred,
            "runway_priority_inferred": self.runway_priority_inferred,
            "confidence": round(self.confidence, 3),
            "signals": list(self.signals),
        }


_EXECUTIVE_RE = re.compile(
    r"\b(?:executives?|board|ceo|cfo|roadshow|client\s+meetings?)\b",
    re.I,
)
_FAMILY_RE = re.compile(
    r"\b(?:family|kids|children|vacation|ski\s+trip|leisure)\b",
    re.I,
)
_OWNER_FLOWN_RE = re.compile(r"\b(?:owner[- ]flown|flying\s+myself|pilot\s+owner)\b", re.I)
_CHARTER_RE = re.compile(r"\b(?:charter|on[- ]demand|offset\s+hours?)\b", re.I)
_COST_RE = re.compile(
    r"\b(?:lowest\s+cost|minimize\s+cost|cost[- ]sensitive|operating\s+cost|"
    r"cheaper\s+to\s+operate|doc)\b",
    re.I,
)
_DISPATCH_RE = re.compile(
    r"\b(?:dispatch|reliability|availability|no\s+delays?|schedule\s+critical)\b",
    re.I,
)
_RUNWAY_RE = re.compile(
    r"\b(?:short\s+field|runway\s+flex|aspen|telluride|mountain|hot[- ]and[- ]high)\b",
    re.I,
)
_NONSTOP_RE = re.compile(r"\b(?:nonstop|non[- ]stop|direct\s+only)\b", re.I)
_TECH_STOP_OK_RE = re.compile(
    r"\b(?:tech\s+stop|fuel\s+stop|gander|keflavik|one\s+stop)\b",
    re.I,
)
_CABIN_RE = re.compile(
    r"\b(?:stand[- ]up|cabin|baggage|comfort|luggage)\b",
    re.I,
)
_RUNWAY_OVER_LUXURY_RE = re.compile(
    r"\brunway\s+flexibility\s+over\s+luxury\b"
    r"|\b(?:runway|field)\s+(?:flex|flexibility|access)\b.*\b(?:over|than|vs)\b.*\b(?:luxury|cabin)\b",
    re.I,
)


def infer_mission_profile(
    query: str,
    profile: Optional[MissionProfile] = None,
    *,
    broker_memory: Optional[Dict[str, Any]] = None,
) -> InferredMissionProfile:
    """
    Infer mission philosophy from query + extracted profile + optional broker memory.
    """
    ql = (query or "").strip()
    out = InferredMissionProfile()
    score = 0.0

    if _EXECUTIVE_RE.search(ql):
        out.utilization_style = UtilizationStyle.EXECUTIVE_SHUTTLE
        out.signals.append("executive_shuttle")
        score += 0.25
    elif _FAMILY_RE.search(ql):
        out.utilization_style = UtilizationStyle.FAMILY_LEISURE
        out.signals.append("family_leisure")
        score += 0.25
    elif _OWNER_FLOWN_RE.search(ql):
        out.utilization_style = UtilizationStyle.OWNER_FLOWN
        out.signals.append("owner_flown")
        score += 0.25
    elif _CHARTER_RE.search(ql):
        out.utilization_style = UtilizationStyle.CHARTER_OFFSET
        out.signals.append("charter_offset")
        score += 0.2

    if profile and profile.passengers and profile.passengers >= 10:
        out.utilization_style = UtilizationStyle.BOARD_TRANSPORT
        out.signals.append("board_transport_pax")
        score += 0.15
    if profile and profile.passengers and 6 <= profile.passengers <= 9:
        if out.utilization_style == UtilizationStyle.UNKNOWN:
            out.utilization_style = UtilizationStyle.EXECUTIVE_SHUTTLE
            out.signals.append("executive_pax_load")
        score += 0.2

    if _RUNWAY_OVER_LUXURY_RE.search(ql):
        out.runway_priority_inferred = True
        out.airport_access_priority = True
        out.signals.append("runway_over_luxury")
        score += 0.15
        # Runway priority — not utility/cost-minimization posture.
        out.cost_sensitive = False

    if _COST_RE.search(ql):
        out.cost_sensitive = True
        out.signals.append("cost_sensitive")
        score += 0.2
    if _DISPATCH_RE.search(ql):
        out.dispatch_priority = True
        out.signals.append("dispatch_priority")
        score += 0.2

    if _RUNWAY_RE.search(ql) or (profile and (
        profile.mountain_airport_priority or profile.mountain_airports
    )):
        out.airport_access_priority = True
        out.runway_priority_inferred = True
        out.signals.append("airport_access")
        score += 0.2

    if _NONSTOP_RE.search(ql) or (profile and profile.nonstop_required):
        out.nonstop_preference = True
        out.signals.append("nonstop_preference")
        score += 0.15

    if _TECH_STOP_OK_RE.search(ql):
        out.tech_stop_tolerance = "flexible"
        out.signals.append("tech_stop_ok")
        score += 0.1
    elif out.nonstop_preference:
        out.tech_stop_tolerance = "none"

    if _CABIN_RE.search(ql):
        out.cabin_priority_inferred = True
        out.signals.append("cabin_priority")
        score += 0.1

    if isinstance(broker_memory, dict):
        if broker_memory.get("nonstop_preference"):
            out.nonstop_preference = True
        if broker_memory.get("runway_flexibility_priority") in ("high", "medium"):
            out.runway_priority_inferred = True
            out.airport_access_priority = True
        if broker_memory.get("cost_sensitivity") in ("high", "medium"):
            out.cost_sensitive = True
        if broker_memory.get("dispatch_priority") in ("high", "medium"):
            out.dispatch_priority = True
        score = min(1.0, score + 0.15)

    out.confidence = min(1.0, score)
    return out


def apply_inference_to_profile(
    profile: MissionProfile,
    inferred: InferredMissionProfile,
) -> MissionProfile:
    """Merge inferred priorities into mission profile (current turn wins on explicit fields)."""
    if inferred.cost_sensitive and profile.operating_cost_priority == PriorityLevel.NONE:
        profile.operating_cost_priority = PriorityLevel.HIGH
    if inferred.runway_priority_inferred and profile.runway_priority == PriorityLevel.NONE:
        profile.runway_priority = PriorityLevel.HIGH
        profile.short_field_priority = PriorityLevel.HIGH
    if inferred.cabin_priority_inferred and profile.cabin_priority == PriorityLevel.NONE:
        profile.cabin_priority = PriorityLevel.MEDIUM
    if inferred.nonstop_preference:
        profile.nonstop_required = True
    return profile
