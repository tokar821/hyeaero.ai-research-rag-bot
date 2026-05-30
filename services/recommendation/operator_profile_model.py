"""
Operator profile model — infer operator type and procurement posture from mission signals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

_EMPLOYEES_RE = re.compile(r"\b(\d{3,5})\s+employees?\b", re.I)
_ENERGY_RE = re.compile(
    r"\b(?:oil|gas|mining|logistics|extraction|drilling|offshore|arctic|gravel)\b",
    re.I,
)
_UHNW_RE = re.compile(
    r"\b(?:family\s+office|uhn?w|personal\s+use|founder|principal)\b",
    re.I,
)
_ENTERPRISE_RE = re.compile(
    r"\b(?:fortune|global\s+enterprise|multinational|public\s+company)\b",
    re.I,
)


class OperatorType(str, Enum):
    FOUNDER_LED = "founder_led"
    MIDSIZE_PUBLIC = "midsize_public"
    GLOBAL_ENTERPRISE = "global_enterprise"
    ENERGY_LOGISTICS = "energy_logistics"
    UHNW_PERSONAL = "uhnw_personal"
    REGIONAL_OPERATOR = "regional_operator"
    UNKNOWN = "unknown"


@dataclass
class OperatorProfile:
    operator_type: OperatorType = OperatorType.UNKNOWN
    employee_count: Optional[int] = None
    geographic_spread: str = "regional"
    utilization_style: str = "mixed"
    cabin_expectation: str = "executive"
    dispatch_tolerance: str = "standard"
    redundancy_expectation: str = "standard"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operator_type": self.operator_type.value,
            "employee_count": self.employee_count,
            "geographic_spread": self.geographic_spread,
            "utilization_style": self.utilization_style,
            "cabin_expectation": self.cabin_expectation,
            "dispatch_tolerance": self.dispatch_tolerance,
            "redundancy_expectation": self.redundancy_expectation,
            "notes": list(self.notes),
        }


def infer_operator_profile(
    mission: Any,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> OperatorProfile:
    ql = (query or "").lower()
    routes = list(getattr(mission, "routes", None) or [])
    pax = int(getattr(mission, "passenger_count", None) or 0)
    profile = OperatorProfile()

    m = _EMPLOYEES_RE.search(query or "")
    if m:
        profile.employee_count = int(m.group(1))
        if profile.employee_count >= 5000:
            profile.operator_type = OperatorType.GLOBAL_ENTERPRISE
            profile.geographic_spread = "global"
            profile.cabin_expectation = "boardroom"
            profile.redundancy_expectation = "high"
            profile.notes.append("Large employee base — enterprise dispatch and cabin standards.")
        elif profile.employee_count >= 800:
            profile.operator_type = OperatorType.MIDSIZE_PUBLIC
            profile.geographic_spread = "multi_region"
            profile.cabin_expectation = "executive"
            profile.notes.append("Mid-size corporate operator — balanced cabin and cost discipline.")

    if _ENERGY_RE.search(ql):
        profile.operator_type = OperatorType.ENERGY_LOGISTICS
        profile.utilization_style = "field_access"
        profile.dispatch_tolerance = "runway_first"
        profile.notes.append("Energy/logistics profile — field performance dominates cabin prestige.")

    if _UHNW_RE.search(ql):
        profile.operator_type = OperatorType.UHNW_PERSONAL
        profile.cabin_expectation = "premium"
        profile.notes.append("UHNW / personal use — cabin and schedule flexibility weighted heavily.")

    if _ENTERPRISE_RE.search(ql):
        profile.operator_type = OperatorType.GLOBAL_ENTERPRISE
        profile.geographic_spread = "global"

    intl_hubs = sum(
        1
        for r in routes
        for hub in ("london", "paris", "dubai", "singapore", "tokyo", "hong kong")
        if hub in r.lower()
    )
    if intl_hubs >= 2:
        profile.geographic_spread = "global"
    elif intl_hubs == 1:
        profile.geographic_spread = "multi_region"

    if pax >= 12:
        profile.cabin_expectation = "boardroom"
    elif pax >= 8:
        profile.cabin_expectation = "executive"

    if profile.operator_type == OperatorType.UNKNOWN:
        if any(c in ql for c in ("dallas", "houston", "chicago", "atlanta")) and intl_hubs <= 1:
            profile.operator_type = OperatorType.REGIONAL_OPERATOR
            profile.utilization_style = "domestic_core"
        elif _UHNW_RE.search(ql) or pax <= 6:
            profile.operator_type = OperatorType.FOUNDER_LED
        else:
            profile.operator_type = OperatorType.MIDSIZE_PUBLIC

    if isinstance(data_used, dict):
        pkt = data_used.get("mission_understanding_packet") or {}
        if isinstance(pkt, dict):
            ic = pkt.get("inferred_constraints") or {}
            if ic.get("domestic_utilization_dominant"):
                profile.utilization_style = "domestic_core"

    return profile


__all__ = ["OperatorType", "OperatorProfile", "infer_operator_profile"]
