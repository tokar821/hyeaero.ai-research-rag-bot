"""Structured mission object from natural language."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.broker_reasoning.mission_interpreter import interpret_mission


@dataclass
class MissionFrame:
    pax: Optional[int] = None
    range_nm: Optional[int] = None
    mission_type: str = "UNKNOWN"
    urgency: str = "NORMAL"
    ownership_stage: str = "EXPLORING"
    route: Optional[str] = None
    missing_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pax": self.pax,
            "range_nm": self.range_nm,
            "mission_type": self.mission_type,
            "urgency": self.urgency,
            "ownership_stage": self.ownership_stage,
            "route": self.route,
            "missing_fields": list(self.missing_fields),
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> Optional[MissionFrame]:
        if not isinstance(raw, dict):
            return None
        return cls(
            pax=raw.get("pax"),
            range_nm=raw.get("range_nm"),
            mission_type=str(raw.get("mission_type") or "UNKNOWN"),
            urgency=str(raw.get("urgency") or "NORMAL"),
            ownership_stage=str(raw.get("ownership_stage") or "EXPLORING"),
            route=raw.get("route"),
            missing_fields=list(raw.get("missing_fields") or []),
        )


_TIMING_RE = re.compile(
    r"(?is)\b(?:buy\s+now|wait|timing|rising\s+prices|market\s+trend)\b",
)
_SHOPPING_RE = re.compile(r"(?is)\b(?:listing|saw|found|tail|registration|due\s+diligence)\b")
_ACQUISITION_RE = re.compile(
    r"(?is)\b(?:buy|purchase|acquire|what\s+should\s+i\s+buy|best\s+jet)\b",
)


def build_mission_frame(
    query: str,
    *,
    client_context: Optional[Dict[str, Any]] = None,
) -> MissionFrame:
    """Convert NL query (+ optional context hints) into MissionFrame."""
    q = (query or "").strip()
    interp = interpret_mission(q)
    ctx = client_context if isinstance(client_context, dict) else {}

    mission_type = "UNKNOWN"
    if _SHOPPING_RE.search(q):
        mission_type = "LISTING_REVIEW"
    elif _ACQUISITION_RE.search(q):
        mission_type = "ACQUISITION"
    elif interp.range_nm or interp.route:
        mission_type = "MISSION_PROFILE"

    urgency = "ELEVATED" if _TIMING_RE.search(q) else "NORMAL"

    ownership_stage = str(ctx.get("stage") or "EXPLORING")
    if ownership_stage not in (
        "EXPLORING",
        "ACTIVE_SHOPPING",
        "NEGOTIATING",
        "DUE_DILIGENCE",
    ):
        ownership_stage = "EXPLORING"

    missing = list(interp.missing_fields)
    if mission_type == "ACQUISITION" and interp.acquisition_budget_musd is None:
        if "acquisition_budget" not in missing:
            missing.append("acquisition_budget")

    return MissionFrame(
        pax=interp.passengers,
        range_nm=interp.range_nm,
        mission_type=mission_type,
        urgency=urgency,
        ownership_stage=ownership_stage,
        route=interp.route,
        missing_fields=missing,
    )


def mission_budget_musd(mission: MissionFrame, query: str) -> Optional[float]:
    """Budget from mission interpreter only (not client memory)."""
    interp = interpret_mission(query or "")
    return interp.acquisition_budget_musd


__all__ = ["MissionFrame", "build_mission_frame", "mission_budget_musd"]
