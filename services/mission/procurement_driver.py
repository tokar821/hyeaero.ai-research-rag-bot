"""
Procurement-driver reasoning — what should and should not drive aircraft selection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.mission.center_of_gravity import detect_center_of_gravity

_VANITY_RE = re.compile(
    r"\b(?:ceo\s+mandate|founder\s+ulr|prestige|flagship|showpiece)\b",
    re.I,
)
_ONE_AIRCRAFT_RE = re.compile(r"\b(?:one\s+aircraft\s+only|single\s+aircraft|only\s+one\s+jet)\b", re.I)
_EDGE_TRAP_RE = re.compile(
    r"\b(?:guaranteed\s+nonstop|must\s+do\s+everything|every\s+mission)\b",
    re.I,
)
_OPTIMIZED_EPISODIC_RE = re.compile(
    r"\boptimized\s+around\s+(?:tokyo|singapore|dubai|sydney|london)\b|"
    r"longest\s+route\b.*\b(?:procure|optimiz)",
    re.I,
)


@dataclass
class ProcurementDriverResult:
    true_dominant_utilization: str = ""
    episodic_missions: List[str] = field(default_factory=list)
    vanity_mission_risk: bool = False
    one_aircraft_fantasy: bool = False
    edge_case_trap: bool = False
    fleet_distortion: bool = False
    guidance: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "true_dominant_utilization": self.true_dominant_utilization,
            "episodic_missions": list(self.episodic_missions),
            "vanity_mission_risk": self.vanity_mission_risk,
            "one_aircraft_fantasy": self.one_aircraft_fantasy,
            "edge_case_trap": self.edge_case_trap,
            "fleet_distortion": self.fleet_distortion,
            "guidance": list(self.guidance),
        }


def analyze_procurement_drivers(
    query: str,
    mission: Any = None,
    *,
    data_used: Optional[Dict[str, Any]] = None,
) -> ProcurementDriverResult:
    cog = detect_center_of_gravity(query, mission)
    result = ProcurementDriverResult(
        true_dominant_utilization=cog.procurement_driver or cog.dominant_band,
        episodic_missions=list(cog.episodic_nodes),
    )

    ql = (query or "").lower()
    if _VANITY_RE.search(ql):
        result.vanity_mission_risk = True
        result.guidance.append("Vanity / prestige language detected — separate from utilization economics.")
    if _ONE_AIRCRAFT_RE.search(ql):
        result.one_aircraft_fantasy = True
        result.guidance.append("Single-aircraft mandate across incompatible domains is structurally risky.")
    if _EDGE_TRAP_RE.search(ql):
        result.edge_case_trap = True
        result.guidance.append("Edge-case optimization should not drive primary procurement.")

    if _OPTIMIZED_EPISODIC_RE.search(ql):
        result.fleet_distortion = True
        result.guidance.append(
            "Optimizing procurement around the longest/episodic leg distorts the domestic utilization center."
        )
    if cog.episodic_distortion_risk and any(
        x in ql for x in ("tokyo", "singapore", "dubai", "sydney", "g700", "global 7500")
    ):
        result.fleet_distortion = True
        result.guidance.append(
            "Occasional ULR continuation is distorting fleet logic vs domestic center of gravity."
        )

    if isinstance(data_used, dict):
        data_used["procurement_driver_analysis"] = result.to_dict()

    return result


__all__ = ["ProcurementDriverResult", "analyze_procurement_drivers"]
