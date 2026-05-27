"""Feasibility engine I/O schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AircraftFeasibilityVerdict:
    """
    Hard feasibility verdict for one aircraft against one mission.

    Runs before any LLM recommendation — impossible aircraft are rejected entirely.
    """

    feasible: bool
    rejection_reasons: List[str] = field(default_factory=list)
    payload_penalty: float = 0.0
    runway_penalty: float = 0.0
    winter_penalty: float = 0.0
    reserves_satisfied: bool = True

    # Audit fields (optional downstream)
    required_nm: float = 0.0
    available_nm: float = 0.0
    margin_nm: float = 0.0
    stage_distance_nm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feasible": self.feasible,
            "rejectionReasons": list(self.rejection_reasons),
            "payloadPenalty": round(self.payload_penalty, 1),
            "runwayPenalty": round(self.runway_penalty, 1),
            "winterPenalty": round(self.winter_penalty, 1),
            "reservesSatisfied": self.reserves_satisfied,
            "requiredNm": round(self.required_nm, 1),
            "availableNm": round(self.available_nm, 1),
            "marginNm": round(self.margin_nm, 1),
            "stageDistanceNm": round(self.stage_distance_nm, 1),
        }

    @classmethod
    def rejected(
        cls,
        *reasons: str,
        **kwargs: Any,
    ) -> "AircraftFeasibilityVerdict":
        return cls(
            feasible=False,
            rejection_reasons=list(reasons),
            reserves_satisfied=kwargs.get("reserves_satisfied", True),
            payload_penalty=float(kwargs.get("payload_penalty", 0.0)),
            runway_penalty=float(kwargs.get("runway_penalty", 0.0)),
            winter_penalty=float(kwargs.get("winter_penalty", 0.0)),
            required_nm=float(kwargs.get("required_nm", 0.0)),
            available_nm=float(kwargs.get("available_nm", 0.0)),
            margin_nm=float(kwargs.get("margin_nm", 0.0)),
            stage_distance_nm=float(kwargs.get("stage_distance_nm", 0.0)),
        )
