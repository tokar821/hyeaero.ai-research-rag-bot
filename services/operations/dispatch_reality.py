"""
Dispatch reality — technically possible vs operationally dependable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from services.consultant.mission_state import MissionState
from services.mission.models import MissionProfile


@dataclass
class DispatchRealityVerdict:
    technically_possible: bool
    operationally_dependable: bool
    broker_label: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "technically_possible": self.technically_possible,
            "operationally_dependable": self.operationally_dependable,
            "broker_label": self.broker_label,
            "explanation": self.explanation,
        }


def assess_dispatch_reality(
    model: str,
    profile: Dict[str, Any],
    mission: Any,
    *,
    query: str = "",
    operational_context: Any = None,
) -> DispatchRealityVerdict:
    """
    Broker-facing dispatch verdict separating brochure feasibility from dependable dispatch.
    """
    if operational_context is not None:
        try:
            from services.operational.mission_operational_assessment import assess_aircraft_operational

            assessment = assess_aircraft_operational(model, profile, operational_context)
            tech = assessment.dispatch.technically_possible
            dependable = assessment.dispatch.works_reliably
        except Exception:
            tech, dependable = True, False
    else:
        margin = float(profile.get("practical_nm") or 0) - 500.0
        tech = margin > 0
        dependable = margin > 400.0

    if dependable:
        label = "OPERATIONALLY DEPENDABLE"
        expl = "Reserve, payload, and seasonal margins support dependable dispatch on this corridor."
    elif tech:
        label = "TECHNICALLY POSSIBLE — MARGINAL DISPATCH"
        expl = (
            "The stage may close on paper, but winter/westbound or payload pressure makes "
            "dependable dispatch uncertain for executive operations."
        )
    else:
        label = "NOT DISPATCH-RELIABLE"
        expl = "Effective range with reserves does not support dependable nonstop operations as stated."

    return DispatchRealityVerdict(
        technically_possible=tech,
        operationally_dependable=dependable,
        broker_label=label,
        explanation=expl,
    )


__all__ = ["DispatchRealityVerdict", "assess_dispatch_reality"]
