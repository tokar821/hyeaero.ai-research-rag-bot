"""Ownership risk factors — regulatory, utilization, program exposure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class OwnershipRiskAssessment:
    model: str
    risk_level: str  # low | moderate | elevated | unknown
    factors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "risk_level": self.risk_level,
            "factors": list(self.factors),
        }


def assess_ownership_risk(
    model: str,
    *,
    utilization_hours_per_year: float = 0,
    fractional_interest: bool = False,
) -> OwnershipRiskAssessment:
    factors: List[str] = []
    level = "low"
    if fractional_interest:
        factors.append("Fractional/part-share structures add program and exit constraints.")
        level = "moderate"
    if utilization_hours_per_year > 450:
        factors.append("High annual utilization accelerates maintenance reserves.")
        level = "moderate" if level == "low" else level
    if not factors:
        factors.append("No elevated ownership-risk flags from stated profile.")
    return OwnershipRiskAssessment(model=model, risk_level=level, factors=factors)
