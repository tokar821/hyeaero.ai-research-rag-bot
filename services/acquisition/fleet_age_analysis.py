"""Fleet age analysis — vintage concentration and replacement pressure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class FleetAgeAnalysis:
    model: str
    vintage_year: Optional[int]
    age_years: Optional[int]
    segment_posture: str  # current | aging | legacy | unknown
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "vintage_year": self.vintage_year,
            "age_years": self.age_years,
            "segment_posture": self.segment_posture,
            "notes": list(self.notes),
        }


def analyze_fleet_age(
    model: str,
    *,
    vintage_year: Optional[int] = None,
    reference_year: int = 2026,
) -> FleetAgeAnalysis:
    notes: List[str] = []
    age = None
    posture = "unknown"
    if vintage_year:
        age = max(0, reference_year - vintage_year)
        if age <= 8:
            posture = "current"
            notes.append("Current-generation vintage for segment.")
        elif age <= 15:
            posture = "aging"
            notes.append("Mid-life vintage — verify avionics and engine programs.")
        else:
            posture = "legacy"
            notes.append("Legacy vintage — resale and maintenance exposure increase.")
    else:
        notes.append("Vintage not stated — age analysis requires serial/year.")
    return FleetAgeAnalysis(
        model=model,
        vintage_year=vintage_year,
        age_years=age,
        segment_posture=posture,
        notes=notes,
    )
