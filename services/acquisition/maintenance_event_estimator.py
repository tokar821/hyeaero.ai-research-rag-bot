"""Maintenance event horizon estimator for acquisition planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class MaintenanceEventEstimate:
    model: str
    horizon_months: int
    events: List[str] = field(default_factory=list)
    reserve_pressure: str  # low | moderate | high | unknown

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "horizon_months": self.horizon_months,
            "events": list(self.events),
            "reserve_pressure": self.reserve_pressure,
        }


def estimate_maintenance_events(
    model: str,
    *,
    airframe_hours: float = 0,
    months_owned: int = 0,
) -> MaintenanceEventEstimate:
    events: List[str] = []
    pressure = "low"
    if airframe_hours > 4000:
        events.append("Approaching major inspection intervals — verify program status.")
        pressure = "moderate"
    if months_owned > 60:
        events.append("Paint/interior refresh cycles may affect near-term capex.")
        pressure = "moderate" if pressure == "low" else pressure
    if not events:
        events.append("No cataloged near-term events from hours alone — verify logbooks.")
    return MaintenanceEventEstimate(
        model=model,
        horizon_months=24,
        events=events,
        reserve_pressure=pressure,
    )
