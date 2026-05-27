"""
Acquisition intelligence bundle — liquidity, maintenance, OEM support, fleet age, ownership risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.acquisition.fleet_age_analysis import analyze_fleet_age
from services.acquisition.maintenance_event_estimator import estimate_maintenance_events
from services.acquisition.market_liquidity import assess_market_liquidity
from services.acquisition.ownership_risk import assess_ownership_risk
from services.acquisition.resale_strength import assess_resale_strength

# OEM support posture (catalog stub — serial-specific in production)
_OEM_SUPPORT: Dict[str, str] = {
    "g650": "strong",
    "g650er": "strong",
    "global 7500": "strong",
    "challenger 350": "strong",
    "citation latitude": "solid",
    "phenom 300e": "solid",
    "pc-24": "solid",
}


@dataclass
class AcquisitionIntelligenceBundle:
    model: str
    liquidity_tier: str = "unknown"
    resale_strength: str = "unknown"
    ownership_risk: str = "unknown"
    maintenance_pressure: str = "unknown"
    fleet_age_posture: str = "unknown"
    oem_support: str = "unknown"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "liquidity_tier": self.liquidity_tier,
            "resale_strength": self.resale_strength,
            "ownership_risk": self.ownership_risk,
            "maintenance_pressure": self.maintenance_pressure,
            "fleet_age_posture": self.fleet_age_posture,
            "oem_support": self.oem_support,
            "notes": list(self.notes),
        }


def build_acquisition_intelligence(
    model: str,
    *,
    vintage_year: Optional[int] = None,
    hours_per_year: float = 0,
    fractional: bool = False,
) -> AcquisitionIntelligenceBundle:
    key = (model or "").strip().lower()
    liq = assess_market_liquidity(model, year=vintage_year)
    resale = assess_resale_strength(model)
    own = assess_ownership_risk(model, utilization_hours_per_year=hours_per_year, fractional_interest=fractional)
    maint = estimate_maintenance_events(model)
    age = analyze_fleet_age(model, vintage_year=vintage_year)
    oem = _OEM_SUPPORT.get(key, "verify program")

    notes: List[str] = []
    notes.append(liq.notes)
    notes.append(resale.commentary)
    notes.extend(own.factors[:2])
    notes.extend(maint.events[:2])
    notes.extend(age.notes[:1])

    return AcquisitionIntelligenceBundle(
        model=model,
        liquidity_tier=liq.liquidity_tier,
        resale_strength=resale.resale_strength,
        ownership_risk=own.risk_level,
        maintenance_pressure=maint.reserve_pressure,
        fleet_age_posture=age.segment_posture,
        oem_support=oem,
        notes=[n for n in notes if n],
    )


def format_acquisition_intelligence_block(
    models: List[str],
    *,
    vintage_year: Optional[int] = None,
    hours_per_year: float = 0,
    fractional: bool = False,
) -> str:
    """Broker-facing acquisition notes for 1–3 models."""
    lines = ["Acquisition Intelligence:", ""]
    for model in models[:3]:
        bundle = build_acquisition_intelligence(
            model,
            vintage_year=vintage_year,
            hours_per_year=hours_per_year,
            fractional=fractional,
        )
        lines.append(f"* {model}")
        lines.append(
            f"  Liquidity: {bundle.liquidity_tier}; Resale: {bundle.resale_strength}; "
            f"OEM support: {bundle.oem_support}; Fleet age: {bundle.fleet_age_posture}"
        )
        lines.append(
            f"  Maintenance reserve pressure: {bundle.maintenance_pressure}; "
            f"Ownership risk: {bundle.ownership_risk}"
        )
        if bundle.notes:
            lines.append(f"  Note: {bundle.notes[0][:180]}")
    return "\n".join(lines)
