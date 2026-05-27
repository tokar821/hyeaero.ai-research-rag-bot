"""
Payload realism — passenger mass, baggage modifiers, fuel-vs-payload range erosion.

Replaces abstract "8 pax + bags" with structured assumptions brokers actually use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.consultant.mission_state import MissionState
from services.mission.models import MissionProfile, PriorityLevel

# Standard broker planning assumptions (lb)
_LB_PER_PASSENGER_STANDARD = 200.0
_LB_PER_PASSENGER_EXECUTIVE = 225.0
_LB_BAGGAGE_PER_PAX_STANDARD = 45.0
_LB_BAGGAGE_PER_PAX_HEAVY = 75.0

_MODIFIER_LB = {
    "ski": 280.0,
    "golf": 120.0,
    "heavy_cargo": 450.0,
    "equipment": 350.0,
    "pets": 40.0,
}

_SKI_RE = re.compile(r"\b(?:ski|skis|snowboard)\b", re.I)
_GOLF_RE = re.compile(r"\b(?:golf|clubs)\b", re.I)
_HEAVY_BAG_RE = re.compile(
    r"\b(?:heavy\s+baggage|lots\s+of\s+luggage|maximum\s+baggage|bulky)\b",
    re.I,
)
_EQUIP_RE = re.compile(r"\b(?:equipment|gear|cargo)\b", re.I)


@dataclass
class MissionPayloadProfile:
    passengers: int
    passenger_weight_lb: float
    baggage_weight_lb: float
    modifier_weight_lb: float
    total_payload_lb: float
    modifiers: List[str] = field(default_factory=list)
    fuel_trade_nm_penalty: float = 0.0
    range_erosion_factor: float = 0.0
    assumptions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passengers": self.passengers,
            "passenger_weight_lb": round(self.passenger_weight_lb, 0),
            "baggage_weight_lb": round(self.baggage_weight_lb, 0),
            "modifier_weight_lb": round(self.modifier_weight_lb, 0),
            "total_payload_lb": round(self.total_payload_lb, 0),
            "modifiers": list(self.modifiers),
            "fuel_trade_nm_penalty": round(self.fuel_trade_nm_penalty, 1),
            "range_erosion_factor": round(self.range_erosion_factor, 4),
            "assumptions": list(self.assumptions),
        }


def _infer_modifiers(query: str, profile: Optional[MissionProfile]) -> List[str]:
    mods: List[str] = []
    ql = (query or "").lower()
    if _SKI_RE.search(ql):
        mods.append("ski")
    if _GOLF_RE.search(ql):
        mods.append("golf")
    if _HEAVY_BAG_RE.search(ql) or (
        profile and profile.baggage_priority in (PriorityLevel.HIGH, PriorityLevel.MEDIUM)
    ):
        mods.append("heavy_baggage")
    if _EQUIP_RE.search(ql):
        mods.append("equipment")
    return list(dict.fromkeys(mods))


def build_mission_payload_profile(
    mission: MissionState,
    *,
    profile: Optional[MissionProfile] = None,
    query: str = "",
    stage_distance_nm: float = 0.0,
) -> MissionPayloadProfile:
    """Structured payload mass and NM penalty from pax + baggage + mission modifiers."""
    pax = int(mission.passenger_count or (profile.passengers if profile else None) or 6)
    mods = _infer_modifiers(query, profile)

    per_pax = _LB_PER_PASSENGER_EXECUTIVE if pax <= 6 else _LB_PER_PASSENGER_STANDARD
    passenger_lb = pax * per_pax

    bag_per_pax = (
        _LB_BAGGAGE_PER_PAX_HEAVY
        if "heavy_baggage" in mods or (mission.baggage_priority or "") == "high"
        else _LB_BAGGAGE_PER_PAX_STANDARD
    )
    baggage_lb = pax * bag_per_pax

    modifier_lb = sum(_MODIFIER_LB.get(m, 0) for m in mods if m in _MODIFIER_LB)
    total_lb = passenger_lb + baggage_lb + modifier_lb

    # Fuel trade: ~0.35 nm per 100 lb payload on long stages (conservative)
    stage = max(stage_distance_nm, 400.0)
    fuel_trade_nm = (total_lb / 100.0) * (0.22 if stage < 1500 else 0.35 if stage < 3000 else 0.42)

    if pax >= 10:
        fuel_trade_nm += 80.0
    elif pax >= 8:
        fuel_trade_nm += 45.0

    if mission.mountain_airport_requirement:
        fuel_trade_nm += 60.0

    # Erosion factor applied to practical available (0–0.25)
    erosion = min(0.25, (total_lb / 8000.0) * 0.12 + len(mods) * 0.03)

    assumptions = [
        f"{pax} passengers at ~{int(per_pax)} lb each (planning weight).",
        f"Baggage ~{int(baggage_lb)} lb total ({bag_per_pax:.0f} lb/pax).",
    ]
    if mods:
        assumptions.append(f"Mission modifiers: {', '.join(mods)} (+{int(modifier_lb)} lb).")
    assumptions.append(f"Payload fuel-trade equivalent ~{int(fuel_trade_nm)} nm on {int(stage)} nm stage.")

    return MissionPayloadProfile(
        passengers=pax,
        passenger_weight_lb=passenger_lb,
        baggage_weight_lb=baggage_lb,
        modifier_weight_lb=modifier_lb,
        total_payload_lb=total_lb,
        modifiers=mods,
        fuel_trade_nm_penalty=fuel_trade_nm,
        range_erosion_factor=erosion,
        assumptions=assumptions,
    )


def effective_practical_nm(
    brochure_practical_nm: float,
    payload: MissionPayloadProfile,
) -> float:
    """Operational available range after payload erosion (not brochure)."""
    base = float(brochure_practical_nm or 0)
    if base <= 0:
        return 0.0
    eroded = base * (1.0 - payload.range_erosion_factor)
    return max(0.0, eroded - payload.fuel_trade_nm_penalty * 0.5)
