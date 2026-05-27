"""
Broker memory — session continuity for advisor-style reasoning.

Persists operational philosophy across turns (not stateless ranking).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

BROKER_MEMORY_KEY = "hye_broker_memory"


@dataclass
class BrokerMemory:
    recurring_routes: List[str] = field(default_factory=list)
    nonstop_preference: bool = False
    tech_stop_tolerance: str = "unknown"  # none | limited | flexible
    cabin_preferences: List[str] = field(default_factory=list)
    runway_flexibility_priority: str = "unknown"  # low | medium | high
    cost_sensitivity: str = "unknown"  # low | medium | high
    preferred_oem: List[str] = field(default_factory=list)
    preferred_categories: List[str] = field(default_factory=list)
    operational_philosophy: str = ""  # dispatch-first | cost-first | field-flexible
    last_mission_style: str = ""
    enterprise_scale: str = ""
    travel_pattern: str = ""
    ownership_profile: str = ""
    international_frequency: str = ""
    # Sticky operational decomposition (mission graph)
    operational_bands: List[str] = field(default_factory=list)
    incompatible_bands: bool = False
    fleet_strategy_required: bool = False
    executive_travel_profile: bool = False
    minimum_jet_cabin_floor: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recurring_routes": list(self.recurring_routes),
            "nonstop_preference": self.nonstop_preference,
            "tech_stop_tolerance": self.tech_stop_tolerance,
            "cabin_preferences": list(self.cabin_preferences),
            "runway_flexibility_priority": self.runway_flexibility_priority,
            "cost_sensitivity": self.cost_sensitivity,
            "preferred_oem": list(self.preferred_oem),
            "preferred_categories": list(self.preferred_categories),
            "operational_philosophy": self.operational_philosophy,
            "last_mission_style": self.last_mission_style,
            "enterprise_scale": self.enterprise_scale,
            "travel_pattern": self.travel_pattern,
            "ownership_profile": self.ownership_profile,
            "international_frequency": self.international_frequency,
            "operational_bands": list(self.operational_bands),
            "incompatible_bands": self.incompatible_bands,
            "fleet_strategy_required": self.fleet_strategy_required,
            "executive_travel_profile": self.executive_travel_profile,
            "minimum_jet_cabin_floor": self.minimum_jet_cabin_floor,
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "BrokerMemory":
        if not isinstance(raw, dict):
            return cls()
        return cls(
            recurring_routes=[str(r) for r in (raw.get("recurring_routes") or []) if r],
            nonstop_preference=bool(raw.get("nonstop_preference")),
            tech_stop_tolerance=str(raw.get("tech_stop_tolerance") or "unknown"),
            cabin_preferences=[str(c) for c in (raw.get("cabin_preferences") or []) if c],
            runway_flexibility_priority=str(
                raw.get("runway_flexibility_priority") or "unknown"
            ),
            cost_sensitivity=str(raw.get("cost_sensitivity") or "unknown"),
            preferred_oem=[str(o) for o in (raw.get("preferred_oem") or []) if o],
            preferred_categories=[
                str(c) for c in (raw.get("preferred_categories") or []) if c
            ],
            operational_philosophy=str(raw.get("operational_philosophy") or ""),
            last_mission_style=str(raw.get("last_mission_style") or ""),
            enterprise_scale=str(raw.get("enterprise_scale") or ""),
            travel_pattern=str(raw.get("travel_pattern") or ""),
            ownership_profile=str(raw.get("ownership_profile") or ""),
            international_frequency=str(raw.get("international_frequency") or ""),
            operational_bands=[
                str(b) for b in (raw.get("operational_bands") or []) if b
            ],
            incompatible_bands=bool(raw.get("incompatible_bands")),
            fleet_strategy_required=bool(raw.get("fleet_strategy_required")),
            executive_travel_profile=bool(raw.get("executive_travel_profile")),
            minimum_jet_cabin_floor=bool(raw.get("minimum_jet_cabin_floor")),
        )


def load_broker_memory(data_used: Optional[Dict[str, Any]]) -> BrokerMemory:
    if not isinstance(data_used, dict):
        return BrokerMemory()
    return BrokerMemory.from_dict(data_used.get(BROKER_MEMORY_KEY))


def save_broker_memory(data_used: Dict[str, Any], memory: BrokerMemory) -> Dict[str, Any]:
    data_used[BROKER_MEMORY_KEY] = memory.to_dict()
    return data_used


def update_broker_memory_from_turn(
    memory: BrokerMemory,
    *,
    route: Optional[str] = None,
    inferred_profile: Optional[Dict[str, Any]] = None,
    mission_style: Optional[str] = None,
) -> BrokerMemory:
    """Merge current turn signals into broker memory."""
    if route:
        route_key = route.strip().lower()
        if route_key and route_key not in memory.recurring_routes:
            memory.recurring_routes = (memory.recurring_routes + [route_key])[-8:]

    if isinstance(inferred_profile, dict):
        if inferred_profile.get("nonstop_preference"):
            memory.nonstop_preference = True
        tol = inferred_profile.get("tech_stop_tolerance")
        if tol and tol != "unknown":
            memory.tech_stop_tolerance = str(tol)
        if inferred_profile.get("airport_access_priority"):
            memory.runway_flexibility_priority = "high"
        if inferred_profile.get("cost_sensitive"):
            memory.cost_sensitivity = "high"
        if inferred_profile.get("dispatch_priority"):
            memory.operational_philosophy = "dispatch-first"
            memory.cost_sensitivity = memory.cost_sensitivity or "low"
        style = inferred_profile.get("utilization_style")
        if style:
            memory.last_mission_style = str(style)

    if mission_style:
        memory.last_mission_style = mission_style

    return memory


def update_broker_memory_from_understanding(
    memory: BrokerMemory,
    packet: Any,
) -> BrokerMemory:
    """Persist operational posture from mission understanding packet."""
    if packet is None:
        return memory
    style = getattr(packet, "utilization_style", None) or (
        packet.get("utilization_style") if isinstance(packet, dict) else None
    )
    if style and style != "unknown":
        memory.last_mission_style = str(style)
    travel = getattr(packet, "travel_pattern", None) or (
        packet.get("travel_pattern") if isinstance(packet, dict) else ""
    )
    if travel and travel != "unknown":
        memory.travel_pattern = str(travel)
    own = getattr(packet, "ownership_profile", None) or (
        packet.get("ownership_profile") if isinstance(packet, dict) else ""
    )
    if own and own != "unknown":
        memory.ownership_profile = str(own)
    inf = getattr(packet, "inferred_constraints", None) or (
        packet.get("inferred_constraints") if isinstance(packet, dict) else {}
    )
    if isinstance(inf, dict):
        if inf.get("enterprise_employees"):
            memory.enterprise_scale = f"~{inf['enterprise_employees']} employees"
        if inf.get("europe_trip_frequency"):
            memory.international_frequency = f"~{inf['europe_trip_frequency']} europe trips/yr"
        if inf.get("ownership_viability_threshold"):
            memory.operational_philosophy = (
                memory.operational_philosophy or "ownership-economics-relevant"
            )
    if getattr(packet, "dispatch_priority", "") == "high" or (
        isinstance(packet, dict) and packet.get("dispatch_priority") == "high"
    ):
        memory.operational_philosophy = "dispatch-first"
    if getattr(packet, "operating_cost_priority", "") == "high" or (
        isinstance(packet, dict) and packet.get("operating_cost_priority") == "high"
    ):
        memory.cost_sensitivity = "high"
    if getattr(packet, "runway_complexity", "") in ("high", "mountain", "regional_access") or (
        isinstance(packet, dict)
        and packet.get("runway_complexity") in ("high", "mountain", "regional_access")
    ):
        memory.runway_flexibility_priority = "high"
    if getattr(packet, "nonstop_priority", "") == "high" or (
        isinstance(packet, dict) and packet.get("nonstop_priority") == "high"
    ):
        memory.nonstop_preference = True

    bands: List[str] = []
    if hasattr(packet, "fallback_operational_band"):
        bands = list(getattr(packet, "fallback_operational_band") or [])
    elif isinstance(packet, dict):
        bands = list(packet.get("fallback_operational_band") or [])
    if bands:
        memory.operational_bands = list(
            dict.fromkeys(list(memory.operational_bands or []) + bands)
        )[:8]

    if isinstance(inf, dict):
        if inf.get("incompatible_mission_bands"):
            memory.incompatible_bands = True
            memory.fleet_strategy_required = True
        if inf.get("executive_travel_profile"):
            memory.executive_travel_profile = True
        if inf.get("minimum_jet_cabin_floor"):
            memory.minimum_jet_cabin_floor = True
            memory.executive_travel_profile = True
    return memory
