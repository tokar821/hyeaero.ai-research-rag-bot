"""
Deterministic mission operational graph — stable decomposition across turns.

Same mission posture + broker memory should yield the same corridor set,
fleet inference, and incompatible-band flags regardless of LLM narration variance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from services.consultant.mission_state import MissionState
from services.mission.models import MissionProfile
from services.mission.mission_understanding_engine import MissionUnderstandingPacket

MISSION_OPERATIONAL_GRAPH_KEY = "mission_operational_graph"

# Rule-derived keys that LLM merge must never clear once set.
PROTECTED_INFERRED_KEYS = frozenset(
    {
        "incompatible_mission_bands",
        "dual_use_or_multi_leg",
        "executive_travel_profile",
        "minimum_jet_cabin_floor",
        "planning_band_ceiling",
        "international_jet_floor",
        "balanced_cost_dispatch",
        "cabin_utilization_modest",
        "transatlantic_super_mid_floor",
        "industrial_airport_access",
        "mountain_ops",
        "ownership_economics_relevant",
        "annual_charter_hours",
        "fuel_stop_averse",
        "runway_over_cabin",
        "arctic_hard_domain",
        "industrial_hard_domain",
        "mining_hard_domain",
        "multi_hard_domain_mission",
        "semantic_invalid_interpretations",
        "hard_domains",
        "mission_semantic_domains",
    }
)


@dataclass
class MissionOperationalGraph:
    """Canonical operational decomposition snapshot for session continuity."""

    corridor_type: str = "unknown"
    travel_pattern: str = "unknown"
    utilization_style: str = "unknown"
    operational_bands: List[str] = field(default_factory=list)
    inferred_flags: Dict[str, Any] = field(default_factory=dict)
    fleet_strategy_required: bool = False
    incompatible_bands: bool = False
    nonstop_priority: bool = False
    executive_profile: bool = False
    turn_fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corridor_type": self.corridor_type,
            "travel_pattern": self.travel_pattern,
            "utilization_style": self.utilization_style,
            "operational_bands": list(self.operational_bands),
            "inferred_flags": dict(self.inferred_flags),
            "fleet_strategy_required": self.fleet_strategy_required,
            "incompatible_bands": self.incompatible_bands,
            "nonstop_priority": self.nonstop_priority,
            "executive_profile": self.executive_profile,
            "turn_fingerprint": self.turn_fingerprint,
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "MissionOperationalGraph":
        if not isinstance(raw, dict):
            return cls()
        inf = raw.get("inferred_flags")
        return cls(
            corridor_type=str(raw.get("corridor_type") or "unknown"),
            travel_pattern=str(raw.get("travel_pattern") or "unknown"),
            utilization_style=str(raw.get("utilization_style") or "unknown"),
            operational_bands=[
                str(b) for b in (raw.get("operational_bands") or []) if b
            ],
            inferred_flags=dict(inf) if isinstance(inf, dict) else {},
            fleet_strategy_required=bool(raw.get("fleet_strategy_required")),
            incompatible_bands=bool(raw.get("incompatible_bands")),
            nonstop_priority=bool(raw.get("nonstop_priority")),
            executive_profile=bool(raw.get("executive_profile")),
            turn_fingerprint=str(raw.get("turn_fingerprint") or ""),
        )


def graph_from_packet(packet: MissionUnderstandingPacket) -> MissionOperationalGraph:
    inf = dict(packet.inferred_constraints or {})
    return MissionOperationalGraph(
        corridor_type=str(packet.corridor_type or "unknown"),
        travel_pattern=str(packet.travel_pattern or "unknown"),
        utilization_style=str(packet.utilization_style or "unknown"),
        operational_bands=list(packet.fallback_operational_band or []),
        inferred_flags=inf,
        fleet_strategy_required=bool(inf.get("incompatible_mission_bands")),
        incompatible_bands=bool(inf.get("incompatible_mission_bands")),
        nonstop_priority=packet.nonstop_priority == "high",
        executive_profile=bool(
            inf.get("executive_travel_profile") or inf.get("minimum_jet_cabin_floor")
        ),
    )


def _fingerprint(query: str, profile: MissionProfile) -> str:
    blob = json.dumps(
        {
            "q": (query or "").strip().lower()[:500],
            "routes": profile.route_labels()[:6],
            "pax": profile.passengers,
            "regions": list(profile.regions or [])[:6],
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def merge_graphs(
    prior: MissionOperationalGraph,
    current: MissionOperationalGraph,
    *,
    allow_prior_merge: bool = True,
) -> MissionOperationalGraph:
    """Session merge — prior structure only when continuity validated."""
    if not allow_prior_merge or not (
        prior.operational_bands or prior.inferred_flags or prior.corridor_type not in ("unknown", "")
    ):
        return current

    bands = list(
        dict.fromkeys(
            list(current.operational_bands or []) + list(prior.operational_bands or [])
        )
    )
    inf: Dict[str, Any] = dict(current.inferred_flags or {})
    for k, v in (prior.inferred_flags or {}).items():
        if k in PROTECTED_INFERRED_KEYS:
            if not inf.get(k):
                inf[k] = v
        elif k not in inf:
            inf[k] = v

    incompatible = bool(
        current.incompatible_bands
        or inf.get("incompatible_mission_bands")
    )
    fleet = bool(
        current.fleet_strategy_required
        or incompatible
        or inf.get("dual_use_or_multi_leg")
    )
    return MissionOperationalGraph(
        corridor_type=current.corridor_type
        if current.corridor_type != "unknown"
        else prior.corridor_type,
        travel_pattern=current.travel_pattern
        if current.travel_pattern != "unknown"
        else prior.travel_pattern,
        utilization_style=current.utilization_style
        if current.utilization_style not in ("unknown", "", None)
        else prior.utilization_style,
        operational_bands=bands,
        inferred_flags=inf,
        fleet_strategy_required=fleet,
        incompatible_bands=incompatible,
        nonstop_priority=current.nonstop_priority or prior.nonstop_priority,
        executive_profile=current.executive_profile or prior.executive_profile,
        turn_fingerprint=current.turn_fingerprint or prior.turn_fingerprint,
    )


def apply_graph_to_packet(
    packet: MissionUnderstandingPacket,
    graph: MissionOperationalGraph,
) -> MissionUnderstandingPacket:
    """Restore persisted operational structure into the understanding packet."""
    if graph.corridor_type not in ("unknown", ""):
        packet.corridor_type = graph.corridor_type
    if graph.travel_pattern not in ("unknown", ""):
        packet.travel_pattern = graph.travel_pattern
    if graph.utilization_style not in ("unknown", "", None):
        packet.utilization_style = graph.utilization_style
    if graph.operational_bands:
        packet.fallback_operational_band = list(
            dict.fromkeys(
                list(packet.fallback_operational_band or []) + graph.operational_bands
            )
        )
    for k, v in (graph.inferred_flags or {}).items():
        packet.inferred_constraints[k] = v
    if graph.incompatible_bands:
        packet.inferred_constraints["incompatible_mission_bands"] = True
    if graph.fleet_strategy_required:
        packet.inferred_constraints["dual_use_or_multi_leg"] = True
    if graph.nonstop_priority:
        packet.nonstop_priority = "high"
    if graph.executive_profile:
        packet.inferred_constraints["executive_travel_profile"] = True
    return packet


def apply_broker_memory_to_packet(
    packet: MissionUnderstandingPacket,
    broker_memory: Optional[Dict[str, Any]],
    *,
    apply_structural: bool = True,
    apply_posture: bool = True,
) -> MissionUnderstandingPacket:
    """Rehydrate broker memory — structural fields only when continuity validated."""
    if not isinstance(broker_memory, dict):
        return packet
    if apply_posture:
        if broker_memory.get("nonstop_preference"):
            packet.nonstop_priority = "high"
    if apply_structural:
        if broker_memory.get("fleet_strategy_required") or broker_memory.get(
            "incompatible_bands"
        ):
            packet.inferred_constraints["incompatible_mission_bands"] = True
            packet.inferred_constraints["dual_use_or_multi_leg"] = True
        if broker_memory.get("executive_travel_profile"):
            packet.inferred_constraints["executive_travel_profile"] = True
        if broker_memory.get("minimum_jet_cabin_floor"):
            packet.inferred_constraints["minimum_jet_cabin_floor"] = True
        bands = broker_memory.get("operational_bands") or []
        if isinstance(bands, list) and bands:
            packet.fallback_operational_band = list(
                dict.fromkeys(list(packet.fallback_operational_band or []) + bands)
            )
    elif apply_posture:
        if broker_memory.get("executive_travel_profile"):
            packet.inferred_constraints.setdefault("executive_travel_profile", True)
    return packet


def stabilize_mission_understanding(
    packet: MissionUnderstandingPacket,
    *,
    query: str,
    profile: MissionProfile,
    mission: MissionState,
    broker_memory: Optional[Dict[str, Any]] = None,
    prior_graph: Optional[MissionOperationalGraph] = None,
    continuity: Optional[Any] = None,
) -> tuple[MissionUnderstandingPacket, MissionOperationalGraph]:
    """
    Reconcile current turn with session memory; rebuild synthesis from rules.
    """
    from services.mission.mission_context_reconciliation import assess_mission_continuity

    cont = continuity
    if cont is None:
        cont = assess_mission_continuity(
            query,
            profile,
            broker_memory=broker_memory,
            prior_graph=prior_graph,
        )

    packet = apply_broker_memory_to_packet(
        packet,
        broker_memory,
        apply_structural=cont.apply_structural_memory,
        apply_posture=cont.apply_posture_memory,
    )
    current = graph_from_packet(packet)
    current.turn_fingerprint = _fingerprint(query, profile)
    merged = merge_graphs(
        prior_graph or MissionOperationalGraph(),
        current,
        allow_prior_merge=cont.apply_structural_memory,
    )
    packet = apply_graph_to_packet(packet, merged)

    if cont.mission_pivot:
        packet.understanding_notes.append(
            f"Mission pivot detected ({cont.reason}) — prior corridor/bands not merged."
        )

    try:
        from services.mission.operational_synthesis import enrich_operational_synthesis

        packet.operational_synthesis = enrich_operational_synthesis(
            packet, mission, profile, query=query
        )
    except Exception:
        pass

    return packet, merged


def load_operational_graph(data_used: Optional[Dict[str, Any]]) -> MissionOperationalGraph:
    if not isinstance(data_used, dict):
        return MissionOperationalGraph()
    raw = data_used.get(MISSION_OPERATIONAL_GRAPH_KEY)
    if isinstance(raw, dict):
        return MissionOperationalGraph.from_dict(raw)
    return MissionOperationalGraph()


def save_operational_graph(
    data_used: Dict[str, Any],
    graph: MissionOperationalGraph,
) -> None:
    data_used[MISSION_OPERATIONAL_GRAPH_KEY] = graph.to_dict()


def should_defer_ranking_to_fleet(
    packet: Optional[MissionUnderstandingPacket],
    data_used: Optional[Dict[str, Any]] = None,
    *,
    profile: Optional[Any] = None,
    mission: Optional[Any] = None,
    query: str = "",
    feasible_models: Optional[Sequence[str]] = None,
) -> bool:
    from services.mission.structural_decomposition import needs_structural_decomposition

    if isinstance(data_used, dict) and data_used.get("fleet_strategy_required"):
        return True
    if needs_structural_decomposition(
        packet,
        profile=profile,
        mission=mission,
        query=query,
        data_used=data_used,
        feasible_models=feasible_models,
    ).required:
        return True
    if isinstance(data_used, dict):
        if data_used.get("fleet_doctrine_lock") or data_used.get("ranking_defer_to_fleet"):
            return True
        fp = data_used.get("fleet_composition_plan")
        if isinstance(fp, dict) and fp.get("single_aircraft_structurally_invalid"):
            return True
    return False


def requires_fleet_decomposition_before_ranking(
    packet: Optional[MissionUnderstandingPacket],
    data_used: Optional[Dict[str, Any]] = None,
    *,
    profile: Optional[Any] = None,
    mission: Optional[Any] = None,
    query: str = "",
    feasible_models: Optional[Sequence[str]] = None,
) -> bool:
    """True when fleet plan must be built before global shortlist generation."""
    from services.mission.structural_decomposition import needs_structural_decomposition

    return needs_structural_decomposition(
        packet,
        profile=profile,
        mission=mission,
        query=query,
        data_used=data_used,
        feasible_models=feasible_models,
    ).required
