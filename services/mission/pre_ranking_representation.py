"""
Encode mission reality before feasibility / ranking — not understanding, not ranking.

Order: passenger distribution → route graph (continuations) → industrial classifier → governance.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from services.mission.models import Route
from services.mission.route_extractor import _build_route_extraction, dedupe_extractions

from services.consultant.mission_state import MissionState
from services.mission.industrial_airport_classifier import (
    apply_industrial_profile_to_mission,
    classify_industrial_airports,
)
from services.mission.mission_governance import (
    apply_governance_resolution,
    resolve_mission_governance,
)
from services.mission.mission_understanding_engine import MissionUnderstandingPacket
from services.mission.models import MissionProfile
from services.mission.passenger_distribution import (
    apply_passenger_distribution_to_profile,
    extract_passenger_distribution,
)
from services.mission.field_access_routes import infer_field_access_spokes
from services.mission.mission_anchor_routes import infer_mission_anchor_routes
from services.mission.route_graph_representation import (
    build_route_graph,
    merge_route_graph_into_mission,
    save_route_graph,
)
from services.mission.structural_representation import apply_structural_representation

PRE_RANKING_REPRESENTATION_KEY = "pre_ranking_representation"


def _industrial_ulr_band_conflict(profile: MissionProfile, packet: MissionUnderstandingPacket) -> bool:
    """Industrial field access + intercontinental nonstop is structurally incompatible."""
    if not packet.inferred_constraints.get("industrial_airport_access"):
        return False
    labels = " ".join(profile.route_labels()).lower()
    if any(c in labels for c in ("london", "dubai", "europe", "abu dhabi", "riyadh", "zurich")):
        return True
    return bool(
        any("transatlantic" in b.lower() or "ultra-long" in b.lower() for b in packet.fallback_operational_band)
    )


def _infer_domestic_triangle_routes(query: str, profile: MissionProfile) -> List[Route]:
    """When text states daily domestic hops but only ME legs were extracted, add US triangle."""
    tl = (query or "").lower()
    if not re.search(r"\b(?:domestic|daily\s+flights?|short\s+hops?|2\s*[-–]\s*3\s+hour)\b", tl):
        return []
    if not re.search(r"\b(?:nyc|new\s+york|chicago|san\s+francisco|\bsf\b)\b", tl):
        return []
    candidates: List[Route] = []
    for o, d in (
        ("New York", "Chicago"),
        ("Chicago", "San Francisco"),
        ("New York", "San Francisco"),
    ):
        ext = _build_route_extraction(o, d, pattern_boost=0.08)
        if ext:
            candidates.append(ext.route)
    existing = {(r.origin.lower(), r.destination.lower()) for r in profile.routes}
    return [r for r in candidates if (r.origin.lower(), r.destination.lower()) not in existing]


def apply_pre_ranking_representation(
    query: str,
    profile: MissionProfile,
    mission: MissionState,
    packet: Optional[MissionUnderstandingPacket],
    data_used: Optional[Dict[str, Any]] = None,
) -> Tuple[MissionProfile, MissionState, Optional[MissionUnderstandingPacket]]:
    """
    Single entry point before ranking — mutates profile, mission, packet, data_used.
    """
    du = data_used if isinstance(data_used, dict) else {}
    snapshot: Dict[str, Any] = {}

    from services.mission.explicit_route_lock import (
        EXPLICIT_ROUTE_LOCK_KEY,
        extract_explicit_routes,
        merge_explicit_routes_into_profile,
    )

    explicit_lock = extract_explicit_routes(query)
    du[EXPLICIT_ROUTE_LOCK_KEY] = explicit_lock.to_dict()
    explicit_merged = merge_explicit_routes_into_profile(explicit_lock, profile)
    if explicit_merged:
        mission.routes = profile.route_labels()
    snapshot["explicit_routes_merged"] = explicit_merged

    # 1. Passenger distribution (planning load = upper bound)
    dist = extract_passenger_distribution(query)
    apply_passenger_distribution_to_profile(profile, dist)
    if dist.planning_load is not None:
        mission.passenger_count = dist.planning_load
    snapshot["passenger_distribution"] = dist.to_dict()

    # 2a. Geographic + route intelligence (regional ontology, industrial spokes, hub anchors)
    from services.mission.geographic_route_intelligence import apply_geographic_route_intelligence

    geo_report = apply_geographic_route_intelligence(query, profile, data_used=du)
    snapshot["geographic_route_intelligence"] = geo_report.to_dict()

    # 2b. Route graph — anchors, field spokes, ME continuations, domestic triangle
    anchor_legs = infer_mission_anchor_routes(query, profile)
    if anchor_legs:
        profile.routes = list(profile.routes) + anchor_legs
        snapshot["anchor_routes_added"] = [r.label() for r in anchor_legs]

    field_spokes = infer_field_access_spokes(query, profile)
    if field_spokes:
        profile.routes = list(profile.routes) + field_spokes
        snapshot["field_access_spokes"] = [r.label() for r in field_spokes]

    extra_domestic = _infer_domestic_triangle_routes(query, profile)
    if extra_domestic:
        profile.routes = list(profile.routes) + extra_domestic
        snapshot["domestic_triangle_added"] = [r.label() for r in extra_domestic]

    route_graph = build_route_graph(query, profile)
    merge_route_graph_into_mission(route_graph, profile, mission)
    save_route_graph(du, route_graph)
    snapshot["route_graph"] = {
        "leg_count": len(route_graph.all_legs()),
        "inferred": route_graph.inferred_leg_labels,
    }

    from services.mission.route_directionality import apply_route_directionality

    dir_report = apply_route_directionality(query, profile, mission, data_used=du)
    snapshot["route_directionality"] = dir_report.to_dict()

    if packet is not None:
        from services.mission.mission_place_index import places_captured_from_mission

        packet.explicit_constraints["routes"] = profile.route_labels()
        packet.explicit_constraints["passengers"] = dist.planning_load
        packet.explicit_constraints["passenger_distribution"] = dist.to_dict()
        packet.explicit_constraints["places_captured"] = places_captured_from_mission(
            profile, query
        )
        snapshot["places_captured"] = packet.explicit_constraints["places_captured"]
        if dist.is_variable:
            packet.inferred_constraints["passenger_load_variable"] = True
            packet.inferred_constraints["planning_passenger_load"] = dist.planning_load

    # 3. Industrial airport classifier
    industrial = classify_industrial_airports(query)
    apply_industrial_profile_to_mission(profile, packet, industrial, data_used=du)
    snapshot["industrial_airport"] = industrial.to_dict()

    if packet is not None and industrial.active and _industrial_ulr_band_conflict(profile, packet):
        packet.inferred_constraints["incompatible_mission_bands"] = True
        if "Transatlantic super-mid / heavy-cabin executive band" not in (
            packet.fallback_operational_band or []
        ):
            packet.fallback_operational_band.append(
                "Transatlantic super-mid / heavy-cabin executive band"
            )
        snapshot["industrial_ulr_conflict"] = True

    # 4. Governance before ranking
    governance = resolve_mission_governance(query, profile, packet, mission=mission)
    apply_governance_resolution(governance, packet, data_used=du)
    snapshot["governance"] = governance.to_dict()

    # 5. Structural representation proofs (engine law for decomposition / fleet)
    structural = apply_structural_representation(
        query,
        profile,
        mission,
        packet,
        data_used=du,
        governance=governance.to_dict(),
        industrial=snapshot.get("industrial_airport") or {},
    )
    snapshot["structural_representation"] = structural.to_dict()

    du[PRE_RANKING_REPRESENTATION_KEY] = snapshot
    du["pre_ranking_applied"] = 1

    from services.mission.route_topology_validator import apply_route_topology_validation

    profile, mission, _topo = apply_route_topology_validation(
        query, profile, mission, packet=packet, data_used=du
    )
    snapshot["route_topology"] = (du.get("route_topology_validation") or {})

    from services.mission.geographic_graph_authority import apply_geographic_graph_authority

    inferred_labels = (snapshot.get("route_graph") or {}).get("inferred") or []
    topo_removed = (du.get("route_topology_validation") or {}).get("removed_routes") or []
    dir_removed = (du.get("route_directionality") or {}).get("removed") or []
    blocked = [f"topo:{x}" for x in topo_removed] + [f"dir:{x}" for x in dir_removed]

    auth_graph = apply_geographic_graph_authority(
        query,
        profile,
        mission,
        data_used=du,
        inferred_labels=inferred_labels,
        blocked_edges=blocked,
    )
    snapshot["geographic_graph_authority"] = auth_graph.to_dict()

    # 7. Geographic + routing stabilization (no semantics changes)
    try:
        from services.mission.mission_graph_stabilizer import stabilize_mission_graph

        stabilize_mission_graph(
            query=query,
            profile=profile,
            original_route_labels=packet.explicit_constraints.get("routes")
            if packet is not None
            else None,
            data_used=du,
        )
    except Exception:
        pass

    return profile, mission, packet
