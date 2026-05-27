"""
Authoritative mission route graph — geographic source of truth before recommendations.

Assembles explicit + inferred edges with origin clusters and domain layers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from services.consultant.mission_state import MissionState, normalize_routes
from services.mission.arctic_industrial_layer import ARCTIC_LAYER_NODES
from services.mission.explicit_route_lock import (
    EXPLICIT_ROUTE_LOCK_KEY,
    ExplicitRouteLock,
    extract_explicit_routes,
    merge_explicit_routes_into_profile,
    strip_conflicting_inferred_routes,
)
from services.mission.models import MissionProfile, Route
from services.mission.route_extractor import resolve_place

GEOGRAPHIC_GRAPH_AUTHORITY_KEY = "geographic_graph_authority"

_EU_EXEC_CITIES = frozenset(
    {
        "paris",
        "geneva",
        "zurich",
        "frankfurt",
        "london",
        "madrid",
        "rome",
        "milan",
        "berlin",
        "munich",
    }
)
_INDUSTRIAL_NODE_NAMES = frozenset(
    {
        "permian basin",
        "remote drilling sites",
        "desert energy corridor",
        "west africa",
        "nigerian energy corridor",
        "northern africa",
        "offshore rigs",
        "australian extraction strips",
        "remote gravel strips",
        "northern alberta oil fields",
        "nunavut field ops",
        "arctic industrial access",
        "arctic oil platforms",
    }
)
_ARCTIC_NODE_NAMES = frozenset(n.lower() for n in ARCTIC_LAYER_NODES) | {
    "remote gravel strips",
    "northern alberta oil fields",
    "nunavut field ops",
    "arctic industrial access",
    "arctic oil platforms",
    "yellowknife",
}


@dataclass
class GraphEdge:
    origin: str
    destination: str
    authority: str  # explicit | inferred | industrial | arctic | eu_exec | continuation
    origin_cluster: str = ""
    locked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": self.origin,
            "destination": self.destination,
            "authority": self.authority,
            "origin_cluster": self.origin_cluster,
            "locked": self.locked,
        }


@dataclass
class AuthoritativeMissionRouteGraph:
    nodes: List[str] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    origin_clusters: Dict[str, List[str]] = field(default_factory=dict)
    industrial_nodes: List[str] = field(default_factory=list)
    arctic_nodes: List[str] = field(default_factory=list)
    eu_exec_layer: List[str] = field(default_factory=list)
    invalid_edges_blocked: List[str] = field(default_factory=list)
    explicit_route_labels: List[str] = field(default_factory=list)
    unresolved_geography_nodes: List[str] = field(default_factory=list)

    def route_labels(self) -> List[str]:
        return [f"{e.origin} -> {e.destination}" for e in self.edges]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "edges": [e.to_dict() for e in self.edges],
            "origin_clusters": {k: list(v) for k, v in self.origin_clusters.items()},
            "industrial_nodes": list(self.industrial_nodes),
            "arctic_nodes": list(self.arctic_nodes),
            "eu_exec_layer": list(self.eu_exec_layer),
            "invalid_edges_blocked": list(self.invalid_edges_blocked),
            "explicit_route_labels": list(self.explicit_route_labels),
            "unresolved_geography_nodes": list(self.unresolved_geography_nodes),
            "route_labels": self.route_labels(),
        }


def _cluster_for_origin(origin: str) -> str:
    place, conf = resolve_place(origin)
    if place and conf >= 0.72:
        if place.country == "US":
            return place.canonical
        return place.canonical
    return origin


def _edge_authority(
    route: Route,
    lock: ExplicitRouteLock,
    inferred_labels: Set[str],
) -> str:
    lbl = route.label()
    if lock.is_locked_route(route):
        return "explicit"
    o_l, d_l = route.origin.lower(), route.destination.lower()
    if d_l in _ARCTIC_NODE_NAMES or o_l in _ARCTIC_NODE_NAMES:
        return "arctic"
    if d_l in _INDUSTRIAL_NODE_NAMES:
        return "industrial"
    if o_l in _EU_EXEC_CITIES or d_l in _EU_EXEC_CITIES:
        return "eu_exec"
    if lbl in inferred_labels:
        return "continuation"
    return "inferred"


def build_authoritative_graph(
    query: str,
    profile: MissionProfile,
    *,
    lock: Optional[ExplicitRouteLock] = None,
    inferred_labels: Optional[List[str]] = None,
    blocked_edges: Optional[List[str]] = None,
) -> AuthoritativeMissionRouteGraph:
    """Build structured graph from final profile routes."""
    lock = lock or extract_explicit_routes(query)
    inferred_set = set(inferred_labels or [])
    blocked = list(blocked_edges or [])

    nodes: Set[str] = set()
    edges: List[GraphEdge] = []
    clusters: Dict[str, List[str]] = {}
    industrial: Set[str] = set()
    arctic: Set[str] = set()
    eu_exec: Set[str] = set()
    unresolved: Set[str] = set()

    for route in profile.routes:
        cluster = _cluster_for_origin(route.origin)
        auth = _edge_authority(route, lock, inferred_set)
        # Block industrial region as origin to another region unless explicit
        o_l, d_l = route.origin.lower(), route.destination.lower()
        if (
            o_l in _INDUSTRIAL_NODE_NAMES
            and d_l in _INDUSTRIAL_NODE_NAMES
            and not lock.is_locked_route(route)
        ):
            blocked = blocked or []
            blocked.append(f"industrial cross-link blocked: {route.label()}")
            continue
        edge = GraphEdge(
            origin=route.origin,
            destination=route.destination,
            authority=auth,
            origin_cluster=cluster,
            locked=lock.is_locked_route(route),
        )
        edges.append(edge)
        nodes.add(route.origin)
        nodes.add(route.destination)
        clusters.setdefault(cluster, []).append(route.label())

        d_l = route.destination.lower()
        o_l = route.origin.lower()
        if d_l in _INDUSTRIAL_NODE_NAMES:
            industrial.add(route.destination)
        if d_l in _ARCTIC_NODE_NAMES or o_l in _ARCTIC_NODE_NAMES:
            arctic.update({route.origin, route.destination})
        if o_l in _EU_EXEC_CITIES or d_l in _EU_EXEC_CITIES:
            eu_exec.add(f"{route.origin} -> {route.destination}")

        for name in (route.origin, route.destination):
            place, conf = resolve_place(name)
            if not place or conf < 0.72:
                unresolved.add(name)

    return AuthoritativeMissionRouteGraph(
        nodes=sorted(nodes),
        edges=edges,
        origin_clusters=clusters,
        industrial_nodes=sorted(industrial),
        arctic_nodes=sorted(arctic),
        eu_exec_layer=sorted(eu_exec),
        invalid_edges_blocked=blocked,
        explicit_route_labels=list(lock.locked_labels),
        unresolved_geography_nodes=sorted(unresolved),
    )


def _dedupe_mountain_hub_spokes(profile: MissionProfile, text: str) -> List[str]:
    """When LA and Denver both spoke to same mountain dest, keep LA if LA is in mission text."""
    tl = (text or "").lower()
    if not re.search(r"\b(?:los\s+angeles|\bla\b)\b", tl):
        return []
    la_hub = "los angeles"
    mountain_dests: Dict[str, Route] = {}
    removed: List[str] = []
    kept: List[Route] = []
    for r in profile.routes:
        d_l = r.destination.lower()
        if d_l not in {
            "aspen", "telluride", "jackson hole", "vail", "eagle", "sun valley"
        }:
            kept.append(r)
            continue
        if d_l not in mountain_dests:
            mountain_dests[d_l] = r
            kept.append(r)
            continue
        existing = mountain_dests[d_l]
        if existing.origin.lower() == la_hub:
            if r.origin.lower() != la_hub:
                removed.append(r.label())
            else:
                kept.append(r)
        elif r.origin.lower() == la_hub:
            kept = [x for x in kept if x.label() != existing.label()]
            removed.append(existing.label())
            mountain_dests[d_l] = r
            kept.append(r)
        else:
            removed.append(r.label())
    profile.routes = kept
    return removed


def _explicit_city_list(text: str) -> List[str]:
    """Parse comma-separated city lists from mission prose."""
    tl = text or ""
    m = re.search(
        r"\b(?:covering|between|including)\s+(.+?)(?:,\s*but\b|\s+but\b|\.\s*How|\.\s*What|\.\s*$)",
        tl,
        re.I,
    )
    if not m:
        return []
    blob = m.group(1).strip().rstrip(".")
    cities: List[str] = []
    from services.mission.route_extractor import resolve_place

    for part in re.split(r"\s*,\s*|\s+and\s+", blob):
        part = part.strip()
        if not part or len(part) < 3:
            continue
        if part.lower().startswith("caribbean"):
            place, conf = resolve_place("Caribbean")
        else:
            place, conf = resolve_place(part)
        if place and conf >= 0.72:
            if place.canonical not in cities:
                cities.append(place.canonical)
    return cities


_MOUNTAIN_CITY_NAMES = frozenset(
    {"aspen", "telluride", "jackson hole", "vail", "eagle", "sun valley", "jackson"}
)
_ME_CITY_NAMES = frozenset(
    {"dubai", "abu dhabi", "doha", "riyadh", "jeddah", "singapore", "tokyo", "hong kong"}
)


def _valid_preservation_link(anchor: str, dest: str) -> bool:
    a, d = anchor.lower(), dest.lower()
    if a in _MOUNTAIN_CITY_NAMES and d in _ME_CITY_NAMES:
        return False
    if a in _ME_CITY_NAMES and d == "caribbean":
        return False
    if a in _MOUNTAIN_CITY_NAMES and d == "caribbean":
        return False
    return True


def _ensure_explicit_cities_in_graph(
    query: str,
    profile: MissionProfile,
    lock: ExplicitRouteLock,
) -> List[str]:
    """Ensure every city named in an explicit list appears in at least one edge."""
    from services.mission.route_extractor import _build_route_extraction

    cities = _explicit_city_list(query)
    if len(cities) < 2:
        return []

    added: List[str] = []
    nodes_in_graph = set()
    for r in profile.routes:
        nodes_in_graph.add(r.origin.lower())
        nodes_in_graph.add(r.destination.lower())

    existing = {(r.origin.lower(), r.destination.lower()) for r in profile.routes}
    for i, city in enumerate(cities):
        if city.lower() in nodes_in_graph:
            continue
        anchor = None
        for j in range(i - 1, -1, -1):
            prior = cities[j]
            if prior.lower() in nodes_in_graph and _valid_preservation_link(prior, city):
                anchor = prior
                break
        if not anchor:
            continue
        ext = _build_route_extraction(anchor, city, pattern_boost=0.1)
        if not ext:
            continue
        key = (ext.route.origin.lower(), ext.route.destination.lower())
        if key in existing:
            continue
        if lock.is_locked(anchor, city):
            profile.routes.append(ext.route)
            existing.add(key)
            nodes_in_graph.add(city.lower())
            added.append(ext.route.label())
            continue
        # Valid ULR / executive hop between listed cities
        profile.routes.append(ext.route)
        existing.add(key)
        nodes_in_graph.add(city.lower())
        added.append(ext.route.label())

    return added


def apply_geographic_graph_authority(
    query: str,
    profile: MissionProfile,
    mission: MissionState,
    *,
    data_used: Optional[Dict[str, Any]] = None,
    inferred_labels: Optional[List[str]] = None,
    blocked_edges: Optional[List[str]] = None,
) -> AuthoritativeMissionRouteGraph:
    """
    Final geographic authority pass — restore explicit routes, strip re-anchor conflicts.
    """
    from services.mission.geographic_route_intelligence import (
        _infer_executive_eu_overlay_routes,
        _strip_field_to_executive_ghost_edges,
    )

    du = data_used if isinstance(data_used, dict) else {}
    lock = extract_explicit_routes(query)
    du[EXPLICIT_ROUTE_LOCK_KEY] = lock.to_dict()

    merge_explicit_routes_into_profile(lock, profile)
    _strip_field_to_executive_ghost_edges(profile)
    stripped = strip_conflicting_inferred_routes(lock, profile)
    blocked = list(blocked_edges or [])
    if stripped:
        blocked.extend(f"re-anchor blocked: {s}" for s in stripped)

    if re.search(r"\b(?:paris|geneva|frankfurt|zurich|london)\b", query or "", re.I):
        existing = {(r.origin.lower(), r.destination.lower()) for r in profile.routes}
        _infer_executive_eu_overlay_routes(query, profile, existing)

    city_added = _ensure_explicit_cities_in_graph(query, profile, lock)
    if city_added:
        blocked.extend(f"explicit city preserved: {a}" for a in city_added)

    mountain_removed = _dedupe_mountain_hub_spokes(profile, query)
    if mountain_removed:
        blocked.extend(f"duplicate mountain hub: {m}" for m in mountain_removed)

    # Strip cross-domain ghosts reintroduced by city-list preservation
    ghost_removed: List[str] = []
    kept_routes: List[Route] = []
    for r in profile.routes:
        if not _valid_preservation_link(r.origin, r.destination):
            ghost_removed.append(r.label())
            continue
        kept_routes.append(r)
    if ghost_removed:
        profile.routes = kept_routes
        blocked.extend(f"ghost re-strip: {g}" for g in ghost_removed)

    graph = build_authoritative_graph(
        query,
        profile,
        lock=lock,
        inferred_labels=inferred_labels,
        blocked_edges=blocked,
    )

    profile.routes = [
        Route(origin=e.origin, destination=e.destination) for e in graph.edges
    ]
    mission.routes = normalize_routes(graph.route_labels())

    du[GEOGRAPHIC_GRAPH_AUTHORITY_KEY] = graph.to_dict()
    du[EXPLICIT_ROUTE_LOCK_KEY] = lock.to_dict()
    return graph


__all__ = [
    "GEOGRAPHIC_GRAPH_AUTHORITY_KEY",
    "AuthoritativeMissionRouteGraph",
    "GraphEdge",
    "apply_geographic_graph_authority",
    "build_authoritative_graph",
]
