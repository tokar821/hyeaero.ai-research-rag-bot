"""
MissionGraph stabilization — geographic + routing consistency ONLY.

Runs after mission understanding semantics and after geographic graph authority,
but before recommendation quality.

Strict rules:
- Never re-anchor existing edges (A->B must remain A->B)
- Preserve multi-hub structure (no hub collapse)
- Restore dropped nodes as isolated nodes (unless explicitly invalid/removed upstream)
- Never flip direction unless explicitly stated in user text (handled earlier)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from services.mission.models import MissionProfile, Route
from services.mission.route_extractor import resolve_place

MISSION_GRAPH_STABILIZED_KEY = "mission_graph_stabilized"

_CONTINUATION_HUBS = frozenset({"dubai", "doha", "singapore", "frankfurt"})


@dataclass
class MissionGraphStabilized:
    nodes: List[str] = field(default_factory=list)
    edges: List[Dict[str, str]] = field(default_factory=list)  # {"origin","destination"}
    hub_clusters: List[Dict[str, Any]] = field(default_factory=list)
    invalid_reanchors_detected: List[str] = field(default_factory=list)
    dropped_nodes_restored: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "edges": list(self.edges),
            "hub_clusters": list(self.hub_clusters),
            "invalid_reanchors_detected": list(self.invalid_reanchors_detected),
            "dropped_nodes_restored": list(self.dropped_nodes_restored),
        }


def _extract_named_places(query: str) -> List[str]:
    """Resolve named places in query into canonical node names."""
    tl = query or ""
    found: List[str] = []

    # Prefer the catalog-ordered span enumerator (best recall for embedded phrases like
    # "flights from New York" or "operate Lagos offshore rigs").
    try:
        from services.mission.mission_corridor_routes import enumerate_ordered_places

        for p in enumerate_ordered_places(tl):
            if p and getattr(p, "canonical", None) and p.canonical not in found:
                found.append(p.canonical)
    except Exception:
        pass

    # Fallback fragment resolution
    if not found:
        parts = re.split(r"[,\n;/]+|\band\b", tl, flags=re.I)
        for part in parts:
            frag = part.strip()
            if not frag or len(frag) < 3:
                continue
            place, conf = resolve_place(frag)
            if place and conf >= 0.72 and place.canonical not in found:
                found.append(place.canonical)

    # Always attempt key continuation hubs if mentioned as plain tokens
    for hub in ("Dubai", "Doha", "Singapore", "Frankfurt"):
        if re.search(rf"\b{re.escape(hub)}\b", tl, re.I) and hub not in found:
            found.append(hub)
    return found


def _edge_pairs(routes: Sequence[Route]) -> Set[Tuple[str, str]]:
    return {(r.origin, r.destination) for r in routes if r and r.origin and r.destination}


def _detect_reanchors(
    *,
    original_edges: Set[Tuple[str, str]],
    final_edges: Set[Tuple[str, str]],
) -> List[str]:
    """Detect when a destination remained but origin changed (re-anchor)."""
    invalid: List[str] = []
    by_dest_orig: Dict[str, Set[str]] = {}
    for o, d in original_edges:
        by_dest_orig.setdefault(d.lower(), set()).add(o.lower())
    for o, d in final_edges:
        origs = by_dest_orig.get(d.lower())
        if not origs:
            continue
        if o.lower() not in origs and len(origs) >= 1:
            invalid.append(f"reanchor:{o} -> {d} (expected origins: {sorted(origs)})")
    return invalid


def _cluster_hubs(
    nodes: Sequence[str],
    edges: Set[Tuple[str, str]],
    *,
    named_places: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    Return independent hub clusters.

    Rules:
    - Preserve multi-hub missions even when some hubs have no surviving edges
      (include named hubs as clusters).
    - Continuation hubs cannot become primary hubs.
    """
    origin_nodes = {o for (o, _d) in edges}
    named_nodes = {n for n in named_places if n}
    hubs: List[str] = []
    for n in nodes:
        if n not in origin_nodes and n not in named_nodes:
            continue
        nl = (n or "").lower()
        place, conf = resolve_place(n)
        if place and conf >= 0.72 and place.kind == "city":
            if nl in _CONTINUATION_HUBS:
                continue
            hubs.append(place.canonical)
    hubs = list(dict.fromkeys(hubs))
    return [{"hub": h, "nodes": [h]} for h in hubs]


def stabilize_mission_graph(
    *,
    query: str,
    profile: MissionProfile,
    original_route_labels: Optional[Sequence[str]] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> MissionGraphStabilized:
    """
    Stabilize graph WITHOUT changing mission semantics:
    - restore original edges if missing
    - restore named nodes if missing
    - record invalid re-anchors
    """
    original_routes: List[Route] = []
    for lbl in (original_route_labels or []):
        r = Route.from_label(lbl)
        if r:
            original_routes.append(r)

    orig_edges = _edge_pairs(original_routes)
    final_edges = _edge_pairs(profile.routes)

    # Restore any original edges that were dropped
    restored_edges: List[Tuple[str, str]] = []
    for o, d in orig_edges:
        if (o, d) not in final_edges:
            profile.routes.append(Route(origin=o, destination=d))
            final_edges.add((o, d))
            restored_edges.append((o, d))

    invalid_reanchors = _detect_reanchors(original_edges=orig_edges, final_edges=final_edges)

    # Nodes from edges + named places in query
    nodes: Set[str] = set()
    for o, d in final_edges:
        nodes.add(o)
        nodes.add(d)

    named_places = _extract_named_places(query)
    dropped_restored: List[str] = []
    for named in named_places:
        if named not in nodes:
            nodes.add(named)
            dropped_restored.append(named)

    stabilized = MissionGraphStabilized(
        nodes=sorted(nodes),
        edges=[{"origin": o, "destination": d} for (o, d) in sorted(final_edges)],
        hub_clusters=_cluster_hubs(sorted(nodes), final_edges, named_places=named_places),
        invalid_reanchors_detected=invalid_reanchors,
        dropped_nodes_restored=sorted(dropped_restored),
    )

    if isinstance(data_used, dict):
        data_used[MISSION_GRAPH_STABILIZED_KEY] = stabilized.to_dict()

    return stabilized


__all__ = [
    "MISSION_GRAPH_STABILIZED_KEY",
    "MissionGraphStabilized",
    "stabilize_mission_graph",
]

