"""
Mission route graph — all corridor legs + inferred ULR continuations before ranking.

Middle East hubs (Abu Dhabi, Dubai, Riyadh, Doha) are first-class nodes, not prose-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from services.consultant.mission_state import MissionState, normalize_routes
from services.mission.models import MissionProfile, Route
from services.mission.hub_selection import is_me_continuation_hub, select_local_hub
from services.mission.route_extractor import _build_route_extraction, dedupe_extractions, resolve_place
from services.mission.route_directionality import literal_direction_in_text

MISSION_ROUTE_GRAPH_KEY = "mission_route_graph"

# First-class continuation nodes (canonical catalog names)
CONTINUATION_HUBS: Tuple[str, ...] = (
    "Abu Dhabi",
    "Dubai",
    "Riyadh",
    "Doha",
)

_CONTINUATION_MENTION_RE = re.compile(
    r"\b(?:abu\s+dhabi|dubai|riyadh|doha|jeddah|middle\s+east)\b",
    re.I,
)
_NONSTOP_CONTINUATION_RE = re.compile(
    r"\bnonstop\s+(?:to\s+)?(abu\s+dhabi|dubai|riyadh|doha)\b"
    r"|\b(?:founder|ceo|chairman)\b.*\b(?:abu\s+dhabi|dubai|riyadh)\b"
    r"|\b(?:abu\s+dhabi|dubai|riyadh)\b.*\bnonstop\b",
    re.I,
)

_US_ORIGIN_HUBS: Tuple[str, ...] = (
    "New York",
    "Boston",
    "Chicago",
    "Los Angeles",
    "San Francisco",
    "Miami",
    "Houston",
    "Dallas",
    "Teterboro",
    "Washington",
)

_CANONICAL_ME: Dict[str, str] = {
    "abu dhabi": "Abu Dhabi",
    "dubai": "Dubai",
    "riyadh": "Riyadh",
    "doha": "Doha",
    "jeddah": "Jeddah",
}


@dataclass
class RouteGraphNode:
    canonical: str
    node_kind: str  # city | region | continuation_hub
    country: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical": self.canonical,
            "node_kind": self.node_kind,
            "country": self.country,
        }


@dataclass
class MissionRouteGraph:
    nodes: List[RouteGraphNode] = field(default_factory=list)
    legs: List[Route] = field(default_factory=list)
    continuation_legs: List[Route] = field(default_factory=list)
    inferred_leg_labels: List[str] = field(default_factory=list)

    def all_legs(self) -> List[Route]:
        seen: Set[Tuple[str, str]] = set()
        out: List[Route] = []
        for r in list(self.legs) + list(self.continuation_legs):
            key = (r.origin.lower(), r.destination.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "legs": [r.to_dict() for r in self.legs],
            "continuation_legs": [r.to_dict() for r in self.continuation_legs],
            "inferred_leg_labels": list(self.inferred_leg_labels),
            "route_labels": [r.label() for r in self.all_legs()],
        }


def _mentioned_continuation_cities(text: str) -> List[str]:
    tl = (text or "").lower()
    out: List[str] = []
    for alias, canon in _CANONICAL_ME.items():
        if re.search(rf"\b{re.escape(alias)}\b", tl) and canon not in out:
            out.append(canon)
    return out


def _route_touching_city(routes: List[Route], city: str) -> bool:
    c = city.lower()
    for r in routes:
        if r.origin.lower() == c or r.destination.lower() == c:
            return True
    return False


def _pick_us_origin(routes: List[Route], text: str) -> Optional[str]:
    """Mission-origin local hub — never a ME continuation city as default origin."""
    origin = select_local_hub(
        routes,
        text,
        _US_ORIGIN_HUBS,
        mission_type="executive",
        default="New York",
    )
    if is_me_continuation_hub(origin):
        for hub in _US_ORIGIN_HUBS:
            if _route_touching_city(routes, hub):
                return hub
        return "New York"
    return origin


def infer_continuation_legs(
    text: str,
    profile: MissionProfile,
) -> List[Route]:
    """Add hub → ME legs from established origin clusters — never single-hub takeover."""
    existing = list(profile.routes)
    mentioned = _mentioned_continuation_cities(text)
    if not mentioned:
        return []

    needs_inference = (
        bool(_NONSTOP_CONTINUATION_RE.search(text or ""))
        or bool(_CONTINUATION_MENTION_RE.search(text or ""))
    )
    if not needs_inference:
        return []

    candidate_origins: List[str] = []
    for r in existing:
        if not is_me_continuation_hub(r.origin):
            candidate_origins.append(r.origin)
    candidate_origins = list(dict.fromkeys(candidate_origins))
    if not candidate_origins:
        picked = _pick_us_origin(existing, text)
        if picked:
            candidate_origins = [picked]

    inferred: List[Route] = []
    existing_pairs = {(r.origin.lower(), r.destination.lower()) for r in existing}
    for origin in candidate_origins:
        for dest in mentioned:
            if any(r.destination.lower() == dest.lower() for r in existing):
                continue
            o, d = origin, dest
            stated = literal_direction_in_text(o, d, text)
            if stated is False:
                o, d = dest, o
            key = (o.lower(), d.lower())
            if key in existing_pairs:
                continue
            ext = _build_route_extraction(o, d, pattern_boost=0.11)
            if ext:
                inferred.append(ext.route)
                existing_pairs.add(key)
    return inferred


def build_route_graph(
    text: str,
    profile: MissionProfile,
) -> MissionRouteGraph:
    """Merge explicit profile legs + inferred continuations into one graph."""
    legs = list(profile.routes)
    continuation = infer_continuation_legs(text, profile)

    nodes_map: Dict[str, RouteGraphNode] = {}
    for r in legs + continuation:
        for name in (r.origin, r.destination):
            place, conf = resolve_place(name)
            if not place or conf < 0.72:
                continue
            kind = (
                "continuation_hub"
                if place.canonical in CONTINUATION_HUBS
                else place.kind
            )
            nodes_map[place.canonical] = RouteGraphNode(
                canonical=place.canonical,
                node_kind=kind,
                country=place.country,
            )

    graph = MissionRouteGraph(
        nodes=list(nodes_map.values()),
        legs=legs,
        continuation_legs=continuation,
        inferred_leg_labels=[r.label() for r in continuation],
    )
    return graph


def merge_route_graph_into_mission(
    graph: MissionRouteGraph,
    profile: MissionProfile,
    mission: MissionState,
) -> None:
    """Authoritative route list for ranking — explicit + inferred continuations."""
    merged = graph.all_legs()
    if not merged:
        return
    profile.routes = merged
    mission.routes = normalize_routes([r.label() for r in merged])


def save_route_graph(data_used: Dict[str, Any], graph: MissionRouteGraph) -> None:
    data_used[MISSION_ROUTE_GRAPH_KEY] = graph.to_dict()


def load_route_graph(data_used: Optional[Dict[str, Any]]) -> Optional[MissionRouteGraph]:
    if not isinstance(data_used, dict):
        return None
    raw = data_used.get(MISSION_ROUTE_GRAPH_KEY)
    if not isinstance(raw, dict):
        return None
    nodes = [
        RouteGraphNode(
            canonical=str(n.get("canonical") or ""),
            node_kind=str(n.get("node_kind") or "city"),
            country=n.get("country"),
        )
        for n in (raw.get("nodes") or [])
        if isinstance(n, dict)
    ]
    legs = []
    for key in ("legs", "continuation_legs"):
        for item in raw.get(key) or []:
            if isinstance(item, dict):
                try:
                    legs.append(Route(origin=item["origin"], destination=item["destination"]))
                except (KeyError, ValueError):
                    pass
    cont = []
    for item in raw.get("continuation_legs") or []:
        if isinstance(item, dict):
            try:
                cont.append(Route(origin=item["origin"], destination=item["destination"]))
            except (KeyError, ValueError):
                pass
    g = MissionRouteGraph(nodes=nodes, legs=[], continuation_legs=cont)
    g.legs = [
        Route.from_label(lbl)
        for lbl in (raw.get("route_labels") or [])
        if Route.from_label(lbl)
    ]
    if not g.legs:
        g.legs = [r for r in legs if r not in cont]
    return g
