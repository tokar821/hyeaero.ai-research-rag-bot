"""
Arctic / Northern Canada logistics layer — deterministic hub→spoke enrichment.

region: arctic_industrial_layer
nodes:
  - Yellowknife (hub)
  - Nunavut Field Ops
  - Northern Alberta Oil Fields (Calgary extension)
  - Remote Gravel Strips (arctic gravel / ice strip operations)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from services.mission.hub_selection import select_local_hub
from services.mission.models import MissionProfile, Route
from services.mission.route_extractor import _build_route_extraction, resolve_place

ARCTIC_INDUSTRIAL_LAYER = "arctic_industrial_layer"

# Canonical node names (catalog-aligned)
NODE_YELLOWKNIFE = "Yellowknife"
NODE_NUNAVUT_FIELD_OPS = "Nunavut Field Ops"
NODE_NORTHERN_ALBERTA_OIL_FIELDS = "Northern Alberta Oil Fields"
NODE_REMOTE_GRAVEL_STRIPS = "Remote Gravel Strips"

ARCTIC_LAYER_NODES: Tuple[str, ...] = (
    NODE_YELLOWKNIFE,
    NODE_NUNAVUT_FIELD_OPS,
    NODE_NORTHERN_ALBERTA_OIL_FIELDS,
    NODE_REMOTE_GRAVEL_STRIPS,
)

_ARCTIC_LAYER_TRIGGER_RE = re.compile(
    r"\b(?:northern\s+canada|yellowknife|nunavut|northern\s+alberta|"
    r"calgary\s+oil\s+fields?|arctic\s+(?:gravel|ice)\s+strips?|"
    r"arctic\s+drilling\s+sites?|"
    r"gravel\s+strips?\s+(?:in\s+)?(?:northern\s+canada|nunavut|the\s+arctic)|"
    r"ice\s+strip\s+operations?|northern\s+canada\s+(?:logistics|operations?))\b",
    re.I,
)
_ARCTIC_DRILLING_SITES_RE = re.compile(
    r"\barctic\s+drilling\s+sites?\b|\bdrilling\s+sites?\s+in\s+(?:northern\s+canada|the\s+arctic)\b",
    re.I,
)

_NORTHERN_ALBERTA_RE = re.compile(
    r"\b(?:northern\s+alberta|calgary\s+oil\s+fields?|oil\s+fields?\s+(?:near|in)\s+calgary)\b",
    re.I,
)
_NUNAVUT_RE = re.compile(r"\bnunavut\b", re.I)
_NORTHERN_CANADA_GRAVEL_RE = re.compile(
    r"\b(?:gravel\s+strips?|ice\s+strips?)\b.*\b(?:northern\s+canada|nunavut|arctic|yellowknife)\b"
    r"|\b(?:northern\s+canada|nunavut|arctic)\b.*\b(?:gravel\s+strips?|ice\s+strips?)\b",
    re.I,
)
_GRAVEL_ARCTIC_RE = re.compile(
    r"\b(?:gravel\s+strips?|ice\s+strips?|arctic\s+gravel)\b",
    re.I,
)


def arctic_layer_active(text: str) -> bool:
    return bool(_ARCTIC_LAYER_TRIGGER_RE.search(text or ""))


def _route_touching(profile: MissionProfile, city: str) -> bool:
    c = city.lower()
    for r in profile.routes:
        if r.origin.lower() == c or r.destination.lower() == c:
            return True
    return False


def _pick_arctic_hub(
    profile: MissionProfile,
    text: str,
    node: str,
) -> str:
    tl = text or ""
    if node == NODE_NORTHERN_ALBERTA_OIL_FIELDS:
        return "Calgary"
    if node == NODE_NUNAVUT_FIELD_OPS:
        return select_local_hub(
            profile,
            tl,
            ("Yellowknife", "Calgary"),
            mission_type="industrial",
            regional_bias=("Yellowknife",),
            default="Yellowknife",
        )
    if node == NODE_REMOTE_GRAVEL_STRIPS:
        if _NORTHERN_CANADA_GRAVEL_RE.search(tl) or _NUNAVUT_RE.search(tl):
            return select_local_hub(
                profile,
                tl,
                ("Yellowknife", "Calgary"),
                mission_type="industrial",
                regional_bias=("Yellowknife",),
                default="Yellowknife",
            )
        return select_local_hub(
            profile,
            tl,
            ("Calgary", "Yellowknife", "Anchorage"),
            mission_type="industrial",
            default="Calgary",
        )
    return select_local_hub(
        profile,
        tl,
        ("Yellowknife", "Calgary", "Houston"),
        mission_type="industrial",
        default="Yellowknife",
    )


def _add_route(
    profile: MissionProfile,
    origin: str,
    destination: str,
    *,
    existing: Set[Tuple[str, str]],
    boost: float = 0.1,
) -> Optional[Route]:
    ext = _build_route_extraction(origin, destination, pattern_boost=boost)
    if not ext:
        return None
    key = (ext.route.origin.lower(), ext.route.destination.lower())
    if key in existing:
        return None
    existing.add(key)
    profile.routes.append(ext.route)
    return ext.route


def infer_arctic_industrial_layer_routes(
    text: str,
    profile: MissionProfile,
    existing: Set[Tuple[str, str]],
) -> List[str]:
    """
    Hub → arctic logistics spokes; Calgary ↔ Yellowknife when both domains appear.
    """
    tl = text or ""
    if not arctic_layer_active(tl):
        return []

    added: List[str] = []

    # Calgary extension hub link when northern network spans Alberta + NWT
    calgary_in_mission = (
        _route_touching(profile, "Calgary")
        or bool(re.search(r"\bcalgary\b", tl, re.I))
    )
    yellowknife_in_mission = (
        _route_touching(profile, "Yellowknife")
        or bool(re.search(r"\byellowknife\b", tl, re.I))
    )
    northern_canada = bool(re.search(r"\bnorthern\s+canada\b|\bnunavut\b", tl, re.I))

    if calgary_in_mission and (yellowknife_in_mission or northern_canada):
        r = _add_route(profile, "Calgary", "Yellowknife", existing=existing, boost=0.09)
        if r:
            added.append(r.label())

    nodes_to_add: List[str] = []

    if _NORTHERN_ALBERTA_RE.search(tl) or (
        calgary_in_mission and re.search(r"\boil\s+fields?\b", tl, re.I)
    ):
        nodes_to_add.append(NODE_NORTHERN_ALBERTA_OIL_FIELDS)

    if _NUNAVUT_RE.search(tl):
        nodes_to_add.append(NODE_NUNAVUT_FIELD_OPS)

    if _ARCTIC_DRILLING_SITES_RE.search(tl) or (
        northern_canada and re.search(r"\bdrilling\s+sites?\b", tl, re.I)
    ):
        nodes_to_add.append(NODE_REMOTE_GRAVEL_STRIPS)
        if calgary_in_mission:
            nodes_to_add.append(NODE_NORTHERN_ALBERTA_OIL_FIELDS)

    if _NORTHERN_CANADA_GRAVEL_RE.search(tl) or (
        northern_canada and _GRAVEL_ARCTIC_RE.search(tl)
    ):
        nodes_to_add.append(NODE_REMOTE_GRAVEL_STRIPS)
    elif _GRAVEL_ARCTIC_RE.search(tl) and re.search(
        r"\b(?:arctic|northern|nunavut|yellowknife)\b", tl, re.I
    ):
        nodes_to_add.append(NODE_REMOTE_GRAVEL_STRIPS)

    seen_nodes: Set[str] = set()
    for node in nodes_to_add:
        if node in seen_nodes:
            continue
        seen_nodes.add(node)
        hub = _pick_arctic_hub(profile, tl, node)
        r = _add_route(profile, hub, node, existing=existing, boost=0.11)
        if r:
            added.append(r.label())

    # Ensure Yellowknife hub appears when layer active but only spokes were inferred
    if added and not _route_touching(profile, "Yellowknife"):
        yk_place, conf = resolve_place("Yellowknife")
        if yk_place and conf >= 0.72 and northern_canada:
            for r in list(profile.routes):
                if r.destination.lower() in {
                    n.lower() for n in (
                        NODE_NUNAVUT_FIELD_OPS,
                        NODE_REMOTE_GRAVEL_STRIPS,
                    )
                }:
                    hub = r.origin
                    if hub.lower() != "yellowknife":
                        _add_route(
                            profile, hub, "Yellowknife", existing=existing, boost=0.08
                        )
                    break

    return added


def rebalance_arctic_hub_spokes(profile: MissionProfile) -> List[str]:
    """Keep mission-origin-correct hub for each arctic node; drop foreign-hub duplicates."""
    removed: List[str] = []
    node_canonical_hub: Dict[str, str] = {
        NODE_NORTHERN_ALBERTA_OIL_FIELDS: "Calgary",
        NODE_NUNAVUT_FIELD_OPS: "Yellowknife",
    }
    for node, canonical_hub in node_canonical_hub.items():
        spokes = [r for r in profile.routes if r.destination == node]
        if len(spokes) <= 1:
            continue
        kept = [r for r in profile.routes if r.destination != node or r.origin == canonical_hub]
        for r in profile.routes:
            if r.destination == node and r.origin != canonical_hub:
                removed.append(r.label())
        profile.routes = kept
    return removed


__all__ = [
    "ARCTIC_INDUSTRIAL_LAYER",
    "ARCTIC_LAYER_NODES",
    "arctic_layer_active",
    "infer_arctic_industrial_layer_routes",
    "rebalance_arctic_hub_spokes",
]
