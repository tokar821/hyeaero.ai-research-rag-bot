"""
Route topology realism — deterministic edge discipline after pre-ranking representation.

Prevents corridor/anchor propagation from stitching unrelated operational domains into
synthetic impossible itineraries while preserving legitimate hub spokes and continuations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from services.consultant.mission_state import MissionState, normalize_routes
from services.mission.models import MissionProfile, Route
from services.mission.route_extractor import resolve_place

ROUTE_TOPOLOGY_KEY = "route_topology_validation"
RENDER_CONFIDENCE_MIN = 0.75

AuthorityType = Literal["explicit", "continuation", "hub_inferred", "domain_bridge"]

# Endpoint operational classes
DOMAIN_MOUNTAIN = "mountain_field"
DOMAIN_INDUSTRIAL = "industrial_field"
DOMAIN_CARIBBEAN = "caribbean_island"
DOMAIN_FIELD_REGION = "field_region"
DOMAIN_EXECUTIVE = "executive_hub"
DOMAIN_ME_CONTINUATION = "me_continuation"
DOMAIN_DOMESTIC = "domestic_executive"
DOMAIN_UNKNOWN = "unknown"

_MOUNTAIN_RE = re.compile(
    r"\b(?:aspen|telluride|jackson\s+hole|sun\s+valley|eagle\s+county|"
    r"kase|ktex|kege|kjac)\b",
    re.I,
)
_INDUSTRIAL_RE = re.compile(
    r"\b(?:remote\s+drilling|arctic\s+oil|gravel|unpaved|mining\s+strip|oil\s+field)\b",
    re.I,
)
_FIELD_REGION_RE = re.compile(
    r"\b(?:remote\s+drilling\s+sites?|arctic\s+oil\s+platforms?|west\s+africa|"
    r"nunavut\s+field\s+ops|northern\s+alberta\s+oil\s+fields?|remote\s+gravel\s+strips)\b",
    re.I,
)
_CARIBBEAN_RE = re.compile(
    r"\b(?:caribbean|nassau|st\s+maarten|st\s+thomas|bahamas|tropical\s+islands?)\b",
    re.I,
)
_ME_RE = re.compile(
    r"\b(?:dubai|abu\s+dhabi|riyadh|doha|jeddah)\b",
    re.I,
)
_INTERCONTINENTAL_CITY_RE = re.compile(
    r"\b(?:london|paris|frankfurt|zurich|geneva|tokyo|singapore|hong\s+kong|seoul|"
    r"madrid|beijing|shanghai|sydney|mumbai|dubai|abu\s+dhabi|riyadh|doha)\b",
    re.I,
)
_FOUNDER_HUB_RE = re.compile(r"\b(?:new\s+york|nyc|teterboro)\b", re.I)
_FOUNDER_ULR_ASYMMETRY_RE = re.compile(
    r"\bfounder\b.*\b(?:nonstop|insists?)\b.*\b(?:dubai|abu\s+dhabi|riyadh)\b"
    r"|\b(?:90\s*%|short\s+east\s+coast)\b.*\bfounder\b",
    re.I,
)
_EXPLICIT_EDGE_RE = re.compile(
    r"\b(?:from\s+)?({origin})\s*(?:->|→|—|–|to|-)\s*({dest})\b",
    re.I,
)

_EXECUTIVE_HUBS = frozenset(
    {
        "new york",
        "teterboro",
        "boston",
        "chicago",
        "los angeles",
        "san francisco",
        "miami",
        "houston",
        "dallas",
        "denver",
        "calgary",
        "anchorage",
        "washington",
        "london",
        "paris",
        "frankfurt",
        "zurich",
        "geneva",
        "madrid",
        "tokyo",
        "singapore",
        "hong kong",
        "seoul",
        "sao paulo",
        "lagos",
    }
)

# Only South Florida hubs may anchor Caribbean spokes (never Pacific/ME/other exec)
_CARIBBEAN_ANCHOR_HUBS = frozenset(
    {
        "miami",
        "palm beach",
        "fort lauderdale",
        "west palm",
    }
)

_PACIFIC_ULR_EXEC_ORIGINS = frozenset(
    {
        "tokyo",
        "singapore",
        "hong kong",
        "seoul",
        "beijing",
        "shanghai",
        "sydney",
        "los angeles",
        "san francisco",
    }
)

_EUROPEAN_FIELD_BLOCK_ORIGINS = frozenset(
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

# EU executive hubs may support stated African industrial regions — not US/Texas/arctic fields.
_EU_ALLOWED_FIELD_DESTINATIONS = frozenset(
    {
        "west africa",
        "nigerian energy corridor",
        "northern africa",
    }
)


@dataclass
class RouteTopologyEdge:
    origin: str
    destination: str
    authority: AuthorityType
    confidence: float
    operational_domain: str
    structurally_valid: bool
    edge_inference_type: str = ""
    rejection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": self.origin,
            "destination": self.destination,
            "authority": self.authority,
            "confidence": round(self.confidence, 3),
            "operational_domain": self.operational_domain,
            "structurally_valid": self.structurally_valid,
            "edge_inference_type": self.edge_inference_type,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class RouteTopologyReport:
    edges: List[RouteTopologyEdge] = field(default_factory=list)
    kept_routes: List[str] = field(default_factory=list)
    removed_routes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edges": [e.to_dict() for e in self.edges],
            "kept_routes": list(self.kept_routes),
            "removed_routes": list(self.removed_routes),
        }


def classify_endpoint(name: str) -> str:
    n = (name or "").strip()
    nl = n.lower()
    if _FIELD_REGION_RE.search(nl):
        return DOMAIN_FIELD_REGION
    if _MOUNTAIN_RE.search(nl):
        return DOMAIN_MOUNTAIN
    if nl == "caribbean" or _CARIBBEAN_RE.search(nl):
        return DOMAIN_CARIBBEAN
    if _INDUSTRIAL_RE.search(nl):
        return DOMAIN_INDUSTRIAL
    if _ME_RE.search(nl):
        return DOMAIN_ME_CONTINUATION
    place, conf = resolve_place(n)
    if place and conf >= 0.72:
        if place.kind == "region" and "caribbean" in place.canonical.lower():
            return DOMAIN_CARIBBEAN
        if place.kind == "region":
            return DOMAIN_FIELD_REGION
    if nl in _EXECUTIVE_HUBS:
        return DOMAIN_EXECUTIVE
    if _INTERCONTINENTAL_CITY_RE.search(nl):
        return DOMAIN_EXECUTIVE
    return DOMAIN_DOMESTIC if place and place.country == "US" else DOMAIN_UNKNOWN


def _domain_pair_tag(origin_dom: str, dest_dom: str) -> str:
    return f"{origin_dom}->{dest_dom}"


def _explicit_edge_in_text(origin: str, destination: str, text: str) -> bool:
    tl = (text or "").lower()
    o = origin.lower()
    d = destination.lower()
    patterns = (
        rf"\b{re.escape(o)}\s*(?:->|→|—|–|to|-)\s*{re.escape(d)}\b",
        rf"\bfrom\s+{re.escape(o)}\s+to\s+{re.escape(d)}\b",
        rf"\b{re.escape(o)}\s*[-–]\s*{re.escape(d)}\b",
    )
    for pat in patterns:
        if re.search(pat, tl):
            return True
    # Alias-aware: resolve canonical names
    op, _ = resolve_place(origin)
    dp, _ = resolve_place(destination)
    if op and dp:
        oc, dc = op.canonical.lower(), dp.canonical.lower()
        if re.search(rf"\b{re.escape(oc)}\s*(?:->|to|-)\s*{re.escape(dc)}\b", tl):
            return True
    return False


def _is_intercontinental(origin: str, destination: str) -> bool:
    try:
        from services.consultant.route_feasibility import estimate_route_distance_nm

        nm = float(estimate_route_distance_nm(f"{origin} -> {destination}") or 0)
        if nm >= 2500:
            return True
    except Exception:
        pass
    od = classify_endpoint(origin)
    dd = classify_endpoint(destination)
    if od == DOMAIN_EXECUTIVE and dd == DOMAIN_EXECUTIVE:
        o_ic = bool(_INTERCONTINENTAL_CITY_RE.search(origin))
        d_ic = bool(_INTERCONTINENTAL_CITY_RE.search(destination))
        if o_ic and d_ic and origin.lower() != destination.lower():
            # US exec to intercontinental exec
            op, _ = resolve_place(origin)
            if op and op.country == "US" and dd == DOMAIN_EXECUTIVE:
                return True
    return bool(
        _INTERCONTINENTAL_CITY_RE.search(destination)
        and classify_endpoint(origin) in (DOMAIN_EXECUTIVE, DOMAIN_DOMESTIC, DOMAIN_ME_CONTINUATION)
        and origin.lower() not in destination.lower()
    )


def _restricted_origin_domains() -> Set[str]:
    return {DOMAIN_MOUNTAIN, DOMAIN_INDUSTRIAL, DOMAIN_FIELD_REGION, DOMAIN_CARIBBEAN}


def _forbidden_cross_domain(
    origin_dom: str,
    dest_dom: str,
    *,
    origin: str = "",
    destination: str = "",
) -> Optional[str]:
    """Return rejection reason when domains must not connect directly."""
    restricted = _restricted_origin_domains()

    if origin_dom in restricted and dest_dom in (
        DOMAIN_EXECUTIVE,
        DOMAIN_ME_CONTINUATION,
    ):
        return "restricted_origin_to_executive_or_continuation"

    if origin_dom == DOMAIN_CARIBBEAN and dest_dom in (
        DOMAIN_ME_CONTINUATION,
        DOMAIN_EXECUTIVE,
    ):
        return "caribbean_to_intercontinental_without_hub"

    if origin_dom == DOMAIN_FIELD_REGION and dest_dom in (
        DOMAIN_EXECUTIVE,
        DOMAIN_ME_CONTINUATION,
    ):
        return "field_region_to_executive_without_hub"

    if origin_dom == DOMAIN_MOUNTAIN and dest_dom in (
        DOMAIN_CARIBBEAN,
        DOMAIN_ME_CONTINUATION,
    ):
        return "mountain_to_unrelated_domain"

    if origin_dom == DOMAIN_MOUNTAIN and dest_dom == DOMAIN_EXECUTIVE:
        return "mountain_to_executive_intercontinental"

    if origin_dom == DOMAIN_CARIBBEAN and dest_dom == DOMAIN_EXECUTIVE:
        return "caribbean_to_intercontinental_without_hub"

    if origin_dom == DOMAIN_EXECUTIVE and dest_dom == DOMAIN_FIELD_REGION:
        ol = origin.lower()
        dl = destination.lower()
        if ol in _EUROPEAN_FIELD_BLOCK_ORIGINS and dl not in _EU_ALLOWED_FIELD_DESTINATIONS:
            return "eu_executive_to_field_without_industrial_hub"

    if origin_dom == DOMAIN_EXECUTIVE and dest_dom == DOMAIN_MOUNTAIN:
        # Executive hub -> mountain is valid hub-spoke
        return None

    if origin_dom == DOMAIN_EXECUTIVE and dest_dom == DOMAIN_CARIBBEAN:
        ol = origin.lower()
        if ol not in _CARIBBEAN_ANCHOR_HUBS:
            if ol in _PACIFIC_ULR_EXEC_ORIGINS or _ME_RE.search(origin):
                return "pacific_me_to_caribbean_without_florida_hub"
            return "non_florida_caribbean_origin"
        return None

    if origin_dom == DOMAIN_ME_CONTINUATION and dest_dom == DOMAIN_CARIBBEAN:
        return "me_to_caribbean_without_florida_hub"

    if origin_dom == DOMAIN_EXECUTIVE and dest_dom == DOMAIN_FIELD_REGION:
        return None

    if origin_dom == DOMAIN_DOMESTIC and dest_dom == DOMAIN_ME_CONTINUATION:
        return None

    if origin_dom == DOMAIN_EXECUTIVE and dest_dom == DOMAIN_ME_CONTINUATION:
        return None

    if origin_dom in restricted and dest_dom in restricted and origin_dom != dest_dom:
        return "cross_restricted_domain_stitch"

    return None


def _allowed_continuation_origins(
    query: str,
    governance: Optional[Dict[str, Any]],
) -> Optional[Set[str]]:
    """When founder/CEO asymmetry applies, continuation may only depart credible mandate hubs."""
    gov = governance or {}
    asymmetry = (
        gov.get("founder_company_asymmetry")
        or (gov.get("ceo_ulr_mandate") and gov.get("domestic_utilization_dominant"))
        or bool(_FOUNDER_ULR_ASYMMETRY_RE.search(query or ""))
    )
    if asymmetry:
        if _FOUNDER_HUB_RE.search(query or ""):
            return {"new york", "teterboro", "nyc"}
        tl = query or ""
        for hub in ("new york", "los angeles", "miami", "houston", "chicago"):
            if re.search(rf"\b{re.escape(hub)}\b", tl, re.I) and re.search(
                r"\b(?:nonstop|ulr|dubai|abu\s+dhabi|riyadh|tokyo|singapore)\b", tl, re.I
            ):
                return {hub}
    return None


def _infer_authority(
    route: Route,
    *,
    inferred_labels: Set[str],
    anchor_labels: Set[str],
    field_labels: Set[str],
    domestic_labels: Set[str],
) -> AuthorityType:
    lbl = route.label()
    if lbl in inferred_labels:
        return "continuation"
    if lbl in field_labels:
        return "hub_inferred"
    if lbl in anchor_labels or lbl in domestic_labels:
        return "domain_bridge"
    return "explicit"


def validate_route_edge(
    route: Route,
    *,
    query: str,
    authority: AuthorityType,
    governance: Optional[Dict[str, Any]] = None,
) -> RouteTopologyEdge:
    origin, dest = route.origin, route.destination
    o_dom = classify_endpoint(origin)
    d_dom = classify_endpoint(dest)
    domain_tag = _domain_pair_tag(o_dom, d_dom)
    literal_explicit = _explicit_edge_in_text(origin, dest, query)
    intercontinental = _is_intercontinental(origin, dest)

    edge = RouteTopologyEdge(
        origin=origin,
        destination=dest,
        authority=authority,
        confidence=0.95 if literal_explicit else 0.85,
        operational_domain=domain_tag,
        structurally_valid=True,
        edge_inference_type=authority,
    )

    # Founder/CEO mandate — ME legs only from authorized hub unless literally stated
    allowed_origins = _allowed_continuation_origins(query, governance)
    if (
        allowed_origins is not None
        and d_dom == DOMAIN_ME_CONTINUATION
        and not literal_explicit
    ):
        ol = origin.lower()
        if not any(ol == h or ol.startswith(h) for h in allowed_origins):
            edge.structurally_valid = False
            edge.confidence = 0.42
            edge.rejection_reason = "me_leg_origin_not_founder_hub"
            return edge

    if literal_explicit:
        edge.confidence = 0.95
        return edge

    reason = _forbidden_cross_domain(
        o_dom, d_dom, origin=origin, destination=dest
    )
    if reason:
        edge.structurally_valid = False
        edge.confidence = 0.35
        edge.rejection_reason = reason
        return edge

    # Intercontinental non-US origin to mountain — requires literal statement
    if d_dom == DOMAIN_MOUNTAIN and intercontinental and not literal_explicit:
        op, _ = resolve_place(origin)
        if not op or op.country != "US":
            edge.structurally_valid = False
            edge.confidence = 0.4
            edge.rejection_reason = "intercontinental_to_mountain_without_us_hub"
            return edge

    # Continuation legs: hub anchor + no restricted origins
    if authority == "continuation":
        allowed_origins = _allowed_continuation_origins(query, governance)
        if allowed_origins is not None:
            ol = origin.lower()
            if not any(ol == h or ol.startswith(h) for h in allowed_origins):
                edge.structurally_valid = False
                edge.confidence = 0.4
                edge.rejection_reason = "continuation_origin_not_founder_hub"
                return edge
        if o_dom in _restricted_origin_domains():
            edge.structurally_valid = False
            edge.confidence = 0.3
            edge.rejection_reason = "continuation_from_restricted_airport"
            return edge
        if d_dom not in (DOMAIN_ME_CONTINUATION, DOMAIN_EXECUTIVE):
            edge.structurally_valid = False
            edge.confidence = 0.45
            edge.rejection_reason = "continuation_must_terminate_at_gateway"
            return edge
        edge.confidence = 0.88
        return edge

    # Hub-inferred spokes: must originate from executive/domestic hub
    if authority in ("hub_inferred", "domain_bridge"):
        if o_dom in _restricted_origin_domains():
            edge.structurally_valid = False
            edge.confidence = 0.35
            edge.rejection_reason = "inferred_edge_from_restricted_origin"
            return edge

        # Domain-bridge stitching across unrelated domains
        if authority == "domain_bridge" and intercontinental:
            if o_dom in (DOMAIN_MOUNTAIN, DOMAIN_CARIBBEAN, DOMAIN_FIELD_REGION):
                edge.structurally_valid = False
                edge.confidence = 0.4
                edge.rejection_reason = "domain_bridge_cross_domain"
                return edge

        # Mountain/industrial destinations must be hub-spoke from executive city
        if d_dom in (DOMAIN_MOUNTAIN, DOMAIN_FIELD_REGION, DOMAIN_INDUSTRIAL):
            if o_dom not in (DOMAIN_EXECUTIVE, DOMAIN_DOMESTIC):
                edge.structurally_valid = False
                edge.confidence = 0.45
                edge.rejection_reason = "field_spoke_requires_executive_hub_origin"
                return edge
            edge.confidence = 0.82
            return edge

        if d_dom == DOMAIN_CARIBBEAN and o_dom in (DOMAIN_EXECUTIVE, DOMAIN_DOMESTIC):
            edge.confidence = 0.85
            return edge

        # Cross-domain bridge without shared hub context
        if o_dom != d_dom and o_dom in _restricted_origin_domains() | {DOMAIN_CARIBBEAN}:
            edge.structurally_valid = False
            edge.confidence = 0.35
            edge.rejection_reason = "domain_stitch_without_explicit_authority"
            return edge

        edge.confidence = 0.78
        return edge

    # Explicit extraction but not stated as literal edge — still validate cross-domain
    if intercontinental and o_dom in _restricted_origin_domains():
        edge.structurally_valid = False
        edge.confidence = 0.4
        edge.rejection_reason = "implicit_cross_domain_intercontinental"
        return edge

    edge.confidence = 0.9
    return edge


def validate_route_topology(
    query: str,
    profile: MissionProfile,
    mission: MissionState,
    *,
    governance: Optional[Dict[str, Any]] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> RouteTopologyReport:
    """
    Filter profile/mission routes to structurally valid edges only.
    """
    du = data_used if isinstance(data_used, dict) else {}
    pre = du.get("pre_ranking_representation") or {}
    route_graph_snap = pre.get("route_graph") or {}
    inferred_labels = set(route_graph_snap.get("inferred") or [])
    anchor_labels = set(pre.get("anchor_routes_added") or [])
    field_labels = set(pre.get("field_access_spokes") or [])
    domestic_labels = set(pre.get("domestic_triangle_added") or [])

    if not governance and isinstance(du.get("mission_governance"), dict):
        governance = du["mission_governance"]

    from services.mission.explicit_route_lock import extract_explicit_routes

    explicit_lock = extract_explicit_routes(query)

    report = RouteTopologyReport()
    kept: List[Route] = []

    for route in list(profile.routes or []):
        if explicit_lock.is_locked_route(route):
            edge = RouteTopologyEdge(
                origin=route.origin,
                destination=route.destination,
                authority="explicit",
                confidence=0.98,
                operational_domain=_domain_pair_tag(
                    classify_endpoint(route.origin),
                    classify_endpoint(route.destination),
                ),
                structurally_valid=True,
                edge_inference_type="explicit",
            )
            report.edges.append(edge)
            kept.append(route)
            report.kept_routes.append(route.label())
            continue

        authority = _infer_authority(
            route,
            inferred_labels=inferred_labels,
            anchor_labels=anchor_labels,
            field_labels=field_labels,
            domestic_labels=domestic_labels,
        )
        edge = validate_route_edge(
            route,
            query=query,
            authority=authority,
            governance=governance,
        )
        report.edges.append(edge)
        lbl = route.label()
        if edge.structurally_valid and edge.confidence >= RENDER_CONFIDENCE_MIN:
            kept.append(route)
            report.kept_routes.append(lbl)
        else:
            report.removed_routes.append(lbl)

    if kept:
        profile.routes = kept
        mission.routes = normalize_routes([r.label() for r in kept])
    elif profile.routes:
        # Keep highest-confidence explicit edges rather than empty graph
        explicit_edges = [e for e in report.edges if e.authority == "explicit" and e.structurally_valid]
        if explicit_edges:
            best = max(explicit_edges, key=lambda e: e.confidence)
            r = Route(origin=best.origin, destination=best.destination)
            profile.routes = [r]
            mission.routes = normalize_routes([r.label()])
            report.kept_routes = [r.label()]

    du[ROUTE_TOPOLOGY_KEY] = report.to_dict()
    return report


def apply_route_topology_validation(
    query: str,
    profile: MissionProfile,
    mission: MissionState,
    *,
    packet=None,
    data_used: Optional[Dict[str, Any]] = None,
) -> Tuple[MissionProfile, MissionState, RouteTopologyReport]:
    """Entry point — run after pre-ranking, before mission graph build."""
    governance = None
    if isinstance(data_used, dict):
        governance = data_used.get("mission_governance")
    report = validate_route_topology(
        query,
        profile,
        mission,
        governance=governance,
        data_used=data_used,
    )
    if packet is not None and hasattr(packet, "explicit_constraints"):
        packet.explicit_constraints["routes"] = profile.route_labels()
    return profile, mission, report


__all__ = [
    "RENDER_CONFIDENCE_MIN",
    "ROUTE_TOPOLOGY_KEY",
    "RouteTopologyEdge",
    "RouteTopologyReport",
    "apply_route_topology_validation",
    "classify_endpoint",
    "validate_route_edge",
    "validate_route_topology",
]
