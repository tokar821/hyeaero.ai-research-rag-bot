"""
Geographic + route intelligence — regional ontology, industrial spokes, hub anchors.

Runs after initial extraction, BEFORE route topology validation.
Only hub → spoke and hub → continuation edges; topology validator remains authoritative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from services.mission.arctic_industrial_layer import (
    ARCTIC_INDUSTRIAL_LAYER,
    arctic_layer_active,
    infer_arctic_industrial_layer_routes,
    rebalance_arctic_hub_spokes,
)
from services.mission.hub_selection import select_local_hub
from services.mission.models import MissionProfile, Route
from services.mission.route_extractor import _build_route_extraction, resolve_place

GEOGRAPHIC_ROUTE_INTELLIGENCE_KEY = "geographic_route_intelligence"

# --- Regional corridor patterns → executive hub clusters (domestic utilization spokes only) ---
_NORTHEAST_CORRIDOR_RE = re.compile(
    r"\b(?:northeast\s+corridor|east\s+coast\s+corridor|domestic\s+northeast)\b",
    re.I,
)
_NORTHEAST_FLORIDA_RE = re.compile(
    r"\b(?:northeast\s+corridor|east\s+coast)\b.*\bflorida\b"
    r"|\bflorida\b.*\b(?:northeast|east\s+coast)\b",
    re.I,
)
_FLORIDA_CORRIDOR_RE = re.compile(
    r"\b(?:south\s+florida|palm\s+beach|miami)\b.*\b(?:corridor|domestic)\b"
    r"|\bflorida\b.*\b(?:corridor|domestic|trips?)\b",
    re.I,
)
_PERTH_MINING_RE = re.compile(
    r"\bperth\b|\baustralian\s+(?:extraction|mining)\b|\bmining\s+strips?\b.*\baustralia\b",
    re.I,
)
_TEXAS_ENERGY_RE = re.compile(
    r"\btexas\b.*\b(?:energy|desert|drilling|oil)\b|\b(?:houston|dallas)\b.*\bdesert\b",
    re.I,
)
_CALIFORNIA_NEVADA_RE = re.compile(
    r"\b(?:california\s+and\s+nevada|nevada\s+and\s+california|short\s+california|"
    r"california\s+(?:and\s+)?nevada\s+trips?)\b",
    re.I,
)

# --- Mountain geography ---
_COLORADO_UTAH_MOUNTAIN_RE = re.compile(
    r"\b(?:colorado|utah)\b.*\b(?:mountain|ski|aspen|vail|telluride|jackson)\b"
    r"|\b(?:mountain\s+airports?)\b.*\b(?:colorado|utah)\b"
    r"|\b(?:aspen|vail|telluride)\b.*\b(?:colorado|utah|winter)\b",
    re.I,
)
_MOUNTAIN_REGION_RE = re.compile(
    r"\b(?:ski\s+operations?|winter\s+operations?|ski\s+regions?)\b.*\b(?:aspen|vail|jackson|telluride|banff)\b"
    r"|\b(?:aspen|vail|jackson\s+hole|telluride|sun\s+valley|banff)\b",
    re.I,
)

# --- Caribbean operational dominance ---
_CARIBBEAN_OPS_DOMINANT_RE = re.compile(
    r"\b(?:caribbean\s+hops?|short\s+caribbean|caribbean\s+islands?|caribbean\s+rotations?|"
    r"humid.*short[- ]runway|island\s+operations?|tropical\s+islands?)\b",
    re.I,
)
_MULTI_DOMAIN_PORTFOLIO_RE = re.compile(r"\boperate\s+across\b", re.I)
_EU_HOME_BASE_RE = re.compile(
    r"\b(?:based\s+in|headquartered\s+in|home\s+(?:base|office)\s+in)\s+"
    r"(?:madrid|paris|london|barcelona)\b",
    re.I,
)

# --- Founder / principal continuation ---
_FOUNDER_CONTINUATION_RE = re.compile(
    r"\b(?:founder|principal|chairman|ceo)\b.*\b(?:requires?|insists?|needs?)\b.*"
    r"\b(?:nonstop|capability)\b"
    r"|\b(?:founder|principal)\b.*\b(?:singapore|tokyo|doha|dubai|hong\s+kong)\b",
    re.I,
)
_DOMESTIC_UTILIZATION_RE = re.compile(
    r"\b(?:\d{1,3}\s*%\s+of\s+(?:our\s+)?flying|mostly\s+flies?|domestic\s+corridor|"
    r"short\s+(?:east\s+coast|california|trips?))\b",
    re.I,
)

# --- Industrial geography (operational nodes, not literal airports) ---
_INDUSTRIAL_GEO_SPECS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    (
        r"\b(?:remote\s+)?(?:middle\s+eastern\s+)?desert\s+strips?\b|"
        r"\bdesert\s+(?:energy|operations?)\b",
        "Desert Energy Corridor",
        ("Dubai", "Abu Dhabi", "Riyadh"),
    ),
    (
        r"\barctic\s+(?:mining|industrial)\b|"
        r"\b(?:mining\s+strips?|northern\s+canada)\b.*\b(?:winter|arctic)\b"
        r"|\barctic\s+mining\b",
        "Arctic Industrial Access",
        ("Calgary", "Anchorage", "Toronto"),
    ),
    (
        r"\btexas\b.*\b(?:energy|desert|drilling|oil)\b|\b(?:houston|dallas)\b.*\bdesert\b",
        "Desert Energy Corridor",
        ("Houston", "Dallas"),
    ),
    (
        r"\b(?:gravel\s+strips?|oil[- ]field\s+transfers?|drilling\s+sites?)\b",
        "Remote Drilling Sites",
        ("Houston", "Calgary", "Dubai"),
    ),
    (
        r"\bperth\b|\baustralian\s+(?:extraction|mining)\b|\bpilbara\b",
        "Australian Extraction Strips",
        ("Perth", "Singapore"),
    ),
    (
        r"\bpermian\s+basin\b|\bpermian\b",
        "Permian Basin",
        ("Houston", "Dallas"),
    ),
    (
        r"\bnigeria\b|\bnigerian\s+energy\b|\b(?:texas|nigeria)\b.*\bdrilling\b",
        "Nigerian Energy Corridor",
        ("Houston", "Lagos"),
    ),
    (
        r"\bwest\s+african\b|\bwest\s+africa\b",
        "West Africa",
        ("Frankfurt", "Houston", "Lagos"),
    ),
    (
        r"\bnorthern\s+africa\b|\bnorth\s+africa\b",
        "Northern Africa",
        ("Houston", "Dallas"),
    ),
    (
        r"\boffshore\s+(?:rigs?|platforms?|oil)\b",
        "Offshore Rigs",
        ("Houston", "Dallas"),
    ),
    (
        r"\bremote\s+(?:resource|mining)\b|"
        r"\b(?:mining|oil)\s+strips?\b",
        "Remote Drilling Sites",
        ("Houston", "Calgary"),
    ),
)

_CONTINUATION_DEST_RE = re.compile(
    r"\b(?:singapore|tokyo|hong\s+kong|seoul|doha|dubai|abu\s+dhabi|riyadh)\b",
    re.I,
)

_EUROPEAN_HUB_CITIES = frozenset(
    {
        "madrid",
        "paris",
        "london",
        "geneva",
        "zurich",
        "frankfurt",
        "barcelona",
        "rome",
        "milan",
    }
)

# Executive EU overlay — preserve hub→EU legs when industrial context dominates extraction
_EXECUTIVE_EU_OVERLAY_RE = re.compile(
    r"\b(?:executives?|leadership|ownership|principal|chairman|ceo)\b.*"
    r"\b(?:fly|travel|require|visit|nonstop)\b"
    r"|\b(?:fly|travel)\s+(?:to|between)\b.*\b(?:paris|geneva|frankfurt|zurich|london)\b"
    r"|\b(?:paris|geneva|frankfurt|zurich|london)\b.*\b(?:executives?|leadership)\b"
    r"|\bparis\s*[-–]\s*geneva\b",
    re.I,
)
_EU_EXEC_CITY_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("Paris", r"\bparis\b"),
    ("Geneva", r"\bgeneva\b"),
    ("Frankfurt", r"\bfrankfurt\b"),
    ("Zurich", r"\bzurich\b"),
    ("London", r"\blondon\b"),
)
_FIELD_REGION_NAMES = frozenset(
    {
        "remote drilling sites",
        "desert energy corridor",
        "arctic industrial access",
        "arctic oil platforms",
        "west africa",
        "remote gravel strips",
        "northern alberta oil fields",
        "nunavut field ops",
        "australian extraction strips",
        "permian basin",
        "nigerian energy corridor",
        "northern africa",
        "offshore rigs",
        "pilbara",
    }
)
# EU executive hub → industrial field spokes are valid when mining/industrial support is stated
_EU_HUB_FIELD_SPOKE_DESTINATIONS = frozenset(
    {
        "west africa",
        "nigerian energy corridor",
        "northern africa",
        "permian basin",
        "remote drilling sites",
        "desert energy corridor",
        "offshore rigs",
        "australian extraction strips",
        "pilbara",
        "remote gravel strips",
        "northern alberta oil fields",
        "nunavut field ops",
        "arctic industrial access",
    }
)


@dataclass
class GeographicEnrichmentReport:
    routes_added: List[str] = field(default_factory=list)
    routes_rebalanced: List[str] = field(default_factory=list)
    regions_activated: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "routes_added": list(self.routes_added),
            "routes_rebalanced": list(self.routes_rebalanced),
            "regions_activated": list(self.regions_activated),
        }


def _existing_pairs(profile: MissionProfile) -> Set[Tuple[str, str]]:
    return {(r.origin.lower(), r.destination.lower()) for r in profile.routes}


def _add_route(
    profile: MissionProfile,
    origin: str,
    destination: str,
    *,
    existing: Set[Tuple[str, str]],
    boost: float = 0.08,
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


def _pick_hub_from_text_or_routes(
    profile: MissionProfile,
    text: str,
    priority: Tuple[str, ...],
    *,
    mission_type: str = "executive",
) -> str:
    default = priority[0] if priority else "New York"
    return select_local_hub(
        profile,
        text,
        priority,
        mission_type=mission_type,
        default=default,
    )


def _infer_northeast_corridor_routes(
    text: str,
    profile: MissionProfile,
    existing: Set[Tuple[str, str]],
) -> List[str]:
    if not _NORTHEAST_CORRIDOR_RE.search(text or ""):
        return []
    added: List[str] = []
    hubs = ("New York", "Boston", "Washington")
    pairs = (
        ("New York", "Boston"),
        ("New York", "Washington"),
        ("Boston", "New York"),
    )
    for o, d in pairs:
        r = _add_route(profile, o, d, existing=existing, boost=0.09)
        if r:
            added.append(r.label())
    return added


def _infer_northeast_florida_corridor(
    text: str,
    profile: MissionProfile,
    existing: Set[Tuple[str, str]],
) -> List[str]:
    tl = text or ""
    if not (
        _NORTHEAST_FLORIDA_RE.search(tl)
        or (
            _NORTHEAST_CORRIDOR_RE.search(tl)
            and re.search(r"\bflorida\b", tl, re.I)
        )
        or _FLORIDA_CORRIDOR_RE.search(tl)
    ):
        return []
    added: List[str] = []
    ne_hub = _pick_hub_from_text_or_routes(
        profile, text, ("New York", "Boston", "Washington", "Teterboro")
    )
    fl_hub = _pick_hub_from_text_or_routes(
        profile, text, ("Miami", "Palm Beach", "Fort Lauderdale")
    )
    for o, d in (
        (ne_hub, "Miami"),
        (ne_hub, "Palm Beach"),
        (fl_hub, "Caribbean"),
    ):
        if o.lower() == d.lower():
            continue
        r = _add_route(profile, o, d, existing=existing, boost=0.09)
        if r:
            added.append(r.label())
    return added


def _infer_california_nevada_cluster(
    text: str,
    profile: MissionProfile,
    existing: Set[Tuple[str, str]],
) -> List[str]:
    if not _CALIFORNIA_NEVADA_RE.search(text or "") and not re.search(
        r"\b(?:california|nevada)\b.*\b(?:short|trips?|domestic)\b", text or "", re.I
    ):
        return []
    added: List[str] = []
    hub = _pick_hub_from_text_or_routes(
        profile, text, ("Los Angeles", "San Francisco", "Las Vegas")
    )
    for dest in ("San Francisco", "Las Vegas", "San Diego"):
        if dest.lower() == hub.lower():
            continue
        r = _add_route(profile, hub, dest, existing=existing, boost=0.08)
        if r:
            added.append(r.label())
    return added


def _infer_mountain_overlay_routes(
    text: str,
    profile: MissionProfile,
    existing: Set[Tuple[str, str]],
) -> List[str]:
    if not (
        _COLORADO_UTAH_MOUNTAIN_RE.search(text or "")
        or _MOUNTAIN_REGION_RE.search(text or "")
        or re.search(r"\b(?:colorado|utah)\b.*\bmountain\b", text or "", re.I)
    ):
        return []
    added: List[str] = []
    tl = text or ""
    la_in_mission = bool(
        re.search(r"\b(?:los\s+angeles|\bla\b)\b", tl, re.I)
        or any(
            r.origin.lower() in ("los angeles", "la") or r.destination.lower() == "los angeles"
            for r in profile.routes
        )
    )
    if la_in_mission:
        hub = _pick_hub_from_text_or_routes(
            profile, text, ("Los Angeles", "Denver", "Salt Lake City")
        )
    elif re.search(r"\bcolorado\b", tl, re.I):
        hub = _pick_hub_from_text_or_routes(profile, text, ("Denver", "Los Angeles"))
    elif re.search(r"\butah\b", tl, re.I):
        hub = _pick_hub_from_text_or_routes(profile, text, ("Salt Lake City", "Denver"))
    else:
        hub = _pick_hub_from_text_or_routes(
            profile,
            text,
            ("Denver", "Los Angeles", "San Francisco", "Salt Lake City"),
        )
    spokes = (
        "Aspen",
        "Telluride",
        "Jackson Hole",
        "Vail",
        "Eagle",
        "Sun Valley",
    )
    if re.search(r"\bbanff\b", tl, re.I):
        spokes = spokes + ("Banff",)
    for dest in spokes:
        if dest.lower() == hub.lower():
            continue
        r = _add_route(profile, hub, dest, existing=existing, boost=0.09)
        if r:
            added.append(r.label())
    return added


def _pick_industrial_hub(
    profile: MissionProfile,
    text: str,
    hub_priority: Tuple[str, ...],
    operational_node: str,
) -> str:
    tl = (text or "").lower()
    if _TEXAS_ENERGY_RE.search(tl) and operational_node in (
        "Desert Energy Corridor",
        "Remote Drilling Sites",
    ):
        return select_local_hub(
            profile,
            text,
            ("Houston", "Dallas"),
            mission_type="industrial",
            default="Houston",
        )
    if operational_node == "Australian Extraction Strips":
        return select_local_hub(
            profile,
            text,
            ("Perth", "Singapore"),
            mission_type="industrial",
            regional_bias=("Perth",),
            default="Perth",
        )
    if operational_node == "Desert Energy Corridor" and re.search(
        r"\bmiddle\s+eastern?\b", tl
    ) and not re.search(r"\btexas\b|\bhouston\b|\bdallas\b", tl, re.I):
        for hub in hub_priority:
            place, conf = resolve_place(hub)
            if place and conf >= 0.72 and place.country in ("UAE", "Saudi Arabia", "Qatar"):
                return hub
        return "Dubai"
    return _pick_hub_from_text_or_routes(
        profile, text, hub_priority, mission_type="industrial"
    )


def _infer_industrial_geography_spokes(
    text: str,
    profile: MissionProfile,
    existing: Set[Tuple[str, str]],
) -> List[str]:
    added: List[str] = []
    tl = text or ""
    seen_nodes: Set[str] = set()
    texas_energy = bool(_TEXAS_ENERGY_RE.search(tl))
    arctic_active = arctic_layer_active(tl)
    for pat, node, hub_priority in _INDUSTRIAL_GEO_SPECS:
        if not re.search(pat, tl, re.I):
            continue
        if node in seen_nodes:
            continue
        # Northern Canada gravel / arctic logistics handled by arctic_industrial_layer
        if arctic_active and node == "Remote Drilling Sites" and re.search(
            r"\b(?:gravel\s+strips?|northern\s+canada|nunavut)\b", tl, re.I
        ):
            continue
        if arctic_active and node == "Arctic Industrial Access" and re.search(
            r"\b(?:northern\s+canada|yellowknife|nunavut|northern\s+alberta)\b", tl, re.I
        ):
            continue
        if texas_energy and node == "Desert Energy Corridor" and not re.search(r"texas", pat, re.I):
            continue
        seen_nodes.add(node)
        hub = _pick_industrial_hub(profile, tl, hub_priority, node)
        r = _add_route(profile, hub, node, existing=existing, boost=0.1)
        if r:
            added.append(r.label())
    return added


def _strip_field_to_executive_ghost_edges(profile: MissionProfile) -> List[str]:
    """Remove field↔EU executive ghosts; hub→EU overlay replaces these."""
    removed: List[str] = []
    kept: List[Route] = []
    for r in profile.routes:
        o_l, d_l = r.origin.lower(), r.destination.lower()
        if o_l in _FIELD_REGION_NAMES and d_l in _EUROPEAN_HUB_CITIES:
            removed.append(r.label())
            continue
        if d_l in _FIELD_REGION_NAMES and o_l in _EUROPEAN_HUB_CITIES:
            if d_l in _EU_HUB_FIELD_SPOKE_DESTINATIONS:
                kept.append(r)
                continue
            removed.append(r.label())
            continue
        kept.append(r)
    profile.routes = kept
    return removed


def _pick_executive_overlay_hub(profile: MissionProfile, text: str) -> str:
    tl = text or ""
    if re.search(r"\b(?:perth|pilbara|australian)\b", tl, re.I):
        return select_local_hub(
            profile, tl, ("Perth",), mission_type="executive", regional_bias=("Perth",), default="Perth"
        )
    if _TEXAS_ENERGY_RE.search(tl) or re.search(r"\bpermian\b", tl, re.I):
        return select_local_hub(
            profile, tl, ("Houston", "Dallas"), mission_type="executive", default="Houston"
        )
    if re.search(r"\b(?:houston|lagos|west\s+africa|nigeria)\b", tl, re.I):
        return select_local_hub(
            profile, tl, ("Houston", "Lagos"), mission_type="executive", default="Houston"
        )
    return _pick_hub_from_text_or_routes(
        profile,
        tl,
        ("Houston", "New York", "London", "Paris", "Calgary", "Miami"),
        mission_type="executive",
    )


def _infer_executive_eu_overlay_routes(
    text: str,
    profile: MissionProfile,
    existing: Set[Tuple[str, str]],
) -> List[str]:
    """
    When industrial/field language coexists with EU executive cities, add hub→EU legs
    from the mission-origin hub — never field-region→EU stitching.
    """
    tl = text or ""
    has_eu = any(re.search(pat, tl, re.I) for _, pat in _EU_EXEC_CITY_PATTERNS)
    if not has_eu:
        return []
    industrial_context = bool(
        _TEXAS_ENERGY_RE.search(tl)
        or any(re.search(spec[0], tl, re.I) for spec in _INDUSTRIAL_GEO_SPECS)
        or re.search(
            r"\b(?:drilling|desert|mining|industrial|field\s+access|gravel)\b", tl, re.I
        )
    )
    if not industrial_context and not _EXECUTIVE_EU_OVERLAY_RE.search(tl):
        return []

    _strip_field_to_executive_ghost_edges(profile)
    added: List[str] = []
    hub = _pick_executive_overlay_hub(profile, tl)
    eu_cities: List[str] = []
    for city, pat in _EU_EXEC_CITY_PATTERNS:
        if re.search(pat, tl, re.I):
            eu_cities.append(city)

    for dest in eu_cities:
        r = _add_route(profile, hub, dest, existing=existing, boost=0.1)
        if r:
            added.append(r.label())

    if "Paris" in eu_cities and "Geneva" in eu_cities and re.search(
        r"\bparis\s*[-–]\s*geneva\b|\b(?:through|via)\s+paris\b.*\bgeneva\b", tl, re.I
    ):
        r = _add_route(profile, "Paris", "Geneva", existing=existing, boost=0.09)
        if r:
            added.append(r.label())

    return added


def _pick_founder_continuation_hub(text: str, profile: MissionProfile) -> str:
    tl = text or ""
    if _NORTHEAST_CORRIDOR_RE.search(tl) or re.search(
        r"\b(?:northeast|east\s+coast|boston|philadelphia)\b", tl, re.I
    ):
        return _pick_hub_from_text_or_routes(profile, tl, ("New York", "Boston", "Washington"))
    if re.search(r"\b(?:california|los\s+angeles|san\s+francisco)\b", tl, re.I):
        return _pick_hub_from_text_or_routes(profile, tl, ("Los Angeles", "San Francisco"))
    return _pick_hub_from_text_or_routes(profile, tl, ("New York", "Los Angeles", "Miami"))


def _infer_founder_continuation_routes(
    text: str,
    profile: MissionProfile,
    existing: Set[Tuple[str, str]],
) -> List[str]:
    if not _FOUNDER_CONTINUATION_RE.search(text or ""):
        return []
    if not _DOMESTIC_UTILIZATION_RE.search(text or "") and not _NORTHEAST_CORRIDOR_RE.search(
        text or ""
    ):
        # Still allow if explicit founder + destination
        if not re.search(r"\bfounder\b", text or "", re.I):
            return []
    added: List[str] = []
    hub = _pick_founder_continuation_hub(text, profile)
    tl = (text or "").lower()
    dests: List[str] = []
    for city, pat in (
        ("Singapore", r"\bsingapore\b"),
        ("Tokyo", r"\btokyo\b"),
        ("Doha", r"\bdoha\b"),
        ("Dubai", r"\bdubai\b"),
        ("Hong Kong", r"\bhong\s+kong\b"),
        ("Abu Dhabi", r"\babu\s+dhabi\b"),
    ):
        if re.search(pat, tl):
            dests.append(city)
    for dest in dests:
        r = _add_route(profile, hub, dest, existing=existing, boost=0.11)
        if r:
            added.append(r.label())
    return added


def _literal_edge_in_text(origin: str, destination: str, text: str) -> bool:
    tl = (text or "").lower()
    o, d = origin.lower(), destination.lower()
    return bool(
        re.search(rf"\b{re.escape(o)}\s*(?:->|to|-)\s*{re.escape(d)}\b", tl)
        or re.search(rf"\bfrom\s+{re.escape(o)}\s+to\s+{re.escape(d)}\b", tl)
    )


def rebalance_caribbean_hub_routes(
    query: str,
    profile: MissionProfile,
) -> List[str]:
    """
    Caribbean ops default to Miami / South Florida — not European cities unless stated as home base.
    """
    if not _CARIBBEAN_OPS_DOMINANT_RE.search(query or ""):
        return []
    if _EU_HOME_BASE_RE.search(query or ""):
        return []

    rebalanced: List[str] = []
    kept: List[Route] = []
    existing: Set[Tuple[str, str]] = set()

    for r in list(profile.routes):
        o_l, d_l = r.origin.lower(), r.destination.lower()
        if d_l == "caribbean" and o_l in _EUROPEAN_HUB_CITIES:
            if _literal_edge_in_text(r.origin, r.destination, query):
                kept.append(r)
                existing.add((o_l, d_l))
                continue
            # Replace with Miami hub spoke
            continue
        kept.append(r)
        existing.add((o_l, d_l))

    profile.routes = kept
    hub = _pick_hub_from_text_or_routes(
        profile, query, ("Miami", "Palm Beach", "Fort Lauderdale")
    )
    if hub.lower() not in ("miami", "palm beach"):
        hub = "Miami"
    if ("miami", "caribbean") not in existing and (hub.lower(), "caribbean") not in existing:
        ext = _build_route_extraction(hub, "Caribbean", pattern_boost=0.1)
        if ext:
            profile.routes.append(ext.route)
            rebalanced.append(ext.route.label())

    # Re-add Europe continuation legs without making Europe the Caribbean anchor
    for r in list(profile.routes):
        if r.destination.lower() in ("dubai", "abu dhabi", "riyadh", "doha") and (
            r.origin.lower() in _EUROPEAN_HUB_CITIES or r.origin.lower() == "miami"
        ):
            existing.add((r.origin.lower(), r.destination.lower()))

    return rebalanced


def _infer_multi_domain_portfolio_routes(
    text: str,
    profile: MissionProfile,
    existing: Set[Tuple[str, str]],
) -> List[str]:
    """
    Decompose ``operate across …`` portfolios into domain-valid hub spokes
    without star-hub corridor stitching.
    """
    if not _MULTI_DOMAIN_PORTFOLIO_RE.search(text or ""):
        return []
    tl = text or ""
    added: List[str] = []

    if re.search(r"\bhouston\b", tl, re.I) and re.search(r"\blondon\b", tl, re.I):
        r = _add_route(profile, "Houston", "London", existing=existing, boost=0.1)
        if r:
            added.append(r.label())

    if re.search(r"\bmiami\b", tl, re.I) and re.search(r"\bcaribbean\b", tl, re.I):
        r = _add_route(profile, "Miami", "Caribbean", existing=existing, boost=0.1)
        if r:
            added.append(r.label())

    if re.search(r"\bsingapore\b", tl, re.I):
        hub = "Houston"
        if re.search(r"\bhouston\b", tl, re.I):
            hub = "Houston"
        elif re.search(r"\bmiami\b", tl, re.I):
            hub = "Miami"
        r = _add_route(profile, hub, "Singapore", existing=existing, boost=0.09)
        if r:
            added.append(r.label())

    return added


def apply_geographic_route_intelligence(
    query: str,
    profile: MissionProfile,
    *,
    data_used: Optional[Dict[str, Any]] = None,
) -> GeographicEnrichmentReport:
    """
    Enrich route graph from regional / industrial / founder language before topology validation.
    """
    report = GeographicEnrichmentReport()
    existing = _existing_pairs(profile)

    for fn, region_name in (
        (_infer_multi_domain_portfolio_routes, "multi_domain_portfolio"),
        (_infer_northeast_corridor_routes, "northeast_corridor"),
        (_infer_northeast_florida_corridor, "northeast_florida"),
        (_infer_california_nevada_cluster, "california_nevada"),
        (_infer_mountain_overlay_routes, "mountain_overlay"),
        (infer_arctic_industrial_layer_routes, ARCTIC_INDUSTRIAL_LAYER),
        (_infer_industrial_geography_spokes, "industrial_geography"),
        (_infer_executive_eu_overlay_routes, "executive_eu_overlay"),
        (_infer_founder_continuation_routes, "founder_continuation"),
    ):
        added = fn(query, profile, existing)
        if added:
            report.routes_added.extend(added)
        if added or (
            region_name == ARCTIC_INDUSTRIAL_LAYER and arctic_layer_active(query)
        ):
            report.regions_activated.append(region_name)

    rebalanced = rebalance_caribbean_hub_routes(query, profile)
    if rebalanced:
        report.routes_rebalanced.extend(rebalanced)
        report.regions_activated.append("caribbean_hub_rebalance")

    if arctic_layer_active(query):
        arctic_rebalanced = rebalance_arctic_hub_spokes(profile)
        if arctic_rebalanced:
            report.routes_rebalanced.extend(arctic_rebalanced)

    if isinstance(data_used, dict):
        data_used[GEOGRAPHIC_ROUTE_INTELLIGENCE_KEY] = report.to_dict()

    return report


__all__ = [
    "GEOGRAPHIC_ROUTE_INTELLIGENCE_KEY",
    "GeographicEnrichmentReport",
    "apply_geographic_route_intelligence",
    "rebalance_caribbean_hub_routes",
]
