"""
Route distance authority — single source for stage length (nm).

Sources (in priority order):
  1. verified catalog
  2. geodesic (resolved airport/city coordinates only)
  3. unresolved — never invent international stage lengths

Unknown routes must not produce ranked recommendations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from services.mission.route_distance_catalog import VERIFIED_ROUTE_DISTANCE_NM

# City phrase → ICAO for geodesic resolution (extends rag geo where needed).
_CITY_ICAO: Dict[str, str] = {
    "teterboro": "KTEB",
    "teb": "KTEB",
    "new york": "KTEB",
    "nyc": "KTEB",
    "jfk": "KJFK",
    "london": "EGLL",
    "los angeles": "KLAX",
    "la": "KLAX",
    "san francisco": "KSFO",
    "sfo": "KSFO",
    "miami": "KMIA",
    "boston": "KBOS",
    "dallas": "KDFW",
    "aspen": "KASE",
    "telluride": "KTEX",
    "dubai": "OMDB",
    "tokyo": "RJTT",
    "paris": "LFPG",
    "geneva": "LFPG",
    "zurich": "LFPG",
    "frankfurt": "LFPG",
    "chicago": "KORD",
    "denver": "KDEN",
    "seattle": "KSEA",
    "atlanta": "KATL",
    "palm beach": "KPBI",
    "caribbean": "TNCM",
    "nassau": "MYNN",
}

_EXTRA_COORDS: Dict[str, Tuple[float, float]] = {
    "KASE": (39.2232, -106.8689),
    "KTEX": (37.9538, -107.9078),
    "RJTT": (35.5523, 139.7798),
    "KPBI": (26.6832, -80.0956),
    "MYNN": (25.0390, -77.4662),
}

# Below this confidence, ranked recommendations are blocked.
ROUTE_CONFIDENCE_RANK_THRESHOLD = 0.65


@dataclass(frozen=True)
class RouteDistanceResolution:
    route_label: str
    distance_nm: float
    source: str  # catalog | geodesic | unresolved
    confidence: float
    origin_ref: str = ""
    dest_ref: str = ""
    authorize_nonstop_feasibility: bool = True
    corridor_classification_only: bool = False
    international_leg: bool = False
    extra_reserve_nm: float = 0.0

    @property
    def is_catalog_verified(self) -> bool:
        return self.source == "catalog" and self.distance_nm > 0

    @property
    def is_verified(self) -> bool:
        """Distance known — catalog or geodesic (corridor-only for geodesic)."""
        return self.source in ("catalog", "geodesic") and self.distance_nm > 0

    @property
    def blocks_ranking(self) -> bool:
        if self.source == "unresolved" or self.distance_nm <= 0:
            return True
        if self.source == "geodesic":
            return self.confidence < ROUTE_CONFIDENCE_RANK_THRESHOLD
        return self.confidence < ROUTE_CONFIDENCE_RANK_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_label": self.route_label,
            "distance_nm": round(self.distance_nm, 1),
            "source": self.source,
            "confidence": round(self.confidence, 3),
            "origin_ref": self.origin_ref,
            "dest_ref": self.dest_ref,
            "authorize_nonstop_feasibility": self.authorize_nonstop_feasibility,
            "corridor_classification_only": self.corridor_classification_only,
            "international_leg": self.international_leg,
            "extra_reserve_nm": round(self.extra_reserve_nm, 1),
            "blocks_ranking": self.blocks_ranking,
        }


def normalize_route_key(label: str) -> str:
    s = (label or "").strip().lower()
    s = s.replace("→", "->")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\bto\b", "->", s)
    return s


def _parse_endpoints(label: str) -> Tuple[str, str]:
    key = normalize_route_key(label)
    for sep in ("->", "→"):
        if sep in key:
            left, right = key.split(sep, 1)
            return left.strip(), right.strip()
    return "", ""


def _resolve_place_icao(place: str) -> Optional[str]:
    p = (place or "").strip().lower()
    if not p:
        return None
    if p.upper() in _EXTRA_COORDS or len(p) == 4 and p.isalpha():
        return p.upper()
    # Longest phrase first — prevents "dallas" matching "la" -> KLAX.
    for phrase, icao in sorted(_CITY_ICAO.items(), key=lambda x: len(x[0]), reverse=True):
        if phrase == p:
            return icao
        if len(phrase) >= 4 and phrase in p:
            return icao
        if p.endswith(phrase) and len(phrase) >= 3:
            return icao
    return None


def _coords(icao: str) -> Optional[Tuple[float, float]]:
    from rag.aviation_engines.geo import ICAO_COORDS, nm_between

    code = icao.upper()
    latlon = ICAO_COORDS.get(code) or _EXTRA_COORDS.get(code)
    if not latlon:
        return None
    return latlon


def _geodesic_nm(origin: str, dest: str) -> Optional[Tuple[float, str, str]]:
    from rag.aviation_engines.geo import nm_between

    o_icao = _resolve_place_icao(origin)
    d_icao = _resolve_place_icao(dest)
    if not o_icao or not d_icao or o_icao == d_icao:
        return None
    o_c = _coords(o_icao)
    d_c = _coords(d_icao)
    if not o_c or not d_c:
        return None
    return nm_between(o_c, d_c), o_icao, d_icao


def resolve_route_distance(route_label: str) -> RouteDistanceResolution:
    """Authoritative stage length for one route label."""
    label = (route_label or "").strip()
    if not label:
        return RouteDistanceResolution(
            route_label="",
            distance_nm=0.0,
            source="unresolved",
            confidence=0.0,
        )

    key = normalize_route_key(label)
    if key in VERIFIED_ROUTE_DISTANCE_NM:
        res = RouteDistanceResolution(
            route_label=label,
            distance_nm=float(VERIFIED_ROUTE_DISTANCE_NM[key]),
            source="catalog",
            confidence=0.95,
            authorize_nonstop_feasibility=True,
            corridor_classification_only=False,
        )
        from services.mission.geodesic_policy import is_international_leg

        if is_international_leg(label):
            return RouteDistanceResolution(
                route_label=res.route_label,
                distance_nm=res.distance_nm,
                source=res.source,
                confidence=res.confidence,
                authorize_nonstop_feasibility=res.authorize_nonstop_feasibility,
                corridor_classification_only=res.corridor_classification_only,
                international_leg=True,
                extra_reserve_nm=0.0,
            )
        return res

    origin, dest = _parse_endpoints(label)
    if origin and dest:
        geo = _geodesic_nm(origin, dest)
        if geo is not None:
            nm, o_ref, d_ref = geo
            from services.mission.geodesic_policy import apply_geodesic_policy

            raw = RouteDistanceResolution(
                route_label=label,
                distance_nm=float(nm),
                source="geodesic",
                confidence=0.88,
                origin_ref=o_ref,
                dest_ref=d_ref,
            )
            return apply_geodesic_policy(raw)

    return RouteDistanceResolution(
        route_label=label,
        distance_nm=0.0,
        source="unresolved",
        confidence=0.0,
        authorize_nonstop_feasibility=False,
        corridor_classification_only=False,
    )


def resolve_mission_route_authority(route_labels: List[str]) -> List[RouteDistanceResolution]:
    return [resolve_route_distance(r) for r in route_labels if (r or "").strip()]


def peak_verified_stage_nm(resolutions: List[RouteDistanceResolution]) -> float:
    """Peak stage length from any verified source (catalog or geodesic corridor)."""
    verified = [r.distance_nm for r in resolutions if r.is_verified]
    return max(verified) if verified else 0.0


def peak_catalog_stage_nm(resolutions: List[RouteDistanceResolution]) -> float:
    """Peak stage length from catalog only — use for nonstop feasibility authority."""
    catalog = [r.distance_nm for r in resolutions if r.is_catalog_verified]
    return max(catalog) if catalog else 0.0


def total_extra_reserve_nm(resolutions: List[RouteDistanceResolution]) -> float:
    return sum(r.extra_reserve_nm for r in resolutions if r.is_verified)


def mission_route_blocks_ranking(route_labels: List[str]) -> Tuple[bool, List[RouteDistanceResolution]]:
    """
    Block ranked shortlist only when no leg is verified enough to anchor corridor classification.

    A catalog-verified peak stage (e.g. SFO-Tokyo) allows ULR ranking even when secondary legs
    are geodesic corridor-classified only.
    """
    resolutions = resolve_mission_route_authority(route_labels)
    if not route_labels:
        return False, resolutions
    if not resolutions:
        return True, resolutions

    peak_catalog = peak_catalog_stage_nm(resolutions)
    peak_verified = peak_verified_stage_nm(resolutions)
    any_unresolved = any(r.source == "unresolved" or r.distance_nm <= 0 for r in resolutions)

    if any_unresolved and peak_verified <= 0:
        return True, resolutions

    # Catalog anchor — ranking permitted; geodesic legs stay corridor-classified.
    if peak_catalog >= 4000:
        return False, resolutions

    if peak_verified >= 4000:
        return False, resolutions

    blocks = any(r.blocks_ranking for r in resolutions)
    return blocks, resolutions

