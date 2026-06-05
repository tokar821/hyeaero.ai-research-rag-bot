"""
Route distance authority — single source for stage length (nm).

Hybrid resolution (deterministic only for ranking / feasibility):

  1. geodesic — great-circle from airport reference coordinates
  2. operational overrides — small curated set where planning ≠ great-circle
  3. unresolved — never invent distance (no LLM/Tavily in this path)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from services.mission.route_distance_catalog import OPERATIONAL_ROUTE_OVERRIDES_NM

# Canonical place name (lowercase) → primary ICAO for distance.
_PLACE_CANONICAL_TO_ICAO: Dict[str, str] = {
    "new york": "KTEB",
    "teterboro": "KTEB",
    "los angeles": "KLAX",
    "san francisco": "KSFO",
    "miami": "KMIA",
    "boston": "KBOS",
    "dallas": "KDFW",
    "aspen": "KASE",
    "telluride": "KTEX",
    "chicago": "KORD",
    "scottsdale": "KSDL",
    "denver": "KDEN",
    "houston": "KIAH",
    "seattle": "KSEA",
    "atlanta": "KATL",
    "palm beach": "KPBI",
    "london": "EGLL",
    "paris": "LFPG",
    "geneva": "GVA",
    "zurich": "LSZH",
    "frankfurt": "EDDF",
    "lisbon": "LPPT",
    "reykjavik": "BIKF",
    "dubai": "OMDB",
    "abu dhabi": "OMAA",
    "tokyo": "RJTT",
    "nassau": "MYNN",
    "caribbean": "TNCM",
}

ROUTE_CONFIDENCE_RANK_THRESHOLD = 0.65


@dataclass(frozen=True)
class RouteDistanceResolution:
    route_label: str
    distance_nm: float
    source: str  # geodesic | operational_override | unresolved
    confidence: float
    origin_ref: str = ""
    dest_ref: str = ""
    authorize_nonstop_feasibility: bool = True
    corridor_classification_only: bool = False
    international_leg: bool = False
    extra_reserve_nm: float = 0.0

    @property
    def is_catalog_verified(self) -> bool:
        """Legacy name — operational override counts as catalog-class authority."""
        return self.source == "operational_override" and self.distance_nm > 0

    @property
    def is_verified(self) -> bool:
        return self.source in ("geodesic", "operational_override") and self.distance_nm > 0

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
    """Resolve endpoint to ICAO via aviation_places + geo aliases."""
    p = (place or "").strip()
    if not p:
        return None
    code = p.upper().replace(" ", "")
    if len(code) == 4 and code.isalpha():
        from rag.aviation_engines.geo import ICAO_COORDS

        if code in ICAO_COORDS:
            return code

    try:
        from services.mission.route_extractor import resolve_place

        av, conf = resolve_place(p)
        if av is not None and conf >= 0.45:
            canon = av.canonical.lower()
            if canon in _PLACE_CANONICAL_TO_ICAO:
                return _PLACE_CANONICAL_TO_ICAO[canon]
    except Exception:
        pass

    low = p.lower()
    if low in _PLACE_CANONICAL_TO_ICAO:
        return _PLACE_CANONICAL_TO_ICAO[low]

    from rag.aviation_engines.geo import _icao_for_city_phrase

    return _icao_for_city_phrase(low)


def _coords(icao: str) -> Optional[Tuple[float, float]]:
    from rag.aviation_engines.geo import ICAO_COORDS

    return ICAO_COORDS.get(icao.upper())


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


def _apply_international_policy(res: RouteDistanceResolution) -> RouteDistanceResolution:
    from services.mission.geodesic_policy import is_international_leg

    if not is_international_leg(res.route_label):
        return res
    return RouteDistanceResolution(
        route_label=res.route_label,
        distance_nm=res.distance_nm,
        source=res.source,
        confidence=res.confidence,
        origin_ref=res.origin_ref,
        dest_ref=res.dest_ref,
        authorize_nonstop_feasibility=res.authorize_nonstop_feasibility,
        corridor_classification_only=res.corridor_classification_only,
        international_leg=True,
        extra_reserve_nm=res.extra_reserve_nm,
    )


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
    origin, dest = _parse_endpoints(label)

    geo_res: Optional[RouteDistanceResolution] = None
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
            geo_res = apply_geodesic_policy(raw)

    if key in OPERATIONAL_ROUTE_OVERRIDES_NM:
        res = RouteDistanceResolution(
            route_label=label,
            distance_nm=float(OPERATIONAL_ROUTE_OVERRIDES_NM[key]),
            source="operational_override",
            confidence=0.95,
            authorize_nonstop_feasibility=True,
            corridor_classification_only=False,
        )
        return _apply_international_policy(res)

    if geo_res is not None:
        return geo_res

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
    verified = [r.distance_nm for r in resolutions if r.is_verified]
    return max(verified) if verified else 0.0


def peak_catalog_stage_nm(resolutions: List[RouteDistanceResolution]) -> float:
    catalog = [r.distance_nm for r in resolutions if r.is_catalog_verified]
    return max(catalog) if catalog else 0.0


def total_extra_reserve_nm(resolutions: List[RouteDistanceResolution]) -> float:
    return sum(r.extra_reserve_nm for r in resolutions if r.is_verified)


def mission_route_blocks_ranking(route_labels: List[str]) -> Tuple[bool, List[RouteDistanceResolution]]:
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

    if peak_catalog >= 4000:
        return False, resolutions

    if peak_verified >= 4000:
        return False, resolutions

    blocks = any(r.blocks_ranking for r in resolutions)
    return blocks, resolutions
