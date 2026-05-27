"""
Geodesic distance policy — corridor classification only; cannot authorize nonstop feasibility.

Catalog distances remain the only source for hard nonstop feasibility and PRIMARY verdicts.
"""

from __future__ import annotations

import re
from typing import List

from services.mission.route_distance_authority import RouteDistanceResolution

GEODESIC_MAX_CONFIDENCE = 0.62
GEODESIC_INTERNATIONAL_EXTRA_RESERVE_NM = 120.0
GEODESIC_BLOCKS_PRIMARY_VERDICT = True

_INTERNATIONAL_ENDPOINT_RE = re.compile(
    r"\b(?:london|paris|geneva|europe|dubai|tokyo|frankfurt|zurich|"
    r"hong\s+kong|singapore|sydney|mumbai|riyadh)\b",
    re.I,
)


def is_international_leg(route_label: str, *, origin_ref: str = "", dest_ref: str = "") -> bool:
    blob = f"{route_label} {origin_ref} {dest_ref}".lower()
    us_refs = ("kteb", "kjfk", "klax", "ksfo", "kmia", "kdfw", "kbos", "kase", "ktex")
    has_us = any(r in blob for r in us_refs) or re.search(
        r"\b(?:new\s+york|nyc|los\s+angeles|san\s+francisco|miami|dallas|boston|aspen)\b",
        blob,
    )
    has_intl = bool(_INTERNATIONAL_ENDPOINT_RE.search(blob))
    if origin_ref and dest_ref:
        o_us = origin_ref.upper().startswith("K") and origin_ref[0] == "K"
        d_us = dest_ref.upper().startswith("K") and dest_ref[0] == "K"
        if o_us != d_us:
            return True
    return has_us and has_intl


def apply_geodesic_policy(resolution: RouteDistanceResolution) -> RouteDistanceResolution:
    """Return resolution with geodesic restrictions applied."""
    if resolution.source != "geodesic":
        return resolution

    intl = is_international_leg(
        resolution.route_label,
        origin_ref=resolution.origin_ref,
        dest_ref=resolution.dest_ref,
    )
    extra_reserve = GEODESIC_INTERNATIONAL_EXTRA_RESERVE_NM if intl else 0.0

    return RouteDistanceResolution(
        route_label=resolution.route_label,
        distance_nm=resolution.distance_nm,
        source=resolution.source,
        confidence=min(resolution.confidence, GEODESIC_MAX_CONFIDENCE),
        origin_ref=resolution.origin_ref,
        dest_ref=resolution.dest_ref,
        authorize_nonstop_feasibility=False,
        corridor_classification_only=True,
        international_leg=intl,
        extra_reserve_nm=extra_reserve,
    )


def mission_has_geodesic_only_routes(resolutions: List[RouteDistanceResolution]) -> bool:
    verified = [r for r in resolutions if r.distance_nm > 0]
    if not verified:
        return False
    return all(r.source == "geodesic" for r in verified)


def mission_forbids_primary_verdict(resolutions: List[RouteDistanceResolution]) -> bool:
    """True when any geodesic leg would forbid PRIMARY / BEST FIT style verdicts."""
    return any(
        r.source == "geodesic" and GEODESIC_BLOCKS_PRIMARY_VERDICT for r in resolutions
    )
