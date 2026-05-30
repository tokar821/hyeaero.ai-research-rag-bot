"""
Replacement hierarchy — credible acquisition alternatives by tier, not category collapse.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence

from services.catalog.catalog_alias_resolver import resolve_canonical_display_name
from services.recommendation.aircraft_positioning import (
    PositionTier,
    aircraft_position_tier,
    is_prestige_collapse,
    tier_distance,
)
from services.sanity.aircraft_class_guard import violates_class_sanity

_REPLACEMENT_RE = re.compile(
    r"\b(?:replace|replacing|instead\s+of|alternative\s+to|step\s+down\s+from|"
    r"lower\s+cost\s+than|cheaper\s+than|below\s+)(?:a|an|the|our)?\s*",
    re.I,
)

# Credible same-mission replacements by target tier (catalog names only).
_TIER_REPLACEMENT_POOL: dict[PositionTier, tuple[str, ...]] = {
    PositionTier.FLAGSHIP_ULR: (
        "Falcon 8X",
        "Global 6500",
        "Gulfstream G650ER",
        "Gulfstream G650",
    ),
    PositionTier.UPPER_LARGE: (
        "Falcon 8X",
        "Challenger 650",
        "Gulfstream G650ER",
        "Global 6500",
    ),
    PositionTier.LARGE_CABIN: (
        "Challenger 650",
        "Falcon 8X",
        "Gulfstream G280",
        "Praetor 600",
    ),
    PositionTier.SUPER_MID: (
        "Praetor 600",
        "Challenger 650",
        "Gulfstream G280",
        "Citation Longitude",
    ),
}


def extract_replacement_target(query: str) -> Optional[str]:
    """Best-effort parse of the aircraft being replaced."""
    q = query or ""
    if not _REPLACEMENT_RE.search(q):
        return None
    try:
        from services.consultant.recommendation_engine import detect_models_from_text

        found = detect_models_from_text(q)
        if found:
            return resolve_canonical_display_name(found[0]) or found[0]
    except Exception:
        pass
    for token in ("Global 7500", "Gulfstream G650ER", "Gulfstream G650", "Falcon 8X", "Global 6500"):
        if token.lower() in q.lower():
            return token
    return None


def realistic_replacement_candidates(
    target_aircraft: str,
    mission: Any,
    *,
    query: str = "",
) -> List[str]:
    """
    Return broker-credible replacement candidates for a target airframe.

    Rejects category collapse (e.g. G650 → PC-24) at the candidate-set level.
    """
    target = resolve_canonical_display_name(target_aircraft) or target_aircraft
    tier = aircraft_position_tier(target)
    pool = list(_TIER_REPLACEMENT_POOL.get(tier, _TIER_REPLACEMENT_POOL[PositionTier.SUPER_MID]))
    out: List[str] = []
    for model in pool:
        if model.lower() == target.lower():
            continue
        if is_prestige_collapse(target, model):
            continue
        if violates_class_sanity(mission, model, query=query):
            continue
        if tier_distance(tier, aircraft_position_tier(model)) > 2:
            continue
        out.append(model)
    return out


def is_credible_replacement(
    target_aircraft: str,
    candidate: str,
    mission: Any,
    *,
    query: str = "",
) -> bool:
    """Whether candidate is a broker-credible replacement for target."""
    if is_prestige_collapse(target_aircraft, candidate):
        return False
    if violates_class_sanity(mission, candidate, query=query):
        return False
    tier_t = aircraft_position_tier(target_aircraft)
    tier_c = aircraft_position_tier(candidate)
    return tier_distance(tier_t, tier_c) <= 2


def filter_recommendations_by_replacement_realism(
    recommendations: Sequence[Any],
    *,
    mission: Any,
    query: str = "",
) -> List[Any]:
    """Drop shortlist rows that violate replacement hierarchy when query implies replacement."""
    target = extract_replacement_target(query or "")
    if not target:
        return list(recommendations)
    out = []
    for rec in recommendations:
        model = getattr(rec, "model", "") or ""
        if is_credible_replacement(target, model, mission, query=query):
            out.append(rec)
    return out


__all__ = [
    "extract_replacement_target",
    "realistic_replacement_candidates",
    "is_credible_replacement",
    "filter_recommendations_by_replacement_realism",
]
