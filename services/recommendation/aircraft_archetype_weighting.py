"""
Aircraft archetype weighting — favor modern operator preferences over legacy defaults.

Scores reflect what active charter, corporate, and owner-operators spec today,
not historically common brochure shortlists (e.g. aging light jets on long missions).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple


class OperatorArchetype(str, Enum):
    MODERN_FLAGSHIP = "modern_flagship"
    MODERN_OPERATOR_PREFERRED = "modern_operator_preferred"
    CONTEMPORARY = "contemporary"
    LEGACY_COMMON = "legacy_common"
    SPECIALIZED_UTILITY = "specialized_utility"


# Models in the operational catalog (see mission.aircraft_profiles)
_MODEL_ARCHETYPE: Dict[str, OperatorArchetype] = {
    # Modern operator preferred — current production / dominant new-buy spec
    "Challenger 350": OperatorArchetype.MODERN_OPERATOR_PREFERRED,
    "Citation Latitude": OperatorArchetype.MODERN_OPERATOR_PREFERRED,
    "Citation CJ4": OperatorArchetype.MODERN_OPERATOR_PREFERRED,
    "Praetor 600": OperatorArchetype.MODERN_OPERATOR_PREFERRED,
    "Gulfstream G280": OperatorArchetype.MODERN_OPERATOR_PREFERRED,
    "Pilatus PC-24": OperatorArchetype.MODERN_OPERATOR_PREFERRED,
    "Challenger Longitude": OperatorArchetype.MODERN_OPERATOR_PREFERRED,
    # Flagship / current-generation ULR & large cabin
    "Gulfstream G650": OperatorArchetype.MODERN_FLAGSHIP,
    "Global 7500": OperatorArchetype.MODERN_FLAGSHIP,
    "Falcon 8X": OperatorArchetype.MODERN_FLAGSHIP,
    # Still operated widely but superseded by newer peers for new missions
    "Challenger 650": OperatorArchetype.CONTEMPORARY,
    "Falcon 7X": OperatorArchetype.CONTEMPORARY,
    # Legacy common — historically listed; not what modern operators lead with
    "Falcon 2000": OperatorArchetype.LEGACY_COMMON,
    "Citation CJ2": OperatorArchetype.LEGACY_COMMON,
    "Learjet 75": OperatorArchetype.LEGACY_COMMON,
    # Specialized — appropriate when mission fits; not "legacy" in the negative sense
    "Pilatus PC-12": OperatorArchetype.SPECIALIZED_UTILITY,
}

_ARCHETYPE_BASE_SCORE: Dict[OperatorArchetype, float] = {
    OperatorArchetype.MODERN_FLAGSHIP: 0.98,
    OperatorArchetype.MODERN_OPERATOR_PREFERRED: 0.92,
    OperatorArchetype.CONTEMPORARY: 0.74,
    OperatorArchetype.LEGACY_COMMON: 0.46,
    OperatorArchetype.SPECIALIZED_UTILITY: 0.84,
}

# Fine adjustments within archetype (dispatch / support / still in production)
_MODEL_ADJUSTMENT: Dict[str, float] = {
    "Challenger 350": 0.03,
    "Praetor 600": 0.02,
    "Citation Latitude": 0.01,
    "Falcon 2000": -0.04,
    "Citation CJ2": -0.06,
    "Learjet 75": -0.08,
    "Gulfstream G650": 0.01,
    "Global 7500": 0.02,
}


def operator_archetype_for_model(model: str) -> OperatorArchetype:
    return _MODEL_ARCHETYPE.get(model, OperatorArchetype.CONTEMPORARY)


def modern_operational_fit_score(
    model: str,
    profile: Optional[Dict[str, Any]] = None,
) -> float:
    """
    0..1 score for how well the type matches modern operator buying and dispatch norms.
    """
    if profile is not None and profile.get("modern_operational_fit") is not None:
        try:
            return max(0.0, min(1.0, float(profile["modern_operational_fit"])))
        except (TypeError, ValueError):
            pass

    archetype = operator_archetype_for_model(model)
    base = _ARCHETYPE_BASE_SCORE.get(archetype, 0.65)
    adj = _MODEL_ADJUSTMENT.get(model, 0.0)
    return max(0.0, min(1.0, base + adj))


def archetype_ranking_note(model: str, score: float) -> str:
    archetype = operator_archetype_for_model(model)
    if score >= 0.9:
        return (
            f"{model} is a current-generation type operators actively spec for this class — "
            "strong modern operational fit."
        )
    if archetype == OperatorArchetype.LEGACY_COMMON:
        return (
            f"{model} is a historically common listing, but modern fleets usually step to "
            "a current-production peer unless budget or availability forces the older airframe."
        )
    if archetype == OperatorArchetype.CONTEMPORARY:
        return (
            f"{model} remains viable, though many operators today prefer a newer large-cabin "
            "or super-midsize alternative for the same mission."
        )
    if archetype == OperatorArchetype.SPECIALIZED_UTILITY:
        return f"{model} fits specialized utility missions where turboprop or STOL economics win."
    return f"{model} — moderate modern operational fit for this mission profile."


def modern_operational_fit_for_ranking(
    model: str,
    profile: Dict[str, Any],
) -> Tuple[float, str]:
    """Score + advisor note for the mission ranker dimension."""
    score = modern_operational_fit_score(model, profile)
    return score, archetype_ranking_note(model, score)


def enrich_profile_with_archetype(model: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Attach archetype metadata to a profile dict (non-destructive copy)."""
    out = dict(profile)
    archetype = operator_archetype_for_model(model)
    out["operator_archetype"] = archetype.value
    out["modern_operational_fit"] = modern_operational_fit_score(model, profile)
    return out
