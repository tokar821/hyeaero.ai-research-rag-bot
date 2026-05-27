"""

Mission-fit policy — result limits, qualitative fit labels (no numeric user scores).

"""



from __future__ import annotations



import re

from typing import List



from services.consultant.mission_state import MissionState

from services.consultant.recommendation_engine import AircraftRecommendation

from services.recommendation.clarification_decision import (

    MissionClarificationNeeds,

    mission_clarification_needs,

    mission_maps_to_category,

    mission_well_defined,

)



STANDARD_RECOMMENDATION_LIMIT = 3

EXTENDED_RECOMMENDATION_LIMIT = 5



FIT_STRONG = "Strong Fit"

FIT_GOOD = "Good Fit"

FIT_PARTIAL = "Partial Fit"

FIT_NOT_RECOMMENDED = "Not Recommended"



# Legacy aliases (internal migration)

FIT_HIGH = FIT_STRONG

FIT_MEDIUM = FIT_GOOD

FIT_LOW = FIT_PARTIAL



_ALL_FIT_LABELS = frozenset({FIT_STRONG, FIT_GOOD, FIT_PARTIAL, FIT_NOT_RECOMMENDED})



_EXTENDED_LIST_RE = re.compile(

    r"\b(?:"

    r"top\s*5|five\s+(?:aircraft|jets|options|models)|"

    r"(?:show|give|list|name)\s+(?:me\s+)?(?:up\s+to\s+)?5|"

    r"shortlist\s+of\s+5|5\s+(?:aircraft|jets|options|models)|"

    r"compare\s+(?:all\s+)?5|all\s+five"

    r")\b",

    re.I,

)





def recommendation_limit_from_query(query: str) -> int:

    """Return 5 only when the user explicitly asks for a larger shortlist."""

    if _EXTENDED_LIST_RE.search(query or ""):

        return EXTENDED_RECOMMENDATION_LIMIT

    return STANDARD_RECOMMENDATION_LIMIT





def score_to_fit_label(score: float, *, avoid: bool = False) -> str:

    """Map internal 0..1 aggregate to qualitative fit (never shown as a decimal)."""

    if avoid:

        return FIT_NOT_RECOMMENDED

    if score >= 0.72:

        return FIT_STRONG

    if score >= 0.55:

        return FIT_GOOD

    if score >= 0.40:

        return FIT_PARTIAL

    return FIT_NOT_RECOMMENDED





def numeric_to_fit_tier(score: float) -> str:

    """Backward-compatible alias for ``score_to_fit_label``."""

    return score_to_fit_label(score)





def fit_tier_for_dimension(score: float) -> str:

    return score_to_fit_label(score)





def normalize_fit_label(label: str) -> str:

    """Coerce legacy High/Medium/Low strings to current labels."""

    key = (label or "").strip().lower()

    mapping = {

        "high": FIT_STRONG,

        "strong": FIT_STRONG,

        "strong fit": FIT_STRONG,

        "medium": FIT_GOOD,

        "good": FIT_GOOD,

        "good fit": FIT_GOOD,

        "low": FIT_PARTIAL,

        "partial": FIT_PARTIAL,

        "partial fit": FIT_PARTIAL,

        "not recommended": FIT_NOT_RECOMMENDED,

        "avoid": FIT_NOT_RECOMMENDED,

    }

    return mapping.get(key, label if label in _ALL_FIT_LABELS else FIT_GOOD)





def assign_fit_tiers(recommendations: List[AircraftRecommendation]) -> None:

    """Apply qualitative fit labels after internal sort — no numeric ranks exposed."""

    if not recommendations:

        return

    top_score = recommendations[0].total_score

    for idx, rec in enumerate(recommendations):

        if rec.avoid:

            rec.fit = FIT_NOT_RECOMMENDED

            continue

        fv = (rec.fit_verdict or "").strip().upper()
        if fv == "NOT A FIT":
            rec.fit = FIT_NOT_RECOMMENDED
            continue
        if fv == "BEST FIT" and rec.total_score >= 0.52:
            rec.fit = FIT_STRONG
            continue
        if fv == "CONDITIONAL FIT":
            rec.fit = FIT_GOOD if rec.total_score >= 0.50 else FIT_PARTIAL
            continue

        delta = max(0.0, top_score - rec.total_score)

        if idx == 0 and rec.total_score >= 0.52:

            rec.fit = FIT_STRONG

        elif delta <= 0.08:

            rec.fit = FIT_GOOD

        elif delta <= 0.20:

            rec.fit = FIT_PARTIAL

        else:

            rec.fit = FIT_PARTIAL

        if rec.total_score < 0.40:

            rec.fit = FIT_NOT_RECOMMENDED

        elif rec.total_score < 0.50 and rec.fit == FIT_STRONG:

            rec.fit = FIT_GOOD





__all__ = [

    "MissionClarificationNeeds",

    "STANDARD_RECOMMENDATION_LIMIT",

    "EXTENDED_RECOMMENDATION_LIMIT",

    "FIT_STRONG",

    "FIT_GOOD",

    "FIT_PARTIAL",

    "FIT_NOT_RECOMMENDED",

    "FIT_HIGH",

    "FIT_MEDIUM",

    "FIT_LOW",

    "mission_clarification_needs",

    "mission_maps_to_category",

    "mission_well_defined",

    "recommendation_limit_from_query",

    "score_to_fit_label",

    "numeric_to_fit_tier",

    "fit_tier_for_dimension",

    "normalize_fit_label",

    "assign_fit_tiers",

]

