"""
Mission-fit aircraft recommendation ranking.
"""

from services.recommendation.diversity_controls import (
    RecommendationSelectionAudit,
    apply_diversity_controls,
    apply_hard_feasibility_gate,
)
from services.recommendation.mission_ranker import (
    MissionCategory,
    classify_mission_category,
    rank_missions,
)
from services.recommendation.recommendation_diversity_guard import (
    DEFAULT_TRIAD_MODELS,
    apply_recommendation_diversity_guard,
    genuinely_scores_highest,
)

__all__ = [
    "MissionCategory",
    "classify_mission_category",
    "rank_missions",
    "RecommendationSelectionAudit",
    "apply_diversity_controls",
    "apply_hard_feasibility_gate",
    "DEFAULT_TRIAD_MODELS",
    "apply_recommendation_diversity_guard",
    "genuinely_scores_highest",
]
