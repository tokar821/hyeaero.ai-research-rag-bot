"""
Canonical deterministic recommendation pipeline.

Architecture (decision maker = code, not LLM):

  User Query
    → Mission Extraction
    → Constraint / Validation Engine
    → Aircraft Filtering (capability graph + feasibility)
    → Weighted Ranking (mission ranker + archetype weighting)
    → LLM Explanation Layer (narration only — see ``llm_explanation_layer``)

The LLM must never invent the shortlist; it only explains pipeline output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.pipeline.run_pipeline import (
    AdvisoryPipelineResult,
    run_advisory_pipeline,
)

from services.orchestration.constants import DECISION_SOURCE, ORCHESTRATION_STAGES

PIPELINE_STAGES = ("query_intent_classification",) + ORCHESTRATION_STAGES


@dataclass
class RecommendationPipelineTrace:
    """Audit trail for telemetry and LLM context."""

    stages_completed: List[str] = field(default_factory=list)
    decision_source: str = DECISION_SOURCE
    query_recommendation_intent: str = ""
    mission_category: str = ""
    feasible_count: int = 0
    eliminated_count: int = 0
    ranked_models: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stages_completed": list(self.stages_completed),
            "decision_source": self.decision_source,
            "query_recommendation_intent": self.query_recommendation_intent,
            "mission_category": self.mission_category,
            "feasible_count": self.feasible_count,
            "eliminated_count": self.eliminated_count,
            "ranked_models": list(self.ranked_models),
        }


def run_recommendation_pipeline(
    query: str,
    *,
    conversation_state: Optional[Dict[str, Any]] = None,
    data_used: Optional[Dict[str, Any]] = None,
    explicit_candidates: Optional[List[str]] = None,
    max_results: int = 3,
    query_intent: Optional[str] = None,
) -> tuple[AdvisoryPipelineResult, RecommendationPipelineTrace]:
    """
    Run the full deterministic recommendation pipeline.

    This is the only path that may decide which aircraft appear in a mission shortlist.
    """
    from services.recommendation.query_recommendation_intent import (
        QueryRecommendationIntent,
        apply_query_intent_metadata,
        classify_query_recommendation_intent,
        requires_ranked_aircraft_pipeline,
    )

    from services.preprocessing import attach_mission_preprocessing

    attach_mission_preprocessing(data_used, query)

    qri = classify_query_recommendation_intent(query)
    if query_intent:
        try:
            qri.intent = QueryRecommendationIntent(query_intent)
            qri.requires_ranked_pipeline = requires_ranked_aircraft_pipeline(qri.intent)
        except ValueError:
            pass
    if isinstance(data_used, dict):
        apply_query_intent_metadata(data_used, qri)

    from services.orchestration.pipeline_orchestrator import run_deterministic_stages

    result, orch_trace = run_deterministic_stages(
        query,
        conversation_state=conversation_state,
        data_used=data_used,
        explicit_candidates=explicit_candidates,
        max_results=max_results,
    )
    trace = RecommendationPipelineTrace(
        stages_completed=["query_intent_classification"] + orch_trace.completed_stage_names(),
        query_recommendation_intent=qri.intent.value,
        mission_category=result.mission_category.value if result.mission_category else "",
        feasible_count=len(result.feasible_models),
        eliminated_count=len(result.eliminated_models),
        ranked_models=[r.model for r in result.recommendations if not r.avoid],
    )
    if isinstance(data_used, dict):
        data_used["recommendation_pipeline"] = trace.to_dict()
        data_used["recommendation_decision_source"] = DECISION_SOURCE
        data_used["orchestration"] = orch_trace.to_dict()
        if result.recommendation_audit:
            data_used["recommendation_audit"] = dict(result.recommendation_audit)
    return result, trace


def pipeline_result_to_storage(result: AdvisoryPipelineResult) -> Dict[str, Any]:
    """Serialize pipeline output for ``data_used`` / client echo."""
    return {
        "mission_category": result.mission_category.value if result.mission_category else "",
        "recommendations": [r.to_dict() for r in result.recommendations],
        "feasible_models": list(result.feasible_models),
        "eliminated_models": list(result.eliminated_models),
        "mission_validation": dict(result.mission_validation or {}),
        "elimination_log": list(result.elimination_log or []),
        "recommendation_audit": dict(result.recommendation_audit or {}),
    }
