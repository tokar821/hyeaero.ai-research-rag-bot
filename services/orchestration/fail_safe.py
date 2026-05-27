"""Fail-safe handling and confidence scoring for orchestration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.consultant.recommendation_engine import AircraftRecommendation
from services.broker.graceful_degradation import (
    degraded_low_confidence_prefix,
    safe_broker_fallback_response,
)
from services.orchestration.constants import LOW_CONFIDENCE_THRESHOLD
from services.orchestration.tracing import OrchestrationTrace
from services.pipeline.run_pipeline import AdvisoryPipelineResult


def compute_stage_confidence(
    pipeline: Optional[AdvisoryPipelineResult],
    *,
    recommendations: Optional[List[AircraftRecommendation]] = None,
    image_confidence: Optional[float] = None,
    validation: Optional[Dict[str, Any]] = None,
    stage_failures: Optional[List[str]] = None,
) -> float:
    """Aggregate confidence from deterministic outputs (not LLM self-report)."""
    scores: List[float] = []

    validation = validation or {}
    if validation.get("needs_route_clarification"):
        scores.append(0.72)

    recs = recommendations
    if recs is None and pipeline is not None:
        recs = pipeline.recommendations

    if pipeline is not None:
        mv = pipeline.mission_validation or {}
        if mv.get("no_feasible_aircraft"):
            scores.append(0.42)
        elif not (pipeline.feasible_models or []):
            scores.append(0.48)
        else:
            scores.append(0.80)

    if recs:
        try:
            from services.recommendation.clarification_decision import (
                recommendation_confidence_sufficient,
            )

            scores.append(0.88 if recommendation_confidence_sufficient(recs) else 0.62)
        except Exception:
            scores.append(0.65)
    elif pipeline is not None and not validation.get("needs_route_clarification"):
        scores.append(0.45)

    if image_confidence is not None:
        scores.append(max(0.0, min(1.0, image_confidence)))

    if stage_failures:
        scores.append(max(0.35, 0.75 - 0.12 * len(stage_failures)))

    if not scores:
        return 0.55
    return min(scores)


def apply_low_confidence_guidance(answer: str, confidence: float) -> tuple[str, bool]:
    """Prefix answer when confidence is below threshold."""
    text = (answer or "").strip()
    if not text:
        return text, False
    if confidence >= LOW_CONFIDENCE_THRESHOLD:
        return text, False
    prefix = degraded_low_confidence_prefix()
    if prefix.lower() in text.lower():
        return text, True
    return f"{prefix}\n\n{text}", True


def finalize_trace_confidence(
    trace: OrchestrationTrace,
    pipeline: Optional[AdvisoryPipelineResult],
    *,
    recommendations: Optional[List[AircraftRecommendation]] = None,
    image_confidence: Optional[float] = None,
) -> float:
    validation = (pipeline.mission_validation if pipeline else None) or {}
    conf = compute_stage_confidence(
        pipeline,
        recommendations=recommendations,
        image_confidence=image_confidence,
        validation=validation,
        stage_failures=trace.failures,
    )
    trace.overall_confidence = conf
    trace.low_confidence = conf < LOW_CONFIDENCE_THRESHOLD
    return conf


def safe_stage_fallback(
    stage: str,
    *,
    query: str = "",
    mission: Any = None,
    pipeline: Any = None,
    recommendations: Optional[List[AircraftRecommendation]] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Broker fallback when a late stage fails — never empty, never generic retry-only text."""
    return safe_broker_fallback_response(
        query,
        mission=mission,
        pipeline=pipeline,
        recommendations=recommendations,
        data_used=data_used,
        failure_stage=stage,
    )
