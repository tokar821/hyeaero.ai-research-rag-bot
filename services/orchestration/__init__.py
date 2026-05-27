"""Consultant pipeline orchestration — traced stages, fail-safe, production/debug modes."""

from services.orchestration.constants import (
    DECISION_SOURCE,
    LLM_ALLOWED_CAPABILITIES,
    LLM_FORBIDDEN_CAPABILITIES,
    LOW_CONFIDENCE_GUIDANCE_PREFIX,
    LOW_CONFIDENCE_THRESHOLD,
    ORCHESTRATION_STAGES,
)
from services.orchestration.modes import OrchestrationMode, orchestration_enabled, orchestration_mode
from services.orchestration.pipeline_orchestrator import (
    ConsultantOrchestrationResult,
    run_consultant_orchestration,
    run_deterministic_stages,
)
from services.orchestration.tracing import OrchestrationTrace, StageRecord

__all__ = [
    "DECISION_SOURCE",
    "LLM_ALLOWED_CAPABILITIES",
    "LLM_FORBIDDEN_CAPABILITIES",
    "LOW_CONFIDENCE_GUIDANCE_PREFIX",
    "LOW_CONFIDENCE_THRESHOLD",
    "ORCHESTRATION_STAGES",
    "ConsultantOrchestrationResult",
    "OrchestrationMode",
    "OrchestrationTrace",
    "StageRecord",
    "orchestration_enabled",
    "orchestration_mode",
    "run_consultant_orchestration",
    "run_deterministic_stages",
]
