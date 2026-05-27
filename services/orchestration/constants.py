"""
Consultant orchestration constants — stage order and LLM authority boundaries.
"""

from __future__ import annotations

from typing import Tuple

# Canonical pipeline order (deterministic stages before LLM narration).
ORCHESTRATION_STAGES: Tuple[str, ...] = (
    "mission_extraction",
    "feasibility_engine",
    "aircraft_filtering",
    "recommendation_ranking",
    "broker_narrative_generation",
    "image_verification",
    "final_response_formatting",
)

DECISION_SOURCE = "deterministic_orchestration"

LOW_CONFIDENCE_THRESHOLD = 0.55

def _low_confidence_prefix() -> str:
    from services.broker.graceful_degradation import degraded_low_confidence_prefix

    return degraded_low_confidence_prefix()


# Lazy alias for tests and legacy imports — degradation prose, not refusal.
LOW_CONFIDENCE_GUIDANCE_PREFIX = _low_confidence_prefix()

# The LLM must NEVER perform these (enforced via authority blocks + reconciliation).
LLM_FORBIDDEN_CAPABILITIES: Tuple[str, ...] = (
    "determine_raw_feasibility",
    "override_hard_rejects",
    "hallucinate_operational_capability",
    "invent_aircraft_shortlist",
    "re_score_eliminated_aircraft",
)

# The LLM may ONLY perform these on advisory turns.
LLM_ALLOWED_CAPABILITIES: Tuple[str, ...] = (
    "explain",
    "compare",
    "advise",
    "narrate_pipeline_output",
)
