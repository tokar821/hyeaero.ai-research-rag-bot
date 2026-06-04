"""
LLM explanation layer — broker advisory narration only.

The model receives mission facts, feasible aircraft, and metadata.
It must NOT invent feasibility or change the shortlist.
"""

from __future__ import annotations

from typing import List, Optional

from services.consultant.broker_advisory_layer import (
    MAX_BROKER_AIRCRAFT,
    build_broker_llm_context_block,
)
from services.pipeline.run_pipeline import AdvisoryPipelineResult
from services.orchestration.constants import DECISION_SOURCE


def intents_requiring_deterministic_pipeline(fine_intent_value: str) -> bool:
    """Legacy fine-intent gate (prefer ``should_run_pre_llm_pipeline`` + query intent)."""
    return fine_intent_value in (
        "aircraft_recommendation",
        "aviation_mission",
        "aircraft_comparison",
    )


def build_pipeline_llm_fact_block(
    result: AdvisoryPipelineResult,
    *,
    query: str = "",
    query_intent: str = "",
    data_used: Optional[dict] = None,
) -> str:
    """
    Facts-only context for the LLM final renderer.

    No report scaffolds, operational-synthesis prose, or copy-paste section templates.
    """
    del query_intent

    validation = result.mission_validation or {}
    if validation.get("needs_route_clarification"):
        return (
            "[VERIFIED MISSION FACTS — clarification required]\n"
            "needs_route_clarification: true\n"
            "clarifying_question: "
            + str(validation.get("clarifying_question") or "What's the primary city pair?")
        ).strip()

    recs = [r for r in result.recommendations if not r.avoid][:MAX_BROKER_AIRCRAFT]
    if not recs:
        if isinstance(data_used, dict) and data_used.get("mission_understanding_authority"):
            return (
                "[VERIFIED MISSION FACTS — class band only; no aircraft passed hard gates]\n"
                "mission_understanding: "
                + str(data_used.get("mission_understanding_authority"))[:3000]
                + "\nrules: Explain conservatively; do not invent models or brochure performance."
            ).strip()
        return (
            "[VERIFIED MISSION FACTS — no feasible aircraft]\n"
            "feasible_aircraft: []\n"
            "rules: Explain why (range, runway, payload) and ask what constraint could change. "
            "Do not invent models."
        ).strip()

    block = build_broker_llm_context_block(
        result.mission_state,
        recs,
        route_assessments=[],
        data_used=data_used,
    )

    lines: List[str] = [
        block,
        "",
        f"decision_source: {DECISION_SOURCE}",
        "rules: Feasibility and ranking are final — narrate only; do not re-score or add aircraft.",
    ]

    try:
        from services.preprocessing import preprocess_mission_json

        if query:
            lines.append("")
            lines.append("[PREPROCESSED MISSION JSON — facts only]")
            lines.append(preprocess_mission_json(query))
    except Exception:
        pass

    try:
        from services.recommendation.hard_mission_elimination import (
            detect_hard_elimination_context,
            hard_excluded_model_set,
        )

        hard_ctx = detect_hard_elimination_context(result.mission_profile)
        if hard_ctx is not None:
            hard_set = hard_excluded_model_set(result.mission_profile)
            lines.append("")
            lines.append("hard_excluded_models:")
            lines.append(f"  summary: {hard_ctx.summary}")
            for model in sorted(hard_set)[:8]:
                lines.append(f"  - {model}")
    except Exception:
        pass

    try:
        from services.telemetry.reasoning_packet_enforcement import (
            extract_reasoning_packet,
            format_immutable_reasoning_packet_block,
        )

        packet = extract_reasoning_packet(data_used)
        if packet:
            lines.append("")
            lines.append(format_immutable_reasoning_packet_block(packet))
    except Exception:
        pass

    return "\n".join(lines).strip()


def build_pipeline_authority_block(
    result: AdvisoryPipelineResult,
    *,
    query: str = "",
    query_intent: str = "",
    data_used: Optional[dict] = None,
) -> str:
    """Alias for fact-only LLM context (legacy name retained for callers)."""
    return build_pipeline_llm_fact_block(
        result,
        query=query,
        query_intent=query_intent,
        data_used=data_used,
    )


def build_narration_system_addendum(*, query_intent: str = "") -> str:
    """System addendum for LLM-primary turns — natural prose, no report templates."""
    base = (
        "You are the sole author of the client-facing answer — one expert aircraft broker voice. "
        "Structured context blocks contain verified facts only; turn them into clear, direct prose. "
        "Do NOT mirror internal labels, bullet scaffolds, or report section headings in the reply. "
        "Forbidden in the user-visible answer: Mission Fit, Aircraft Options, Verdict, "
        "Operational synthesis, Key risk, What I would do, Before treating it as a bargain, "
        "mission profile, mission score, confidence score. "
        "Lead with the direct answer to the latest question; short paragraphs; no spec dumps unless asked. "
        "Aircraft names only from verified context — never invent feasibility, range, or models. "
        "At most three aircraft when recommending. No marketing or generic AI phrasing. "
        "Comparisons: range, cabin, operating cost, runway capability, liquidity only."
    )
    qi = (query_intent or "").strip().lower()
    if qi in ("aircraft_critique", "ownership_economics", "payload_range_analysis"):
        return base + f" This turn is {qi.replace('_', ' ')} — no acquisition shortlist."
    if qi == "aircraft_comparison":
        return base + " Compare named aircraft directly; no separate buy list."
    return base
