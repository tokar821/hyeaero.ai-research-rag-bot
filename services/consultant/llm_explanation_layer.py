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


def build_pipeline_authority_block(
    result: AdvisoryPipelineResult,
    *,
    query: str = "",
    query_intent: str = "",
    data_used: Optional[dict] = None,
) -> str:
    """
    Mandatory pre-LLM context: mission + feasible aircraft + metadata only.

    The LLM narrates; the deterministic engine decides feasibility and ranking.
    """
    del query, query_intent

    validation = result.mission_validation or {}
    if validation.get("needs_route_clarification"):
        return (
            "[BROKER ADVISORY — CLARIFICATION ONLY]\n"
            "Do not recommend aircraft yet. Ask one focused question:\n"
            + str(validation.get("clarifying_question") or "What's the primary city pair?")
        ).strip()

    recs = [r for r in result.recommendations if not r.avoid][:MAX_BROKER_AIRCRAFT]
    if not recs:
        if isinstance(data_used, dict) and data_used.get("mission_understanding_authority"):
            return (
                "[BROKER ADVISORY — OPERATIONAL SYNTHESIS FIRST]\n"
                + str(data_used.get("mission_understanding_authority"))
                + "\n\nNo aircraft passed every hard gate — use fallback class band from understanding. "
                "Do NOT leave Aircraft Options empty. Do NOT invent brochure performance."
            ).strip()
        return (
            "[BROKER ADVISORY — NO FEASIBLE AIRCRAFT]\n"
            "No aircraft passed hard feasibility for this mission as stated.\n"
            "Explain why conservatively (range, runway, payload) and ask what constraint could change.\n"
            "Do NOT invent models. Do NOT use middleware phrasing."
        ).strip()

    block = build_broker_llm_context_block(
        result.mission_state,
        recs,
        route_assessments=[],
    )

    lines: List[str] = [
        block,
        "",
        f"Decision source: {DECISION_SOURCE} (feasibility already evaluated — do not re-score).",
    ]

    try:
        from services.preprocessing import preprocess_mission_json

        if query:
            lines.append("")
            lines.append("[PREPROCESSED MISSION JSON — facts only, do not invent routes]")
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
            lines.append("HARD-EXCLUDED (mention only if asked — never recommend):")
            lines.append(f"  {hard_ctx.summary}")
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


def build_narration_system_addendum(*, query_intent: str = "") -> str:
    """Broker-style system addendum for advisory turns."""
    base = (
        "You are a top-tier aircraft acquisition consultant — not middleware, not a report generator. "
        "Aircraft names come ONLY from the [BROKER ADVISORY CONTEXT] and [IMMUTABLE REASONING PACKET] blocks when present. "
        "You explain and critique; the engine decides feasibility and broker verdicts. "
        "You MUST NOT add aircraft beyond PRESENTED, recommend ELIMINATED aircraft, or upgrade verdicts "
        "(e.g. never call a MISSION-RISKY aircraft PRIMARY RECOMMENDATION or 'best fit'). "
        "Be concise, factual, decisive; slightly critical when appropriate. "
        "No marketing language. No generic AI phrasing. Maximum 3 aircraft. "
        "Use fixed structure: Mission Fit (Route, Pax, Priorities), "
        "Aircraft Options (Why it fits, Key compromise), Verdict (broker verdict labels exactly as given). "
        "Comparisons: only range, cabin, operating cost, runway capability, liquidity — no other dimensions. "
        "Never use: mission profile, mission score, confidence score, operationally, "
        "worth considering, if priorities shift, stage length, balanced capability, Mission Summary."
    )
    qi = (query_intent or "").strip().lower()
    if qi in ("aircraft_critique", "ownership_economics", "payload_range_analysis"):
        return base + f" This turn is {qi.replace('_', ' ')} — no acquisition shortlist."
    if qi == "aircraft_comparison":
        return base + " Compare named aircraft directly; no separate buy list."
    return base
