"""
Per-intent answer shape contract — one LLM voice, shape varies by turn type.

Formatting policy only; does not change ranking, retrieval, or dispatch logic.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from services.broker_execution.broker_execution_category import (
    BrokerExecutionCategory,
    classify_broker_execution_category,
)
from services.broker_execution.response_mode_classifier import ResponseMode


def build_intent_answer_contract_suffix(
    query: str,
    *,
    data_used: Optional[Dict[str, Any]] = None,
    response_mode: Optional[ResponseMode] = None,
) -> str:
    """System-prompt addendum: required answer shape for this turn."""
    q = (query or "").strip()
    du = data_used if isinstance(data_used, dict) else {}
    mode = response_mode or ResponseMode.ANALYSIS
    cat = classify_broker_execution_category(q, data_used=du)

    lines = [
        "",
        "**Answer shape contract (this turn only):**",
        "You are the sole author of the final reply. Write natural broker prose — never copy internal block labels.",
        "Forbidden section headers: Overview, Analysis, Recommendation, Risks, Mission Fit, Aircraft Options, "
        "Verdict, Operational synthesis, Key risk, What I would do, Where I would look, Before treating.",
    ]

    depth = str(du.get("tail_depth_mode") or "").strip().lower()
    if cat in (
        BrokerExecutionCategory.TAIL_OWNERSHIP,
        BrokerExecutionCategory.TAIL_LOOKUP,
        BrokerExecutionCategory.REGISTRY_LOOKUP,
    ) or mode == ResponseMode.FACT_ONLY:
        if depth == "owner":
            lines.extend(
                [
                    "Shape: ONE sentence with the registered owner name first, then aircraft type.",
                    "Optional second line: year or serial only if in context. No other fields unless asked.",
                    "Do NOT mention logbooks, engine programs, maintenance, or acquisition diligence.",
                ]
            )
        elif depth == "sale_status":
            lines.extend(
                [
                    "Shape: First sentence must answer for-sale yes/no (and approximate ask if in context).",
                    "Then at most 3 supporting facts (aircraft type, owner, year). No diligence checklist.",
                ]
            )
        elif depth == "detail":
            lines.extend(
                [
                    "Shape: short registry summary, then listing/market/usage context when present in facts.",
                    "Up to 8–12 lines; still no Mission Fit / Verdict / Operational synthesis headings.",
                ]
            )
        else:
            lines.extend(
                [
                    "Shape: 2–5 short lines — registration, aircraft type, owner, year/serial if known.",
                    "Do not add acquisition advice, mission fit, or diligence essay unless explicitly asked.",
                ]
            )
    elif cat == BrokerExecutionCategory.COMPARISON or mode == ResponseMode.COMPARISON:
        lines.extend(
            [
                "Shape: open with which model wins for the user's stated mission or buyer profile.",
                "Then give 2–3 concrete wins per side, one key tradeoff, and a 'buy X if…' line for each aircraft.",
                "Cover range, cabin, operating cost, runway, and liquidity using comparison broker facts only.",
                "Do not use Overview/Analysis/Recommendation/Risks headings.",
            ]
        )
    elif cat == BrokerExecutionCategory.LISTING or mode == ResponseMode.LISTING:
        lines.extend(
            [
                "Shape: price realism first (1–2 sentences), then 2–3 concrete risks or verification steps.",
                "No 'Where I would look' lists, no 'send me the listing package', no bargain boilerplate.",
            ]
        )
    elif cat == BrokerExecutionCategory.MISSION or mode == ResponseMode.MISSION:
        lines.extend(
            [
                "Shape: conversational opening, then at most three aircraft from verified mission facts only.",
                "State nonstop feasibility honestly. If no aircraft pass filters, explain why — do not invent models.",
                "Never output INSUFFICIENT_DATA or template verdict blocks.",
            ]
        )
    elif cat == BrokerExecutionCategory.TAIL_HISTORY:
        lines.append("Shape: factual history summary only; no purchase recommendation.")
    else:
        lines.append("Shape: concise expert answer; lead with the direct response to the latest question.")

    if du.get("authority_dispatch_deferred_llm") and str(du.get("authority_dispatch_kind") or "") == "comparison":
        lines.append("Use only structured comparison facts in context; do not invent metrics.")

    return "\n".join(lines)
