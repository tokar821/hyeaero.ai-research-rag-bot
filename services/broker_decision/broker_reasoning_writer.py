"""Transform BrokerDecision into acquisition-advisor prose."""

from __future__ import annotations

import re
from typing import List

from services.broker_decision.broker_decision_builder import BrokerDecision


def write_broker_decision(
    decision: BrokerDecision,
    *,
    raw_answer: str = "",
    preserve_supporting: bool = True,
) -> str:
    """
    Render decision with direct answer first; specs from raw_answer become supporting evidence.
    """
    parts: List[str] = []

    # Lead with direct answer — mandatory first paragraph.
    lead = (decision.direct_answer or "").strip()
    if decision.answer_type == "yes_no" and lead.lower() in ("no", "no."):
        parts.append("No.")
        rest = [p for p in decision.supporting_points if p.strip()]
        if rest:
            parts.append("\n\n".join(rest))
    else:
        parts.append(lead)

    if decision.key_risk:
        parts.append(f"\n\nKey risk: {decision.key_risk.strip()}")

    if decision.alternatives:
        parts.append("\n\nWhere I would look:")
        for alt in decision.alternatives[:4]:
            model = alt.get("model", "")
            rationale = alt.get("rationale", "")
            if model:
                line = f"• {model}"
                if rationale:
                    line += f" — {rationale}"
                parts.append(line)

    if decision.recommended_action:
        parts.append(f"\n\nWhat I would do: {decision.recommended_action.strip()}")

    synthesized = "\n".join(parts).strip()

    if preserve_supporting and raw_answer:
        supporting = _extract_supporting_evidence(raw_answer)
        if supporting and supporting.lower() not in synthesized.lower():
            synthesized = f"{synthesized}\n\nSupporting market context:\n{supporting}"

    return synthesized.strip()


def _extract_supporting_evidence(raw_answer: str) -> str:
    """Pull market band / deal lines from pipeline answer without leading dumps."""
    lines: List[str] = []
    skip_headers = re.compile(
        r"(?i)^(?:verified catalog|aircraft options|mission fit|insufficient|verdict:\s*INSUFFICIENT)",
    )

    for line in (raw_answer or "").splitlines():
        s = line.strip()
        if not s or skip_headers.match(s):
            continue
        if re.search(r"(?i)\b(?:market band|median|liquidity|good deal|overpriced|deal assessment)\b", s):
            lines.append(f"• {s.lstrip('•- ')}")
        if len(lines) >= 4:
            break
    return "\n".join(lines)


__all__ = ["write_broker_decision"]
