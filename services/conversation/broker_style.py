"""
Broker conversation style — direct answer first, minimal template scaffolding.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set

# Marketing language to strip (Phase 39 Goal 7).
_FORBIDDEN_MARKETING_RE = re.compile(
    r"\b(?:amazing|luxurious|incredible|perfect choice|world[- ]class|unparalleled|"
    r"best[- ]in[- ]class|game[- ]changing|ideal aircraft|great choice)\b",
    re.I,
)

# Template headers removed for non-structured intents.
_TEMPLATE_HEADERS = (
    "Overview",
    "Analysis",
    "Recommendation",
    "Risks",
    "Mission Fit",
    "Aircraft Options",
    "Mission Summary",
    "Constraint Summary",
    "Operational Synthesis",
    "Ranked Aircraft Shortlist",
    "Mission Interpretation",
    "Bottom-Line Recommendation",
    "Side-by-Side Comparison",
    "Top Aircraft Options",
)

_STRUCTURED_INTENTS: Set[str] = frozenset({"buy_decision", "comparison", "valuation", "alternative"})

# Headers allowed even in structured mode (concise).
_STRUCTURED_KEEP_HEADERS: Set[str] = frozenset(
    {
        "Market Reality",
        "Deal Assessment",
        "Red Flags",
        "Verdict",
        "Aircraft",
        "Year",
    }
)

_HEADER_LINE_RE = re.compile(
    r"(?im)^\s*(" + "|".join(re.escape(h) for h in _TEMPLATE_HEADERS + tuple(_STRUCTURED_KEEP_HEADERS)) + r")\s*:?\s*$"
)


def _strip_template_headers(text: str, intent_type: str) -> str:
    """Remove auto template headers unless intent warrants structure."""
    keep = _STRUCTURED_KEEP_HEADERS if intent_type in _STRUCTURED_INTENTS else set()
    lines: List[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        matched = False
        for header in _TEMPLATE_HEADERS:
            if re.match(rf"(?i)^{re.escape(header)}\s*:?\s*$", stripped):
                if header in keep:
                    lines.append(line)
                matched = True
                break
        if not matched:
            lines.append(line)
    return "\n".join(lines).strip()


def _comparison_to_conversational(text: str) -> str:
    """Lead comparison answers with direct contrast, not a catalog banner."""
    lines = (text or "").splitlines()
    if not lines:
        return text
    out: List[str] = []
    specs: List[str] = []
    for line in lines:
        s = line.strip()
        if re.match(r"(?i)^verified catalog comparison", s):
            continue
        if re.match(r"^-\s+\w", s):
            specs.append(re.sub(r"^-\s*", "", s))
            continue
        if specs and s.upper().startswith("VERDICT"):
            if not out:
                out.append("Side by side:")
                out.extend(f"• {sp}" for sp in specs[:2])
            specs = []
        out.append(line)
    if specs and not out:
        out.append("Side by side:")
        out.extend(f"• {sp}" for sp in specs[:2])
    elif specs:
        out.extend(f"• {sp}" for sp in specs[:2])
    return "\n".join(out).strip()


def _verdict_to_prose(text: str) -> str:
    """Convert VERDICT: Choose X if ... to inline broker recommendation."""
    out_lines: List[str] = []
    capture_verdict = False
    verdict_lines: List[str] = []

    for line in (text or "").splitlines():
        if re.match(r"(?i)^verdict\s*:?\s*$", line.strip()):
            capture_verdict = True
            continue
        if capture_verdict:
            if line.strip() and not re.match(r"(?i)^(?:overview|analysis|recommendation|risks)\s*:?\s*$", line.strip()):
                verdict_lines.append(line.strip())
            else:
                capture_verdict = False
                out_lines.append(line)
            continue
        out_lines.append(line)

    if verdict_lines:
        verdict_text = " ".join(verdict_lines)
        verdict_text = re.sub(r"^Choose\s+", "I would lean toward ", verdict_text, flags=re.I)
        if out_lines and out_lines[-1].strip():
            out_lines.append("")
        out_lines.append(verdict_text)

    return "\n".join(out_lines).strip()


def _mission_to_conversational(text: str) -> str:
    """Flatten mission advisory template into direct prose."""
    blocks: List[str] = []
    for para in re.split(r"\n\s*\n", text or ""):
        p = para.strip()
        if not p:
            continue
        if re.match(r"(?i)^(mission fit|aircraft options)\s*:", p):
            p = re.sub(r"(?im)^(Mission Fit|Aircraft Options)\s*:\s*", "", p)
        blocks.append(p)
    if not blocks:
        return text
    if len(blocks) == 1:
        return blocks[0]
    return f"{blocks[0]}\n\n" + "\n\n".join(blocks[1:])


def apply_broker_style(text: str, *, intent_type: str = "") -> str:
    """
    Apply broker conversation priorities:
    1. Direct answer first
    2. Explain reasoning
    3. Ask only necessary follow-up (handled by fallbacks)
    """
    out = (text or "").strip()
    if not out:
        return out

    out = _FORBIDDEN_MARKETING_RE.sub("", out)

    if intent_type == "comparison":
        out = _comparison_to_conversational(out)
        out = _verdict_to_prose(out)
    elif intent_type == "mission":
        out = _mission_to_conversational(out)

    out = _strip_template_headers(out, intent_type)
    out = re.sub(r" +", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def render_conversational_sections(
    *,
    overview: str = "",
    analysis: str = "",
    recommendation: str = "",
    risks: Optional[List[str]] = None,
    verdict: str = "",
    intent_type: str = "",
) -> str:
    """
    Build answer text without Overview/Analysis/Recommendation/Risks scaffolding.

    Structured intents keep concise labeled blocks only where content exists.
    """
    parts: List[str] = []

    if intent_type in _STRUCTURED_INTENTS:
        if overview.strip():
            parts.append(overview.strip())
        if analysis.strip():
            parts.append(analysis.strip())
        if recommendation.strip():
            parts.append(recommendation.strip())
        if risks:
            flagged = [r for r in risks if r.strip()]
            if flagged:
                parts.append("Worth checking:")
                parts.extend(f"• {r.strip().lstrip('-• ')}" for r in flagged[:4])
        if verdict.strip() and verdict.upper() not in (
            "INSUFFICIENT_DATA",
            "CLARIFICATION_REQUIRED",
            "INFEASIBLE_BUDGET_CONSTRAINT",
            "MARKET_CONTEXT_AVAILABLE",
        ):
            parts.append(f"My read: {verdict.strip()}")
        return "\n\n".join(parts).strip()

    # Default: conversational merge — lead with recommendation or overview.
    lead = recommendation.strip() or overview.strip() or analysis.strip()
    if lead:
        parts.append(lead)
    if analysis.strip() and analysis.strip() != lead:
        parts.append(analysis.strip())
    if risks:
        flagged = [r for r in risks if r.strip()]
        if flagged:
            parts.append("Before you commit:")
            parts.extend(f"• {r.strip().lstrip('-• ')}" for r in flagged[:3])
    if verdict.strip() and verdict.upper() not in (
        "INSUFFICIENT_DATA",
        "CLARIFICATION_REQUIRED",
        "INFEASIBLE_BUDGET_CONSTRAINT",
    ):
        parts.append(verdict.strip())

    return "\n\n".join(parts).strip()


__all__ = [
    "apply_broker_style",
    "render_conversational_sections",
]
