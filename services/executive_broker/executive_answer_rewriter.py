"""Rewrite multi-option answers into one-broker executive structure."""

from __future__ import annotations

import re
from typing import List, Optional

from services.executive_broker.broker_consistency_audit import BrokerConsistencyScore
from services.executive_broker.executive_recommendation import ExecutiveRecommendation

_EQUAL_LIST_RE = re.compile(
    r"(?is)\b(?:focus on|would look at|options include|several options|here are)\b.{0,80}"
    r"(?:,|\band\b).{0,40}(?:,|\band\b)",
)
_BULLET_MODEL_RE = re.compile(
    r"(?im)^\s*[•\-]\s+(?:Gulfstream\s+)?[A-Za-z][\w\s+\-]{2,40}(?:\s+—|\s+-|:)",
)
_PRIMARY_LEAD_RE = re.compile(r"(?is)^my primary recommendation would be")


def has_equal_weight_recommendations(answer: str) -> bool:
    """Detect committee-style equal lists in prose."""
    body = (answer or "").strip()
    if not body:
        return False
    if _PRIMARY_LEAD_RE.search(body[:200]):
        return False
    if _EQUAL_LIST_RE.search(body):
        return True
    bullets = _BULLET_MODEL_RE.findall(body)
    if len(bullets) >= 3:
        return True
    if re.search(
        r"(?is)\bwhere i would look:\s*\n(?:\s*[•\-].+\n){2,}",
        body,
    ):
        return True
    # "X, Y, and Z — not every jet"
    if len(re.findall(r",\s*(?:Gulfstream\s+)?[A-Z][\w\s+\-]{2,30}", body[:500])) >= 2:
        if "primary recommendation" not in body.lower():
            return True
    return False


def _strip_equal_weight_sections(answer: str) -> str:
    """Remove 'Where I would look' blocks and comma-separated option dumps."""
    lines = (answer or "").splitlines()
    out: List[str] = []
    skip = False
    for line in lines:
        if re.match(r"(?i)^\s*where i would look:\s*$", line.strip()):
            skip = True
            continue
        if skip and re.match(r"^\s*[•\-]\s+", line):
            continue
        if skip and line.strip() and not re.match(r"^\s*[•\-]", line):
            skip = False
        if not skip:
            out.append(line)
    return "\n".join(out).strip()


def rewrite_executive_answer(
    answer: str,
    recommendation: ExecutiveRecommendation,
    *,
    consistency: Optional[BrokerConsistencyScore] = None,
    preserve_market_block: bool = True,
) -> str:
    """
    Structure: direct answer → primary recommendation → rationale → alternatives.
    """
    raw = (answer or "").strip()
    primary = recommendation.primary_recommendation
    rationale = recommendation.rationale

    parts: List[str] = []

    direct = (recommendation.direct_answer or "").strip()
    if direct and not _PRIMARY_LEAD_RE.search(direct):
        # Shorten direct if it already lists multiple models equally
        if has_equal_weight_recommendations(direct):
            budget_m = re.search(r"\$(\d+(?:\.\d+)?)\s*M", direct)
            if budget_m:
                parts.append(
                    f"At ${budget_m.group(1)}M, you have real options — but I would not treat them as equal."
                )
            else:
                parts.append("You have several paths — I would not treat them as equal.")
        else:
            parts.append(direct)
    elif raw and not has_equal_weight_recommendations(raw):
        first = re.split(r"\n\s*\n", raw)[0].strip()
        if first and not _PRIMARY_LEAD_RE.search(first):
            parts.append(first)

    parts.append(f"My primary recommendation would be the {primary} - {rationale}.")

    if recommendation.alternatives:
        parts.append("\nIf the first tail doesn't check out, my backup would be:")
        for alt in recommendation.alternatives[:2]:
            model = alt.get("model", "")
            why = alt.get("rationale", "")
            if model:
                line = f"• {model}"
                if why:
                    line += f" - {why}"
                parts.append(line)

    if recommendation.rejected_options and recommendation.confidence in ("HIGH", "MODERATE"):
        rej = recommendation.rejected_options[:2]
        if rej:
            parts.append("\nI would not lead with:")
            for r in rej:
                parts.append(f"• {r.get('model', '')} - {r.get('reason', '')}")

    if consistency and consistency.notes and recommendation.confidence == "LOW":
        parts.append(f"\nNote: {' '.join(consistency.notes[:2])}")

    synthesized = "\n".join(parts).strip()

    if preserve_market_block and raw:
        tail = _extract_preservable_tail(raw, synthesized)
        if tail:
            synthesized = f"{synthesized}\n\n{tail}".strip()

    return synthesized


def _extract_preservable_tail(raw: str, synthesized: str) -> str:
    """Keep market-reality or supporting blocks not duplicated in executive lead."""
    blocks: List[str] = []
    for marker in (
        r"(?is)supporting market context:",
        r"(?is)before treating it as a bargain",
        r"(?is)leverage:",
        r"(?is)inventory:",
    ):
        m = re.search(marker, raw)
        if m:
            chunk = raw[m.start() :].strip()
            if chunk.lower() not in synthesized.lower():
                blocks.append(chunk)
                break
    return blocks[0] if blocks else ""


__all__ = ["has_equal_weight_recommendations", "rewrite_executive_answer"]
