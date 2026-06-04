"""Remove re-confirmed or competing decisions from stacked layer prose."""

from __future__ import annotations

import re
from typing import List, Optional

from services.truth_compression.truth_synthesizer import BrokerTruthState

_WHERE_LOOK_BLOCK = re.compile(
    r"(?im)(?:\n\s*)?where i would look:\s*\n(?:\s*[•\-][^\n]+\n)+",
)
_FOCUS_PARA = re.compile(
    r"(?im)^at \$[\d.]+\s*m,?\s*i would focus on[^\n]+\.\s*\n?",
)
_EQUAL_FOCUS = re.compile(
    r"(?im)^[^\n]*i would focus on[^\n]*(?:,|\band\b)[^\n]*(?:,|\band\b)[^\n]*\.\s*\n?",
)
_WHAT_I_WOULD_DO = re.compile(r"(?im)\n\s*what i would do:\s*[^\n]+\s*")
_KEY_RISK_DUP = re.compile(r"(?im)(\n\s*key risk:\s*[^\n]+)(?:\s*\1)+")


def deduplicate_decisions_in_answer(
    answer: str,
    truth: BrokerTruthState,
    *,
    primary_model: Optional[str] = None,
) -> str:
    """
    Strip decision-layer re-assertions when executive recommendation is authoritative.
    """
    text = (answer or "").strip()
    if not text or not truth.has_executive_recommendation:
        return text

    primary = primary_model or truth.primary_model or ""

    text = _WHERE_LOOK_BLOCK.sub("\n", text)
    text = _FOCUS_PARA.sub("", text)
    text = _EQUAL_FOCUS.sub("", text)

    if truth.recommendation and truth.recommendation.get("recommended_action"):
        text = _WHAT_I_WOULD_DO.sub("\n", text)

    eval_d = truth.evaluation or {}
    direct = str(eval_d.get("direct_answer") or "").strip()
    if direct and _has_executive_lead(text):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        filtered: List[str] = []
        direct_low = direct.lower()
        direct_prefix = direct_low[:50] if len(direct_low) >= 20 else ""
        for p in paragraphs:
            pl = p.lower()
            if pl == direct_low:
                continue
            if (
                direct_prefix
                and len(pl) < 200
                and direct_prefix in pl
                and "primary recommendation" not in pl
            ):
                continue
            if primary and _is_equal_weight_para(p, primary):
                continue
            filtered.append(p)
        if filtered:
            text = "\n\n".join(filtered)

    text = _KEY_RISK_DUP.sub(r"\1", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _has_executive_lead(text: str) -> bool:
    return bool(re.search(r"(?is)my primary recommendation would be", text))


def _is_equal_weight_para(paragraph: str, primary: str) -> bool:
    if not primary:
        return False
    low = paragraph.lower()
    if "focus on" not in low and "would look" not in low:
        return False
    if primary.lower() in low:
        return "," in low or " and " in low
    return False


__all__ = ["deduplicate_decisions_in_answer"]
