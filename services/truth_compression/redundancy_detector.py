"""Detect overlapping truth expression across broker layers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from services.truth_compression.truth_synthesizer import BrokerTruthState

_PATHWAY_PRIMARY_DUPLICATE = "REDUNDANT_PRIMARY_RECOMMENDATION"
_PATHWAY_EQUAL_OPTIONS = "REDUNDANT_EQUAL_WEIGHT_OPTIONS"
_PATHWAY_DECISION_MIRROR = "REDUNDANT_DECISION_MIRROR"
_PATHWAY_TEMPLATE_HEADERS = "REDUNDANT_TEMPLATE_HEADERS"
_PATHWAY_SUPPORTING_DUP = "REDUNDANT_SUPPORTING_CONTEXT"
_PATHWAY_CONFIDENCE_CHAIN = "REDUNDANT_CONFIDENCE_ASSERTIONS"

_PRIMARY_RE = re.compile(r"(?is)my primary recommendation would be")
_FOCUS_RE = re.compile(r"(?is)i would focus on")
_WHERE_LOOK_RE = re.compile(r"(?is)where i would look")
_HEADER_RE = re.compile(r"(?im)^\s*(?:overview|analysis|recommendation|risks)\s*:?\s*$")
_SUPPORTING_RE = re.compile(r"(?is)supporting market context")
_CONFIDENCE_RE = re.compile(
    r"(?is)\b(?:high confidence|moderate confidence|confidence:\s*|i am confident)\b",
)


def detect_redundant_pathways(
    answer: str,
    truth: BrokerTruthState,
    *,
    data_used: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return pathway flags for redundant multi-layer expression."""
    del data_used
    text = (answer or "").strip()
    if not text:
        return []

    pathways: List[str] = []
    primaries = len(_PRIMARY_RE.findall(text))
    if primaries > 1:
        pathways.append(_PATHWAY_PRIMARY_DUPLICATE)

    if truth.has_executive_recommendation:
        if _FOCUS_RE.search(text) and _PRIMARY_RE.search(text):
            pathways.append(_PATHWAY_EQUAL_OPTIONS)
        if _WHERE_LOOK_RE.search(text) and _PRIMARY_RE.search(text):
            pathways.append(_PATHWAY_EQUAL_OPTIONS)
        if primaries >= 1 and _FOCUS_RE.search(text):
            pathways.append(_PATHWAY_DECISION_MIRROR)

    headers = _HEADER_RE.findall(text)
    if len(headers) >= 2:
        pathways.append(_PATHWAY_TEMPLATE_HEADERS)

    if len(_SUPPORTING_RE.findall(text)) > 1:
        pathways.append(_PATHWAY_SUPPORTING_DUP)

    if len(_CONFIDENCE_RE.findall(text)) >= 2:
        pathways.append(_PATHWAY_CONFIDENCE_CHAIN)

    eval_d = truth.evaluation or {}
    if truth.has_executive_recommendation and eval_d.get("direct_answer"):
        direct = str(eval_d["direct_answer"]).strip().lower()
        if direct and direct[:60] in text.lower() and _PRIMARY_RE.search(text):
            if text.lower().index(direct[:40]) < text.lower().find("my primary recommendation"):
                pathways.append(_PATHWAY_DECISION_MIRROR)

    return list(dict.fromkeys(pathways))


__all__ = ["detect_redundant_pathways"]
