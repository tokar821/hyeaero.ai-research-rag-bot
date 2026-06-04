"""Classify where the buyer is in the acquisition journey."""

from __future__ import annotations

import re
from enum import Enum


class AcquisitionStage(str, Enum):
    EXPLORING = "EXPLORING"
    SHORTLISTING = "SHORTLISTING"
    ACTIVE_SHOPPING = "ACTIVE_SHOPPING"
    NEGOTIATING = "NEGOTIATING"
    DUE_DILIGENCE = "DUE_DILIGENCE"


_TAIL_RE = re.compile(r"\bN\d{1,5}[A-Z]{0,2}\b", re.I)
_SAW_LISTING_RE = re.compile(
    r"(?is)\b(?:saw|found|listing|asking|listed|for\s+sale)\b",
)
_COMPARE_RE = re.compile(r"\b(?:compare|vs\.?|versus)\b", re.I)
_BUDGET_DISCOVERY_RE = re.compile(
    r"(?is)\b(?:what\s+(?:can|should)\s+i\s+buy|best\s+(?:jet|aircraft)|smartest\s+jet)\b",
)
_DD_RE = re.compile(
    r"(?is)\b(?:logbook|pre[- ]?buy|prebuy|inspection|maintenance\s+records|damage\s+history|"
    r"engine\s+program|spec\s+sheet)\b",
)
_NEGOTIATE_RE = re.compile(
    r"(?is)\b(?:offer|loi|letter\s+of\s+intent|counter|negotiat|escrow|deposit)\b",
)
_SHORTLIST_RE = re.compile(
    r"(?is)\b(?:shortlist|narrow\s+down|top\s+(?:two|three|2|3)|which\s+(?:one|jet)\s+should)\b",
)
_TIMING_RE = re.compile(
    r"(?is)\b(?:should\s+i\s+buy\s+now|buy\s+now\s+or\s+wait|wait\s+until)\b",
)


def detect_acquisition_stage(query: str, *, prior_stage: str = "") -> AcquisitionStage:
    q = (query or "").strip()
    if not q:
        return AcquisitionStage(prior_stage) if prior_stage in AcquisitionStage.__members__ else AcquisitionStage.EXPLORING

    if _DD_RE.search(q):
        return AcquisitionStage.DUE_DILIGENCE
    if _NEGOTIATE_RE.search(q) or _TAIL_RE.search(q):
        return AcquisitionStage.NEGOTIATING
    if _SAW_LISTING_RE.search(q) or re.search(r"(?is)\bgood\s+deal\b", q):
        return AcquisitionStage.ACTIVE_SHOPPING
    if _TIMING_RE.search(q) and prior_stage in (
        AcquisitionStage.ACTIVE_SHOPPING.value,
        AcquisitionStage.NEGOTIATING.value,
    ):
        return AcquisitionStage.ACTIVE_SHOPPING
    if _SHORTLIST_RE.search(q) or _COMPARE_RE.search(q):
        return AcquisitionStage.SHORTLISTING
    if _BUDGET_DISCOVERY_RE.search(q):
        return AcquisitionStage.EXPLORING

    if prior_stage in AcquisitionStage.__members__:
        return AcquisitionStage(prior_stage)
    return AcquisitionStage.EXPLORING


def stage_rank(stage: str) -> int:
    order = [
        AcquisitionStage.EXPLORING,
        AcquisitionStage.SHORTLISTING,
        AcquisitionStage.ACTIVE_SHOPPING,
        AcquisitionStage.NEGOTIATING,
        AcquisitionStage.DUE_DILIGENCE,
    ]
    try:
        return order.index(AcquisitionStage(stage))
    except ValueError:
        return 0


def merge_stage(prior: str, detected: AcquisitionStage) -> str:
    """Never regress stage unless user resets topic."""
    if not prior:
        return detected.value
    if stage_rank(detected.value) >= stage_rank(prior):
        return detected.value
    return prior


__all__ = [
    "AcquisitionStage",
    "detect_acquisition_stage",
    "merge_stage",
]
