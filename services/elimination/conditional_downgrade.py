"""
Conditional elimination — hard-remove only when operationally impossible; otherwise downgrade.

Hard eliminate: corridor-invalid, runway-invalid, unsafe mountain leg, impossible range band.
Soft downgrade: in-band comparison losers, marginal band mismatch, tight payload/margin.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional

from services.mission.feasibility_engine import FeasibilityResult


class CompromiseLabel(str, Enum):
    VIABLE_WITH_COMPROMISES = "VIABLE WITH COMPROMISES"
    PAYLOAD_LIMITED = "PAYLOAD LIMITED"
    TECH_STOP_LIKELY = "TECH STOP LIKELY"
    SEASONALLY_CONDITIONAL = "SEASONALLY CONDITIONAL"
    OUT_OF_BAND_COMPROMISE = "OUT OF BAND — COMPROMISE CANDIDATE"


_HARD_REASON_PATTERNS = (
    re.compile(r"corridor[- ]invalid|corridor\s+elimination|corridor[- ]hard", re.I),
    re.compile(r"runway\s+footprint.*not\s+aligned", re.I),
    re.compile(r"heavy\s+cabin.*mountain", re.I),
    re.compile(r"insufficient\s+hot/high\s+field", re.I),
    re.compile(r"ulr-only|transatlantic.*nonstop", re.I),
    re.compile(r"operationally\s+impossible", re.I),
)

_SOFT_IN_BAND_RE = re.compile(r"not\s+comparable\s+in[- ]band", re.I)
_SOFT_BAND_RE = re.compile(r"outside\s+operational\s+band", re.I)


def elimination_severity(
    reason: str,
    *,
    distance_nm: float = 0.0,
    elimination_kind: str = "band",
) -> str:
    """Return ``hard`` or ``soft``."""
    text = (reason or "").strip()
    if not text:
        return "soft"

    for pat in _HARD_REASON_PATTERNS:
        if pat.search(text):
            return "hard"

    if _SOFT_IN_BAND_RE.search(text):
        return "soft"

    if _SOFT_BAND_RE.search(text) or elimination_kind == "band":
        # Light/mid/turboprop on long stage — operationally implausible as primary platform.
        if distance_nm >= 2400 and re.search(
            r"\b(?:light_jet|midsize|turboprop)\b", text, re.I
        ):
            return "hard"
        # Super-mid on ULR-class corridor — downgrade (tech-stop / payload), do not disappear.
        return "soft"

    if elimination_kind == "mountain" and "heavy cabin" in text.lower():
        return "hard"

    return "soft"


def compromise_label_for_reason(
    reason: str,
    *,
    distance_nm: float = 0.0,
    seasonal: bool = False,
) -> CompromiseLabel:
    r = (reason or "").lower()
    if seasonal or "winter" in r or "season" in r:
        return CompromiseLabel.SEASONALLY_CONDITIONAL
    if "payload" in r or "passenger" in r:
        return CompromiseLabel.PAYLOAD_LIMITED
    if distance_nm >= 2200 and ("tech" in r or "stop" in r or "nonstop" in r):
        return CompromiseLabel.TECH_STOP_LIKELY
    if _SOFT_BAND_RE.search(r) or "in-band" in r:
        return CompromiseLabel.OUT_OF_BAND_COMPROMISE
    if "margin" in r or "range" in r:
        return CompromiseLabel.TECH_STOP_LIKELY
    return CompromiseLabel.VIABLE_WITH_COMPROMISES


def feasibility_for_soft_elimination(
    model: str,
    reason: str,
    *,
    label: Optional[CompromiseLabel] = None,
    distance_nm: float = 0.0,
    seasonal: bool = False,
) -> FeasibilityResult:
    """Keep aircraft in play with explicit compromise — not ``feasible=False`` silence."""
    verdict = label or compromise_label_for_reason(
        reason, distance_nm=distance_nm, seasonal=seasonal
    )
    return FeasibilityResult(
        feasible=True,
        operational_risk_level="compromise",
        notes=[verdict.value, reason],
        elimination_reasons=[],
    )


def apply_conditional_elimination_map(
    feasibility_map: Dict[str, FeasibilityResult],
    *,
    eliminated: List[str],
    reasons: Dict[str, str],
    distance_nm: float = 0.0,
    elimination_kind: str = "band",
    seasonal: bool = False,
) -> tuple[List[str], List[str], Dict[str, str]]:
    """
    Split eliminated models into hard (removed) vs soft (downgraded in map).

    Returns ``(hard_eliminated, soft_downgraded, compromise_labels)``.
    """
    hard: List[str] = []
    soft: List[str] = []
    labels: Dict[str, str] = {}
    for model in eliminated:
        reason = reasons.get(model, "")
        if elimination_severity(
            reason, distance_nm=distance_nm, elimination_kind=elimination_kind
        ) == "hard":
            hard.append(model)
            feasibility_map[model] = FeasibilityResult(
                feasible=False,
                elimination_reasons=[reason or f"{elimination_kind} elimination"],
                operational_risk_level="eliminated",
            )
        else:
            soft.append(model)
            lbl = compromise_label_for_reason(
                reason, distance_nm=distance_nm, seasonal=seasonal
            )
            labels[model] = lbl.value
            feasibility_map[model] = feasibility_for_soft_elimination(
                model, reason, label=lbl, distance_nm=distance_nm, seasonal=seasonal
            )
    return hard, soft, labels
