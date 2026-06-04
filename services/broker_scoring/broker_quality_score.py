"""
Phase 50 — per-answer broker quality score (0–100).

Measurement only: attaches ``data_used["broker_quality_score"]`` without altering prose.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from services.client_context.recommendation_consistency import _tier_musd

_FORBIDDEN_PHRASES: Tuple[str, ...] = (
    "buyer leverage",
    "seller leverage",
    "inventory pressure",
    "deterministic execution",
    "verified catalog",
    "insufficient verified data",
    "mission kernel",
    "catalog authority",
    "insufficient_data",
)

_BUDGET_QUERY_RE = re.compile(
    r"(?is)\b(?:can\s+i\s+buy|only\s+have|budget|for\s+\$|under\s+\$|have\s+\$)\b"
)
_MISSION_QUERY_RE = re.compile(
    r"(?is)\b(?:nonstop|passengers?|pax|coast.?to.?coast|→| to )\b"
)
_REALITY_MARKERS = (
    "no.",
    "not realistically",
    "far below",
    "does not trade",
    "below typical",
    "not realistic",
    "cannot",
    "conflict",
    "exceed",
    "beyond",
)
_RECOMMENDATION_RE = re.compile(
    r"(?is)\b(?:i'd focus on|if i were buying|i would buy|i would choose|"
    r"i'd choose|i would lean|i'd lean|lean toward|choose the)\b"
)
_RATIONALE_RE = re.compile(r"(?is)\b(?:because|since|for your|given your|where)\b")
_SPEC_DUMP_RE = re.compile(r"(?is)\b(?:\d{3,4}\s*nm|practical range|max speed|mtow)\b")
_BUDGET_IN_QUERY_RE = re.compile(
    r"(?is)\$\s*(?P<amt>\d+(?:\.\d+)?)\s*(?P<unit>m|mm|million|mil|k)\b"
)


def _parse_budget_musd(query: str) -> Optional[float]:
    m = _BUDGET_IN_QUERY_RE.search(query or "")
    if not m:
        return None
    try:
        val = float(m.group("amt"))
    except (TypeError, ValueError):
        return None
    unit = (m.group("unit") or "m").lower()
    if unit == "k":
        return val / 1000.0
    return val if val < 1000 else val


def _extract_primary(data_used: Dict[str, Any], answer: str) -> Optional[str]:
    rec = data_used.get("executive_recommendation") or {}
    if isinstance(rec, dict) and rec.get("primary_recommendation"):
        return str(rec["primary_recommendation"]).strip()
    m = re.search(
        r"(?is)(?:i'd focus on|if i were buying(?: today)?,?\s*i'd focus on)\s+(?:the\s+)?([^.\n]+)",
        answer or "",
    )
    return m.group(1).strip() if m else None


def _score_budget_realism(answer: str, query: str, data_used: Dict[str, Any]) -> float:
    if not _BUDGET_QUERY_RE.search(query or ""):
        return 25.0

    score = 25.0
    low = (answer or "").lower()
    first = (answer or "").split("\n\n")[0].lower()

    if data_used.get("acquisition_budget_infeasible"):
        if not any(m in first for m in _REALITY_MARKERS):
            score -= 12.0
    elif not any(m in first for m in _REALITY_MARKERS + ("at $", "with $")):
        score -= 8.0

    if re.search(r"(?is)\b(?:can be plausible|stretch case|motivated seller)\b", low):
        score -= 10.0

    budget = _parse_budget_musd(query) or (data_used.get("client_context") or {}).get(
        "remembered_budget_musd"
    )
    if budget is not None:
        primary = _extract_primary(data_used, answer)
        if primary:
            try:
                if _tier_musd(primary) > float(budget) * 1.2:
                    if not re.search(r"(?is)(?:would not|not lead|above the|budget cap)", answer):
                        score -= 15.0
            except (TypeError, ValueError):
                pass

    if re.search(r"(?is)^\s*(?:before treating|i would verify|verify:)", first):
        score -= 8.0

    return max(0.0, min(25.0, score))


def _score_mission_realism(answer: str, query: str, data_used: Dict[str, Any]) -> float:
    if not _MISSION_QUERY_RE.search(query or ""):
        return 25.0

    score = 25.0
    low = (answer or "").lower()

    ultra_long = re.search(r"(?is)\b(?:tokyo|singapore|sydney|beijing|hong kong)\b", query or "")
    budget = _parse_budget_musd(query)
    if ultra_long and budget is not None and budget < 25:
        conflict_markers = (
            "conflict",
            "not realistic",
            "cannot",
            "exceed",
            "beyond",
            "not nonstop",
            "refuel",
            "does not close",
            "mission",
            "band",
        )
        if not any(m in low for m in conflict_markers):
            score -= 15.0

    if re.search(r"(?is)\b\d+\s+passengers?\b", query or "") and budget is not None and budget < 8:
        if not any(m in low for m in _REALITY_MARKERS):
            score -= 10.0

    br = data_used.get("broker_reasoning") or {}
    if isinstance(br, dict) and br.get("mission_conflict"):
        if "conflict" not in low and "not realistic" not in low:
            score -= 8.0

    return max(0.0, min(25.0, score))


def _score_recommendation_consistency(answer: str, query: str, data_used: Dict[str, Any]) -> float:
    audit = data_used.get("recommendation_consistency_audit_v2") or {}
    if isinstance(audit, dict):
        if audit.get("recommendation_drift"):
            return max(0.0, 20.0 - 12.0)
        if audit.get("drift_severity") == "HIGH":
            return 5.0
    return 20.0


def _score_decision_clarity(answer: str, query: str, data_used: Dict[str, Any]) -> float:
    low = (answer or "").lower()
    is_comparison = " vs " in (query or "").lower() or " versus " in (query or "").lower()
    is_decision = bool(
        _RECOMMENDATION_RE.search(answer or "")
        or data_used.get("executive_broker_layer_applied")
        or is_comparison
        or re.search(r"(?is)\b(?:what should i buy|would you buy|buy now|wait)\b", query or "")
    )

    if not is_decision:
        return 15.0

    score = 15.0
    if not _RECOMMENDATION_RE.search(answer or ""):
        if not data_used.get("acquisition_budget_infeasible"):
            score -= 10.0

    if is_comparison and not re.search(
        r"(?is)\b(?:i would choose|i'd choose|i would lean|i'd lean|choose the)\b", answer or ""
    ):
        score -= 8.0

    if is_decision and not _RATIONALE_RE.search(answer or ""):
        if not data_used.get("acquisition_budget_infeasible"):
            score -= 4.0

    if re.search(r"(?is)^\s*(?:gulfstream|g\d{3}|citation|falcon)\s+(?:has|offers|features)\s+\d", low):
        score -= 6.0

    return max(0.0, min(15.0, score))


def _score_natural_language(answer: str, query: str, data_used: Dict[str, Any]) -> float:
    score = 15.0
    low = (answer or "").lower()

    for phrase in _FORBIDDEN_PHRASES:
        if phrase in low:
            score -= 3.0

    spec_hits = len(_SPEC_DUMP_RE.findall(answer or ""))
    if spec_hits >= 4 and not _RECOMMENDATION_RE.search(answer or ""):
        score -= 5.0

    bullets = len(re.findall(r"(?m)^\s*[•\-]\s+", answer or ""))
    if bullets >= 10:
        score -= 4.0

    return max(0.0, min(15.0, score))


def score_broker_answer(
    answer: str,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return 0–100 broker judgment score with dimensional breakdown."""
    du = data_used if isinstance(data_used, dict) else {}
    text = (answer or "").strip()

    breakdown = {
        "budget_realism": round(_score_budget_realism(text, query, du), 2),
        "mission_realism": round(_score_mission_realism(text, query, du), 2),
        "recommendation_consistency": round(
            _score_recommendation_consistency(text, query, du), 2
        ),
        "decision_clarity": round(_score_decision_clarity(text, query, du), 2),
        "natural_broker_language": round(_score_natural_language(text, query, du), 2),
    }
    total = round(sum(breakdown.values()), 2)

    return {
        "total": total,
        "breakdown": breakdown,
        "grade": _grade(total),
    }


def _grade(total: float) -> str:
    if total >= 90:
        return "A"
    if total >= 80:
        return "B"
    if total >= 70:
        return "C"
    if total >= 60:
        return "D"
    return "F"


def attach_broker_quality_score(
    answer: str,
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score answer and persist on ``data_used`` (no prose mutation)."""
    du = data_used if isinstance(data_used, dict) else {}
    from services.broker_scoring.recommendation_consistency_audit_v2 import (
        audit_recommendation_consistency_v2,
    )

    audit_recommendation_consistency_v2(answer, query=query, data_used=du)
    result = score_broker_answer(answer, query=query, data_used=du)
    du["broker_quality_score"] = result
    return result


__all__ = ["attach_broker_quality_score", "score_broker_answer"]
