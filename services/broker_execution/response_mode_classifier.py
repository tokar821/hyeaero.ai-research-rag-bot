"""
Phase 56.5 — response mode classification (formatting only).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, Optional

from services.broker_execution.broker_execution_category import (
    BrokerExecutionCategory,
    classify_broker_execution_category,
    comparison_requests_recommendation,
)

_ANALYSIS_RE = re.compile(
    r"(?is)\b(?:analyze|analysis|deep\s+dive|explain\s+in\s+detail|walk\s+me\s+through|"
    r"comprehensive|full\s+breakdown)\b"
)
_SALE_STATUS_RE = re.compile(
    r"(?is)\b(?:for\s+sale|sale\s+status|aircraft\s+status|listed|on\s+the\s+market)\b"
)


class ResponseMode(str, Enum):
    FACT_ONLY = "FACT_ONLY"
    COMPARISON = "COMPARISON"
    LISTING = "LISTING"
    MISSION = "MISSION"
    ANALYSIS = "ANALYSIS"


def classify_response_mode(
    query: str,
    *,
    data_used: Optional[dict] = None,
) -> ResponseMode:
    q = (query or "").strip()
    ql = q.lower()
    du = data_used if isinstance(data_used, dict) else {}

    if _ANALYSIS_RE.search(q) and not re.search(r"(?is)\bwho\s+owns\b", ql):
        return ResponseMode.ANALYSIS

    cat = classify_broker_execution_category(q, data_used=du)

    try:
        from services.broker_execution.tail_depth_mode import TailDepthMode, classify_tail_depth_mode

        tdepth, _ = classify_tail_depth_mode(q)
        if tdepth in (TailDepthMode.OWNER, TailDepthMode.SALE_STATUS):
            return ResponseMode.FACT_ONLY
    except Exception:
        pass

    if cat == BrokerExecutionCategory.TAIL_OWNERSHIP:
        return ResponseMode.FACT_ONLY

    try:
        from rag.aviation_tail import primary_registration_from_query

        if _SALE_STATUS_RE.search(q) and primary_registration_from_query(q):
            return ResponseMode.FACT_ONLY
    except Exception:
        if _SALE_STATUS_RE.search(q) and re.search(r"\bN(?=[A-Z0-9]*\d)[A-Z0-9]{2,6}\b", q.upper()):
            return ResponseMode.FACT_ONLY

    if _SALE_STATUS_RE.search(q) and (
        du.get("tail_registration") or du.get("tail_investigation_dispatch")
    ):
        return ResponseMode.FACT_ONLY

    if cat == BrokerExecutionCategory.COMPARISON and not comparison_requests_recommendation(q):
        return ResponseMode.COMPARISON

    if cat == BrokerExecutionCategory.LISTING:
        return ResponseMode.LISTING

    if cat == BrokerExecutionCategory.MISSION:
        return ResponseMode.MISSION

    audit = du.get("listing_parse_audit") or {}
    if isinstance(audit, dict) and audit.get("parse_success"):
        return ResponseMode.LISTING

    if re.search(r"(?is)\b(?:vs\.?|versus)\b", q):
        return ResponseMode.COMPARISON

    return ResponseMode.ANALYSIS


IDEAL_TOKENS_BY_MODE: Dict[ResponseMode, int] = {
    ResponseMode.FACT_ONLY: 60,
    ResponseMode.LISTING: 280,
    ResponseMode.COMPARISON: 320,
    ResponseMode.MISSION: 300,
    ResponseMode.ANALYSIS: 700,
}

MAX_TOKENS_BY_MODE: Dict[ResponseMode, int] = {
    ResponseMode.FACT_ONLY: 90,
    ResponseMode.LISTING: 420,
    ResponseMode.COMPARISON: 450,
    ResponseMode.MISSION: 420,
    ResponseMode.ANALYSIS: 900,
}


__all__ = [
    "IDEAL_TOKENS_BY_MODE",
    "MAX_TOKENS_BY_MODE",
    "ResponseMode",
    "classify_response_mode",
]
