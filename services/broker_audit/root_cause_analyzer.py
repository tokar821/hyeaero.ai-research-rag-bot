"""
Phase 51 — root-cause classification for broker-quality failures.

Operates on existing metadata and optional ground-truth expectations.
Does not modify answers or routing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from services.broker_audit.broker_trace import BrokerTrace, build_broker_trace


class FailureCause(str, Enum):
    AUTHORITY_ERROR = "AUTHORITY_ERROR"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    MARKET_DATA_ERROR = "MARKET_DATA_ERROR"
    VALUATION_ERROR = "VALUATION_ERROR"
    RECOMMENDATION_ERROR = "RECOMMENDATION_ERROR"
    SYNTHESIS_ERROR = "SYNTHESIS_ERROR"
    UNKNOWN = "UNKNOWN"


@dataclass
class RootCauseResult:
    cause: FailureCause
    confidence: float
    evidence: List[str]
    trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cause": self.cause.value,
            "confidence": round(self.confidence, 3),
            "evidence": list(self.evidence),
            "trace": dict(self.trace),
        }


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def _model_in_text(model: str, text: str) -> bool:
    if not model or not text:
        return False
    return model.lower().replace(" ", "") in text.lower().replace(" ", "")


def analyze_root_cause(
    *,
    query: str,
    answer: str,
    data_used: Optional[Dict[str, Any]] = None,
    expected_authority: Optional[str] = None,
    expected_primary: Optional[str] = None,
    forbidden_primary: Optional[str] = None,
    expect_infeasible: bool = False,
    expect_listing_skepticism: bool = False,
    failure_reasons: Optional[List[str]] = None,
) -> RootCauseResult:
    """
    Classify why an answer failed certification or ground-truth checks.
    """
    du = data_used if isinstance(data_used, dict) else {}
    trace = build_broker_trace(answer, query=query, data_used=du)
    evidence: List[str] = []
    reasons = [r.lower() for r in (failure_reasons or [])]

    # --- Explicit expectation mismatches ---
    if expect_infeasible:
        first = (answer or "").split("\n\n")[0].lower()
        if not trace.acquisition_infeasible and not first.startswith(("no.", "not realistically")):
            if forbidden_primary and _model_in_text(forbidden_primary, answer):
                evidence.append(f"endorsed forbidden model {forbidden_primary} on infeasible budget")
                return RootCauseResult(
                    FailureCause.RECOMMENDATION_ERROR,
                    0.9,
                    evidence,
                    trace.to_dict(),
                )
            evidence.append("missing budget reality rejection")
            return RootCauseResult(
                FailureCause.RECOMMENDATION_ERROR,
                0.85,
                evidence,
                trace.to_dict(),
            )

    if expected_authority:
        auth = _normalize(trace.authority_selected)
        exp = _normalize(expected_authority)
        if exp and exp not in auth and auth != exp:
            evidence.append(f"authority={trace.authority_selected} expected={expected_authority}")
            return RootCauseResult(
                FailureCause.AUTHORITY_ERROR,
                0.88,
                evidence,
                trace.to_dict(),
            )

    if expected_primary:
        prim = trace.executive_primary or ""
        if not _model_in_text(expected_primary, prim) and not _model_in_text(expected_primary, answer):
            evidence.append(f"primary={prim or 'none'} expected={expected_primary}")
            if not trace.executive_primary and not du.get("executive_broker_layer_applied"):
                return RootCauseResult(
                    FailureCause.RECOMMENDATION_ERROR,
                    0.82,
                    evidence,
                    trace.to_dict(),
                )
            return RootCauseResult(
                FailureCause.RECOMMENDATION_ERROR,
                0.85,
                evidence,
                trace.to_dict(),
            )

    if forbidden_primary and _model_in_text(forbidden_primary, answer):
        if trace.acquisition_infeasible or "no." in answer[:80].lower():
            pass
        else:
            evidence.append(f"forbidden endorsement of {forbidden_primary}")
            return RootCauseResult(
                FailureCause.RECOMMENDATION_ERROR,
                0.9,
                evidence,
                trace.to_dict(),
            )

    if expect_listing_skepticism:
        low = answer.lower()
        if not any(
            m in low
            for m in ("unusual", "below", "verify", "bargain", "skeptical", "materially below")
        ):
            if "plausible" in low or "can trade in-band" in low:
                evidence.append("listing treated as plausible without skepticism")
                return RootCauseResult(
                    FailureCause.MARKET_DATA_ERROR,
                    0.86,
                    evidence,
                    trace.to_dict(),
                )
            return RootCauseResult(
                FailureCause.MARKET_DATA_ERROR,
                0.75,
                evidence,
                trace.to_dict(),
            )

    # --- Reason-string heuristics ---
    joined = " ".join(reasons)
    if "insufficient" in joined and "comparison" in joined:
        if not trace.retrieval_sources or "comparison" not in " ".join(trace.retrieval_sources).lower():
            evidence.append("comparison dispatch or retrieval did not produce structured data")
            return RootCauseResult(
                FailureCause.RETRIEVAL_ERROR,
                0.8,
                evidence,
                trace.to_dict(),
            )

    if "drift" in joined or "recommendation_drift" in joined:
        evidence.append("multi-turn recommendation drift")
        return RootCauseResult(
            FailureCause.RECOMMENDATION_ERROR,
            0.84,
            evidence,
            trace.to_dict(),
        )

    if "mission" in joined and "conflict" in joined:
        evidence.append("mission/budget conflict not surfaced")
        return RootCauseResult(
            FailureCause.RECOMMENDATION_ERROR,
            0.8,
            evidence,
            trace.to_dict(),
        )

    if "forbidden" in joined or "humanization" in joined or "buyer leverage" in joined:
        evidence.append("late synthesis / humanization leak")
        return RootCauseResult(
            FailureCause.SYNTHESIS_ERROR,
            0.78,
            evidence,
            trace.to_dict(),
        )

    if "valuation" in joined or "worth" in joined:
        evidence.append("valuation interpretation mismatch")
        return RootCauseResult(
            FailureCause.VALUATION_ERROR,
            0.72,
            evidence,
            trace.to_dict(),
        )

    if "comparison" in joined and "conclusion" in joined:
        if "comparison" not in _normalize(trace.authority_selected):
            return RootCauseResult(
                FailureCause.AUTHORITY_ERROR,
                0.75,
                evidence + ["comparison expected but authority mismatch"],
                trace.to_dict(),
            )
        return RootCauseResult(
            FailureCause.SYNTHESIS_ERROR,
            0.7,
            evidence + ["comparison authority ok but missing broker conclusion"],
            trace.to_dict(),
        )

    if trace.broker_quality_score is not None and trace.broker_quality_score < 65:
        if not trace.executive_primary:
            return RootCauseResult(
                FailureCause.RECOMMENDATION_ERROR,
                0.65,
                evidence + [f"low quality score {trace.broker_quality_score}"],
                trace.to_dict(),
            )

    if not trace.retrieval_sources and len(answer) < 60:
        return RootCauseResult(
            FailureCause.RETRIEVAL_ERROR,
            0.6,
            evidence + ["empty or weak retrieval path"],
            trace.to_dict(),
        )

    return RootCauseResult(
        FailureCause.UNKNOWN,
        0.4,
        evidence + (reasons[:3] if reasons else ["no specific classifier match"]),
        trace.to_dict(),
    )


__all__ = ["FailureCause", "RootCauseResult", "analyze_root_cause"]
