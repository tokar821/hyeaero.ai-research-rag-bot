"""Phase 33 — Valuation accuracy audit (final answer only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from tests.response_quality._text_extract import extract_aircraft_like_tokens, extract_year, normalize


@dataclass
class ValuationAccuracyAudit:
    score: float
    failures: List[str]


def _resolve(token: str) -> str:
    from services.aircraft.aircraft_authority_service import resolve_aircraft_alias

    return resolve_aircraft_alias(token) or token


def _query_target(query: str) -> Tuple[Optional[int], Optional[str]]:
    year = extract_year(query)
    tokens = extract_aircraft_like_tokens(query)
    model = _resolve(tokens[0]) if tokens else None
    return year, model


def audit_valuation_accuracy(*, query: str, answer: str) -> ValuationAccuracyAudit:
    failures: List[str] = []
    q_year, q_model = _query_target(query)

    a_year = extract_year(answer)
    a_tokens = [_resolve(t) for t in extract_aircraft_like_tokens(answer)]
    a_tokens = [t for t in a_tokens if t]

    if q_model and a_tokens:
        # Cross-model valuation: answer clearly values a different model than requested.
        if q_model not in a_tokens and any(t != q_model for t in a_tokens[:3]):
            failures.append("CROSS_MODEL_VALUATION")

    if q_year and a_year and q_year != a_year:
        failures.append("WRONG_YEAR_VALUATION")

    # Unknown valuation source: includes a dollar figure but does not mention verified/sources/listings.
    t = normalize(answer)
    has_price = "$" in (answer or "")
    has_source = any(k in t for k in ("verified", "listing", "market", "comps", "broker", "catalog", "insufficient verified data"))
    if has_price and not has_source:
        failures.append("UNKNOWN_VALUATION_SOURCE")

    score = 100.0
    if "CROSS_MODEL_VALUATION" in failures:
        score = 0.0  # stop condition in Phase 33
    if "WRONG_YEAR_VALUATION" in failures:
        score -= 40
    if "UNKNOWN_VALUATION_SOURCE" in failures:
        score -= 20
    score = max(0.0, round(score, 2))
    return ValuationAccuracyAudit(score=score, failures=sorted(set(failures)))

