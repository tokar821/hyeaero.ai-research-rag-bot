"""Phase 33 — Comparison answer quality audit (final answer only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from tests.response_quality._text_extract import normalize


@dataclass
class ComparisonQualityAudit:
    score: float
    failures: List[str]


def audit_comparison_quality(*, answer: str) -> ComparisonQualityAudit:
    t = normalize(answer)
    failures: List[str] = []

    has_range = any(k in t for k in ("range", "nm", "nautical"))
    has_cabin = any(k in t for k in ("cabin", "seats", "pax", "layout", "berth"))
    has_cost = any(k in t for k in ("operating cost", "hourly", "cost", "doc", "fuel burn"))

    verdict_tokens = ("verdict", "bottom line", "recommend", "i would pick", "best for", "choose")
    has_verdict = any(k in t for k in verdict_tokens)

    missing = []
    if not has_range:
        missing.append("range")
    if not has_cabin:
        missing.append("cabin")
    if not has_cost:
        missing.append("operating_cost")
    if missing:
        failures.append("COMPARISON_INCOMPLETE")
    if not has_verdict:
        failures.append("COMPARISON_NO_VERDICT")

    score = 100.0
    if "COMPARISON_INCOMPLETE" in failures:
        score -= 40
    if "COMPARISON_NO_VERDICT" in failures:
        score -= 60
    score = max(0.0, round(score, 2))
    return ComparisonQualityAudit(score=score, failures=sorted(set(failures)))

