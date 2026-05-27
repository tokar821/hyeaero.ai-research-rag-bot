"""
Cross-response repetition detection for QA runs.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

# Stock patterns that make the assistant sound templated (aligned with phrase_repetition_guard).
GLOBAL_STOCK_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("starts the conversation", re.compile(r"\bstarts\s+the\s+conversation\b", re.I)),
    ("worth revisiting", re.compile(r"\bworth\s+revisiting\b", re.I)),
    ("on my list", re.compile(r"\bon\s+my\s+list\b", re.I)),
    ("if the trip stays in this profile", re.compile(r"\bif\s+the\s+trip\s+stays\s+in\s+this\s+profile\b", re.I)),
    ("most balanced operator", re.compile(r"\bmost\s+balanced\s+operator\b", re.I)),
    ("my recommendation is", re.compile(r"\bmy\s+recommendation\s+is\b", re.I)),
    ("mission summary", re.compile(r"\bmission\s+summary\b", re.I)),
    ("best fit aircraft", re.compile(r"\bbest\s+fit\s+aircraft\b", re.I)),
    ("operational tradeoffs:", re.compile(r"\boperational\s+tradeoffs\s*:", re.I)),
    ("great range and comfort", re.compile(r"\bgreat\s+range\s+and\s+comfort\b", re.I)),
    ("excellent performance", re.compile(r"\bexcellent\s+performance\b", re.I)),
    ("good operating economics", re.compile(r"\bgood\s+operating\s+economics\b", re.I)),
)


@dataclass
class RepetitionReport:
    """Repetition metrics for one answer or a batch."""

    phrase_hits: Dict[str, int] = field(default_factory=dict)
    repetition_score: float = 0.0  # 0..1 higher = worse
    overused_phrases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "phrase_hits": dict(self.phrase_hits),
            "repetition_score": round(self.repetition_score, 4),
            "overused_phrases": list(self.overused_phrases),
        }


def scan_phrase_hits(text: str) -> Dict[str, int]:
    hits: Dict[str, int] = {}
    for label, pat in GLOBAL_STOCK_PATTERNS:
        n = len(pat.findall(text or ""))
        if n:
            hits[label] = n
    return hits


def score_answer_repetition(
    answer: str,
    *,
    forbidden_extra: Sequence[str] = (),
) -> RepetitionReport:
    """Score a single answer for templated / banned phrasing."""
    hits = scan_phrase_hits(answer)
    for phrase in forbidden_extra:
        p = (phrase or "").strip()
        if not p:
            continue
        n = len(re.findall(re.escape(p), answer or "", flags=re.I))
        if n:
            hits[p] = hits.get(p, 0) + n

    total_hits = sum(hits.values())
    unique = len(hits)
    # Saturating score: one hit is noticeable; many hits is severe.
    repetition_score = min(1.0, 0.25 * unique + 0.12 * total_hits)
    overused = [k for k, v in hits.items() if v >= 1]
    return RepetitionReport(
        phrase_hits=hits,
        repetition_score=repetition_score,
        overused_phrases=overused,
    )


@dataclass
class BatchRepetitionTracker:
    """Track phrase frequency across a QA suite run."""

    global_counts: Counter[str] = field(default_factory=Counter)
    answers_seen: int = 0

    def observe(self, answer: str, *, forbidden_extra: Sequence[str] = ()) -> RepetitionReport:
        report = score_answer_repetition(answer, forbidden_extra=forbidden_extra)
        self.answers_seen += 1
        for k, v in report.phrase_hits.items():
            self.global_counts[k] += v
        return report

    def suite_repetition_score(self) -> float:
        """0..1 — high if same stock phrases appear across many cases."""
        if self.answers_seen <= 1:
            return 0.0
        # Phrases appearing in >30% of answers are a structural problem.
        bad = 0
        for _phrase, count in self.global_counts.items():
            if count >= max(2, int(self.answers_seen * 0.3)):
                bad += 1
        return min(1.0, bad * 0.2 + sum(self.global_counts.values()) / (self.answers_seen * 4))

    def to_dict(self) -> Dict[str, object]:
        return {
            "answers_seen": self.answers_seen,
            "global_phrase_counts": dict(self.global_counts.most_common(20)),
            "suite_repetition_score": round(self.suite_repetition_score(), 4),
        }
