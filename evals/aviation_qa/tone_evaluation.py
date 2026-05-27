"""
Tone evaluation — broker/operator vs AI recommendation engine.
"""

from __future__ import annotations

import re
from typing import Tuple

_BROKER_SIGNALS = re.compile(
    r"\b("
    r"in\s+practice|realistically|most\s+operators|on\s+paper|"
    r"dispatch|crew|maintenance\s+downtime|NBAA|reserve|payload|baggage|"
    r"hot[- ]and[- ]high|runway|tech[- ]stop|westbound|headwind|"
    r"wouldn'?t\s+(?:try|force)|steer\s+clear|brochure|margin|"
    r"ownership\s+friction|hourly\s+cost|liquidity|"
    r"depends\s+how\s+aggressively|typically|usually"
    r")\b",
    re.I,
)

_AI_ENGINE_SIGNALS = re.compile(
    r"\b("
    r"mission\s+summary|mission\s+type|best\s+fit\s+aircraft|mission\s+score|"
    r"confidence\s*:|operational\s+tradeoffs\s*:|alternatives\s+scored|"
    r"here\s+are\s+(?:\d+|a\s+few)\s+(?:realistic\s+)?fits|"
    r"assuming\s+6[\s–-]8\s+passengers|"
    r"excellent\s+performance|great\s+range|good\s+operating\s+economics|"
    r"my\s+recommendation\s+is\s+the|starts\s+the\s+conversation|"
    r"on\s+my\s+list|if\s+the\s+trip\s+stays\s+in\s+this\s+profile"
    r")\b",
    re.I,
)

_FAKE_CERTAINTY = re.compile(
    r"\b("
    r"definitely|without\s+doubt|guaranteed|always\s+nonstop|"
    r"will\s+make\s+it\s+nonstop|perfect\s+for\s+this\s+mission|"
    r"the\s+clear\s+winner|no\s+question"
    r")\b",
    re.I,
)

_CONFIDENT_UNCERTAINTY = re.compile(
    r"\b("
    r"in\s+practice|realistically|most\s+operators|on\s+paper|"
    r"typically|usually|depends|assuming\s+typical|"
    r"based\s+on\s+what\s+you'?ve\s+described"
    r")\b",
    re.I,
)

_BROCHURE_LANGUAGE = re.compile(
    r"\b("
    r"excellent\s+performance|outstanding\s+range|best[- ]in[- ]class|"
    r"unparalleled|world[- ]class|industry[- ]leading|"
    r"great\s+range\s+and\s+comfort|superior\s+comfort|"
    r"exceptional\s+capabilities|cutting[- ]edge"
    r")\b",
    re.I,
)


def score_tone(answer: str) -> Tuple[float, float, float, float]:
    """
    Returns ``(humanness, broker_tone, fake_confidence_risk, brochure_risk)`` each 0..1.
    """
    text = answer or ""
    if not text.strip():
        return 0.0, 0.0, 0.5, 0.5

    broker_hits = len(_BROKER_SIGNALS.findall(text))
    ai_hits = len(_AI_ENGINE_SIGNALS.findall(text))
    uncertainty_hits = len(_CONFIDENT_UNCERTAINTY.findall(text))
    fake_hits = len(_FAKE_CERTAINTY.findall(text))
    brochure_hits = len(_BROCHURE_LANGUAGE.findall(text))

    words = max(1, len(text.split()))
    broker_norm = min(1.0, broker_hits / max(2, words / 80))
    ai_norm = min(1.0, ai_hits / max(1, words / 100))

    humanness = _clamp(broker_norm * 0.7 + uncertainty_hits * 0.08 - ai_norm * 0.5 + 0.25)
    broker_tone = _clamp(broker_norm - ai_norm * 0.4 + 0.35)
    fake_risk = _clamp(fake_hits * 0.35 - uncertainty_hits * 0.08)
    brochure_risk = _clamp(brochure_hits * 0.4)

    return humanness, broker_tone, fake_risk, brochure_risk


def score_operational_depth(answer: str) -> float:
    """0..1 — presence of real aviation thinking."""
    if not (answer or "").strip():
        return 0.0
    depth = 0.35
    if re.search(r"\b(NBAA|reserve|payload|baggage|westbound|headwind|runway|hot)\b", answer, re.I):
        depth += 0.25
    if re.search(r"\b(tradeoff|margin|tech[- ]stop|dispatch|downtime|cabin\s+altitude)\b", answer, re.I):
        depth += 0.2
    if re.search(r"\b(avoid|wouldn'?t|brochure|on\s+paper|in\s+practice)\b", answer, re.I):
        depth += 0.15
    return _clamp(depth)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))
