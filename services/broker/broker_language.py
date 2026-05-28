"""
Broker language enforcement — anti-brochure copy and refusal phrasing.
"""

from __future__ import annotations

import re
from typing import List, Tuple

FORBIDDEN_BROCHURE_RE = re.compile(
    r"\b(?:"
    r"supports\s+(?:the\s+)?mission|"
    r"excellent\s+range|"
    r"ideal\s+solution|"
    r"ideal\s+aircraft|"
    r"great\s+choice|"
    r"luxurious|"
    r"luxury\s+experience|"
    r"tailored\s+experience|"
    r"enhance\s+flexibility|"
    r"enhances\s+flexibility|"
    r"comfortable\s+journey|"
    r"best[- ]in[- ]class|"
    r"world[- ]class|"
    r"unparalleled|"
    r"worth\s+considering|"
    r"game[- ]changing"
    r")\b",
    re.I,
)

_PREFERRED_SUBSTITUTIONS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsupports\s+(?:the\s+)?mission\b", re.I), "operationally realistic for this leg"),
    (re.compile(r"\bexcellent\s+range\b", re.I), "practical range with margin"),
    (re.compile(r"\bideal\s+aircraft\b", re.I), "credible option in this band"),
    (re.compile(r"\bgreat\s+choice\b", re.I), "viable with stated tradeoffs"),
    (re.compile(r"\bluxurious\b", re.I), "cabin-focused"),
    (re.compile(r"\btailored\s+experience\b", re.I), "mission-specific tradeoffs"),
    (re.compile(r"\benhance(?:s)?\s+flexibility\b", re.I), "adds dispatch optionality"),
    (re.compile(r"\bcomfortable\s+journey\b", re.I), "acceptable cabin for stage length"),
)

REFUSAL_TEMPLATES: Tuple[str, ...] = (
    "I would not advise this with confidence based on verified performance data.",
    "I don't have enough verified field-performance data to position this as a reliable recommendation.",
    "I can't position this aircraft as operationally credible without verified performance data for this mission.",
    "I would not position this aircraft as a credible nonstop solution.",
    "This mission exceeds realistic payload-range margins.",
    "You are combining conflicting operational requirements into one platform.",
    "Stage length is corridor-classified only — I won't treat geodesic distance as verified nonstop capability.",
)

ROBOTIC_REFUSAL_RE = re.compile(
    r"I\s+don'?t\s+have\s+reliable\s+data\s+for\s+this\.?",
    re.I,
)


def broker_refusal_message(*, context: str = "general") -> str:
    """Bounded broker guidance (legacy name — not a hard refusal)."""
    from services.broker.graceful_degradation import broker_degraded_message

    return broker_degraded_message(context=context)


def sanitize_broker_language(text: str) -> str:
    """Strip brochure phrasing; apply operational substitutions."""
    from services.broker.graceful_degradation import transform_refusal_prose

    if not (text or "").strip():
        return text
    out = ROBOTIC_REFUSAL_RE.sub(broker_refusal_message(), text)
    for pat, repl in _PREFERRED_SUBSTITUTIONS:
        out = pat.sub(repl, out)
    out = FORBIDDEN_BROCHURE_RE.sub("", out)
    out = transform_refusal_prose(out)
    out = re.sub(r" +", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def apply_broker_language_rules(text: str) -> Tuple[str, List[str]]:
    """Return sanitized text and list of violations found."""
    violations: List[str] = []
    if not text:
        return text, violations
    for m in FORBIDDEN_BROCHURE_RE.finditer(text):
        violations.append(m.group(0))
    if ROBOTIC_REFUSAL_RE.search(text):
        violations.append("robotic_refusal_phrase")
    return sanitize_broker_language(text), violations
