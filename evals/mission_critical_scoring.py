"""
Heuristic scoring for Mission Understanding Critical Test Suite.

Focus: operational inference, synthesis, continuity — NOT aircraft accuracy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


_COLLAPSE_RE = re.compile(
    r"I couldn't get a response|I don't have enough verified|"
    r"I don't have reliable data|try again later",
    re.I,
)
_BROCHURE_RE = re.compile(
    r"\b(?:best-in-class|world-class|unparalleled|luxury redefined|game-changing)\b",
    re.I,
)
_ROBOTIC_RE = re.compile(
    r"\b(?:mission profile|worth considering|if priorities shift|clearest fit)\b",
    re.I,
)

_OP_TERMS = re.compile(
    r"\b(?:dispatch|runway|nonstop|fuel stop|tech stop|operating cost|ownership|"
    r"fractional|charter|reposition|margin|payload|winter|westbound|mountain|"
    r"short field|corrosion|island|portfolio|multi[- ]aircraft|incompatible|"
    r"compromise|class band|utilization|hours per year|economics|liquidity|"
    r"availability|reliability|access|flexibility)\b",
    re.I,
)


@dataclass
class ScenarioScore:
    scenario_id: int
    category: str
    passed: bool
    score: float
    issues: List[str] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "category": self.category,
            "passed": self.passed,
            "score": round(self.score, 3),
            "issues": list(self.issues),
            "signals": list(self.signals),
        }


def _text_blob(parts: Sequence[str]) -> str:
    return "\n".join(p for p in parts if p).strip()


def _has_any(text: str, patterns: Sequence[str]) -> bool:
    low = text.lower()
    return any(p.lower() in low for p in patterns)


def score_response(
    *,
    scenario_id: int,
    category: str,
    answer: str,
    expectations: Optional[List[str]] = None,
    packet: Optional[Dict[str, Any]] = None,
    turn2_answer: Optional[str] = None,
    turn1_query: Optional[str] = None,
) -> ScenarioScore:
    issues: List[str] = []
    signals: List[str] = []
    text = _text_blob([answer or "", turn2_answer or ""])
    score = 1.0

    if not text or len(text.strip()) < 60:
        issues.append("collapse_or_too_short")
        score -= 0.6
    if _COLLAPSE_RE.search(text):
        issues.append("orchestration_collapse_phrase")
        score -= 0.5
    if _BROCHURE_RE.search(text):
        issues.append("brochure_language")
        score -= 0.15
    if _ROBOTIC_RE.search(text):
        issues.append("robotic_template_phrase")
        score -= 0.15

    op_hits = _OP_TERMS.findall(text)
    if op_hits:
        signals.append(f"operational_terms:{len(set(h.lower() for h in op_hits))}")
    else:
        issues.append("no_operational_vocabulary")
        score -= 0.25

    if packet:
        conf = float(packet.get("overall_confidence") or 0)
        signals.append(f"packet_confidence:{conf:.2f}")
        if packet.get("operational_synthesis"):
            signals.append("has_operational_synthesis")
        if packet.get("inferred_constraints"):
            signals.append(f"inferred_keys:{len(packet.get('inferred_constraints') or {})}")
        if packet.get("corridor_type") and packet.get("corridor_type") != "unknown":
            signals.append(f"corridor:{packet.get('corridor_type')}")

    exp = expectations or []
    for e in exp:
        tag = e.strip().lower()
        if tag == "no_collapse":
            if not text or len(text) < 60 or _COLLAPSE_RE.search(text):
                issues.append("expected_no_collapse")
                score -= 0.4
        elif tag == "class_band_or_followup":
            if not (
                _has_any(text, ["class band", "aircraft class", "city pair", "origin", "destination", "?"])
                or _has_any(text, ["operational read", "mission fit", "synthesis"])
            ):
                issues.append("missing_class_band_or_followup")
                score -= 0.3
        elif tag == "multi_aircraft":
            if not _has_any(
                text,
                ["multi-aircraft", "two aircraft", "split", "portfolio", "incompatible", "different band", "separate"],
            ):
                issues.append("missing_multi_aircraft_synthesis")
                score -= 0.35
        elif tag == "resist_single_aircraft":
            if _has_any(text, ["one aircraft can do it all", "single platform covers"]) and not _has_any(
                text, ["compromise", "unlikely", "would not", "hard to", "incompatible"]
            ):
                issues.append("oversimplified_single_aircraft")
                score -= 0.35
        elif tag == "ownership_economics":
            if not _has_any(
                text,
                ["hour", "utilization", "fractional", "ownership", "charter", "capital", "economics", "cost"],
            ):
                issues.append("missing_ownership_economics")
                score -= 0.35
        elif tag == "continuity":
            if turn2_answer and turn1_query:
                # Turn 2 should reflect turn-1 themes
                t1 = turn1_query.lower()
                t2 = turn2_answer.lower()
                anchors = []
                for tok in ("tokyo", "caribbean", "london", "14", "cost", "miami", "charter", "europe", "asia", "midsize", "executive"):
                    if tok in t1 and tok not in t2 and tok not in text.lower():
                        anchors.append(tok)
                if len(anchors) >= 2:
                    issues.append(f"continuity_lost:{','.join(anchors[:3])}")
                    score -= 0.35
                else:
                    signals.append("continuity_ok")
        elif tag == "honest_refusal_guidance":
            if not _has_any(text, ["would not", "not credible", "not operationally", "incompatible", "tech stop", "class"]):
                issues.append("missing_honest_boundary")
                score -= 0.3
            if len(text) < 100:
                issues.append("refusal_without_guidance")
                score -= 0.25
        elif tag == "broker_synthesis":
            if len(set(_OP_TERMS.findall(text))) < 3:
                issues.append("thin_broker_synthesis")
                score -= 0.25

    score = max(0.0, min(1.0, score))
    passed = score >= 0.65 and "collapse_or_too_short" not in issues and "orchestration_collapse_phrase" not in issues
    return ScenarioScore(
        scenario_id=scenario_id,
        category=category,
        passed=passed,
        score=score,
        issues=issues,
        signals=signals,
    )
