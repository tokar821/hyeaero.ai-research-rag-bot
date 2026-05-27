"""
Deterministic regression scenarios — decomposition, fleet doctrine, continuity.

Not aircraft-accuracy tests: they assert stable operational structure and
broker-grade strategic signals across turns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

_COLLAPSE_RE = re.compile(
    r"I couldn't get a response|I don't have enough verified|try again later",
    re.I,
)
_EMPTY_SECTION_RE = re.compile(
    r"^Distance Considerations:\s*$",
    re.MULTILINE,
)
_UTILITY_ANCHOR_RE = re.compile(
    r"\b(?:caravan|stationair|baron)\b.*\b(?:primary|lead|start with|best fit)\b",
    re.I,
)
_SINGLE_JET_COLLAPSE_RE = re.compile(
    r"\b(?:one aircraft|single aircraft|only one jet)\b.*\b(?:covers? everything|handles? all)\b",
    re.I,
)


@dataclass
class ConsistencyScenario:
    id: str
    title: str
    query: str
    follow_up: Optional[str] = None
    must_signals: List[str] = field(default_factory=list)
    must_not: List[str] = field(default_factory=list)
    packet_keys: List[str] = field(default_factory=list)
    min_turn_agreement: float = 0.55


CONSISTENCY_SCENARIOS: List[ConsistencyScenario] = [
    ConsistencyScenario(
        id="pe_multi_region",
        title="PE group multi-region strategy",
        query=(
            "Private equity group — NYC, Dallas, London hubs with occasional Middle East. "
            "Frequent executive movement, industrial airport access, recurring transatlantic, "
            "occasional ULR CEO continuation. What aircraft strategy?"
        ),
        follow_up="Same mission — confirm fleet vs single-aircraft posture.",
        must_signals=[
            "fleet",
            "multi",
            "incompatible",
            "transatlantic",
            "portfolio",
        ],
        must_not=["learjet 60", "citation x as the only"],
        packet_keys=["incompatible_mission_bands", "dual_use_or_multi_leg"],
    ),
    ConsistencyScenario(
        id="aspen_london_coexist",
        title="Aspen + London incompatible bands",
        query=(
            "We fly Aspen regularly and London monthly — same leadership team. "
            "What operational structure makes sense?"
        ),
        follow_up="Does one aircraft cover both?",
        must_signals=["incompatible", "mountain", "ulr", "band", "fleet"],
        must_not=["one aircraft covers everything"],
        packet_keys=["incompatible_mission_bands"],
        min_turn_agreement=0.5,
    ),
    ConsistencyScenario(
        id="ceo_middle_east_continuation",
        title="CEO continuation to Middle East",
        query=(
            "US-based corporate — Dallas and New York primary, London transatlantic, "
            "CEO occasionally continues to Dubai or Riyadh nonstop. Recommend structure."
        ),
        must_signals=["ulr", "nonstop", "continuation", "fleet", "fuel"],
        packet_keys=["dual_use_or_multi_leg"],
    ),
    ConsistencyScenario(
        id="ownership_250_300",
        title="250–300 hr ownership crossover",
        query=(
            "About 275 flight hours per year, mix of coast-to-coast and Caribbean. "
            "Ownership vs fractional vs charter — capital and dispatch view?"
        ),
        must_signals=[
            "fractional",
            "charter",
            "utilization",
            "dispatch",
            "capital",
            "management",
        ],
        must_not=["worth considering if priorities shift"],
    ),
    ConsistencyScenario(
        id="miami_caribbean_executive",
        title="Miami Caribbean executive shuttle realism",
        query=(
            "Miami base, 8–10 executives, Caribbean and northern South America weekly, "
            "short runways at islands but cabin and dispatch reliability matter. Options?"
        ),
        must_signals=["pressur", "dispatch", "runway", "jet"],
        must_not=["caravan", "stationair", "baron"],
        packet_keys=["minimum_jet_cabin_floor", "executive_travel_profile"],
    ),
]


@dataclass
class ConsistencyScore:
    scenario_id: str
    passed: bool
    score: float
    issues: List[str] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "passed": self.passed,
            "score": round(self.score, 3),
            "issues": list(self.issues),
            "signals": list(self.signals),
        }


def _has_any(text: str, patterns: Sequence[str]) -> bool:
    low = text.lower()
    return any(p.lower() in low for p in patterns)


def _token_overlap(a: str, b: str) -> float:
    ta = {t for t in re.findall(r"[a-z]{4,}", (a or "").lower())}
    tb = {t for t in re.findall(r"[a-z]{4,}", (b or "").lower())}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def score_consistency(
    *,
    scenario: ConsistencyScenario,
    answer: str,
    follow_up_answer: Optional[str] = None,
    packet: Optional[Dict[str, Any]] = None,
    turn2_packet: Optional[Dict[str, Any]] = None,
) -> ConsistencyScore:
    issues: List[str] = []
    signals: List[str] = []
    score = 1.0
    blob = "\n".join(x for x in [answer, follow_up_answer] if x)

    if _COLLAPSE_RE.search(blob):
        issues.append("collapse_or_refusal")
        score -= 0.5

    if _EMPTY_SECTION_RE.search(answer or ""):
        issues.append("empty_distance_considerations_header")
        score -= 0.2

    for sig in scenario.must_signals:
        if _has_any(blob, [sig]):
            signals.append(f"has:{sig}")
        else:
            issues.append(f"missing_signal:{sig}")
            score -= 0.12

    for bad in scenario.must_not:
        if bad.lower() in blob.lower():
            issues.append(f"forbidden:{bad}")
            score -= 0.18

    if _UTILITY_ANCHOR_RE.search(blob) and scenario.id == "miami_caribbean_executive":
        issues.append("utility_aircraft_as_primary")
        score -= 0.35

    if _SINGLE_JET_COLLAPSE_RE.search(blob) and scenario.id in (
        "pe_multi_region",
        "aspen_london_coexist",
    ):
        issues.append("single_aircraft_collapse")
        score -= 0.3

    inf = {}
    if isinstance(packet, dict):
        inf = packet.get("inferred_constraints") or {}
    for key in scenario.packet_keys:
        if inf.get(key) or (isinstance(turn2_packet, dict) and (turn2_packet.get("inferred_constraints") or {}).get(key)):
            signals.append(f"packet:{key}")
        else:
            issues.append(f"missing_packet:{key}")
            score -= 0.15

    if follow_up_answer and scenario.follow_up:
        overlap = _token_overlap(answer, follow_up_answer)
        if overlap < scenario.min_turn_agreement:
            issues.append(f"turn_drift:{overlap:.2f}")
            score -= 0.2
        else:
            signals.append(f"turn_overlap:{overlap:.2f}")

    score = max(0.0, min(1.0, score))
    passed = score >= 0.72 and not any(
        i.startswith("collapse") or i.startswith("utility_aircraft") for i in issues
    )
    return ConsistencyScore(
        scenario_id=scenario.id,
        passed=passed,
        score=score,
        issues=issues,
        signals=signals,
    )
