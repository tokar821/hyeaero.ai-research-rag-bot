"""
Orchestration response mode classifier — interpretation vs recommendation discipline.

Every advisory turn MUST classify into ONE primary mode. Interpretation and structure
modes suppress aircraft recommendations unless the user explicitly requests them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

ORCHESTRATION_RESPONSE_MODE_KEY = "orchestration_response_mode"


class OrchestrationResponseMode(str, Enum):
    INTERPRETATION_MODE = "interpretation_mode"
    STRUCTURE_MODE = "structure_mode"
    RECOMMENDATION_MODE = "recommendation_mode"
    BUY_DECISION_MODE = "buy_decision_mode"
    COMPARISON_MODE = "comparison_mode"


_EXPLICIT_AIRCRAFT_REQUEST_RE = re.compile(
    r"\b(?:"
    r"what\s+(?:aircraft|jet|plane)s?\b|"
    r"which\s+(?:aircraft|jet|plane)s?\b|"
    r"what\s+should\s+i\s+buy\b|"
    r"which\s+(?:jet|aircraft)\s+(?:fits|fit|should)\b|"
    r"recommend\s+(?:aircraft|a\s+jet|jets?|models?)\b|"
    r"(?:shortlist|name)\s+(?:\d+|three|five)\s+(?:jets?|aircraft|options)\b|"
    r"what\s+(?:jet|aircraft)\s+(?:fits|should)\b|"
    r"shopping\s+for\s+(?:a\s+)?(?:jet|aircraft)\b|"
    r"acquisition\s+(?:target|recommendation)\b"
    r")\b",
    re.I,
)

_INTERPRETATION_RE = re.compile(
    r"\b(?:"
    r"how\s+should\s+(?:this|the\s+network|this\s+network)\s+be\s+(?:interpreted|understood|represented)\b|"
    r"how\s+should\s+(?:this|the)\s+(?:network|mission)\s+be\s+(?:interpreted|understood)\b|"
    r"what\s+operational\s+domains\s+(?:exist|are\s+present|apply)\b|"
    r"what\s+(?:actually\s+)?dominates?\s+(?:utilization|usage|hours?|flying)\b|"
    r"how\s+should\s+continuation\s+hubs?\s+be\s+(?:represented|interpreted|understood)\b|"
    r"dominant\s+(?:mission\s+)?(?:domains?|utilization)\b|"
    r"interpret(?:ation|ing)?\s+(?:of\s+)?(?:this|the)\s+(?:network|mission)\b|"
    r"mission\s+interpretation\b|"
    r"utilization\s+hierarchy\b|"
    r"what\s+dominates?\s+(?:the\s+)?(?:network|mission|utilization)\b|"
    r"which\s+routes?\s+should\s+actually\s+drive\b|"
    r"should\s+we\s+optimize\s+around\s+those\b"
    r")\b",
    re.I,
)

_STRUCTURE_RE = re.compile(
    r"\b(?:"
    r"what\s+structure\s+fits\b|"
    r"is\s+this\s+(?:structurally\s+)?(?:coherent|viable)\b|"
    r"structural(?:ly)?\s+coherent\b|"
    r"how\s+should\s+(?:this|the|continuation)\s+(?:network|mission|continuation)?\s*be\s+(?:structured|decomposed)\b|"
    r"how\s+should\s+continuation\b|"
    r"mission\s+structure\b|"
    r"structural\s+(?:analysis|interpretation|conflict|decomposition)\b|"
    r"what\s+breaks\b|"
    r"decompos(?:e|ition)\b|"
    r"single[\s-]platform\s+optimi[sz]ation\b|"
    r"one\s+aircraft\s+solution\b|"
    r"leadership\s+wants\s+one\b|"
    r"multiple\s+missions?\b|"
    r"structurally\s+(?:multiple|incompatible)\b|"
    r"without\s+breaking\s+origin\s+integrity\b|"
    r"what\s+is\s+structurally\s+wrong\b|"
    r"operationally\s+coherent\b|"
    r"one\s+flagship\s+aircraft\b|"
    r"leadership\s+believes\s+one\b"
    r")\b",
    re.I,
)

_COMPARISON_RE = re.compile(
    r"\b(?:compare|comparison|versus|vs\.?\b|head[- ]to[- ]head|which\s+is\s+better)\b",
    re.I,
)

_BUY_DECISION_RE = re.compile(
    r"\b(?:"
    r"should\s+i\s+buy\b|"
    r"worth\s+buying\b|"
    r"purchase\s+decision\b|"
    r"acquire\s+or\s+charter\b|"
    r"buy\s+vs\s+(?:charter|fractional)\b"
    r")\b",
    re.I,
)

_RECOMMENDATION_RE = re.compile(
    r"\b(?:"
    r"recommend(?:ation)?\b|"
    r"best\s+(?:jet|aircraft|option)\b|"
    r"top\s+(?:\d+|three|five)\b|"
    r"options\s+for\s+this\s+mission\b|"
    r"what\s+(?:jet|aircraft)\s+fits\b"
    r")\b",
    re.I,
)


@dataclass
class OrchestrationResponseModeResult:
    mode: OrchestrationResponseMode
    confidence: float
    source: str = "heuristic"
    signals: List[str] = field(default_factory=list)
    suppresses_aircraft_recommendations: bool = False
    explicit_aircraft_request: bool = False
    structural_first: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "confidence": round(float(self.confidence), 4),
            "source": self.source,
            "signals": list(self.signals),
            "suppresses_aircraft_recommendations": self.suppresses_aircraft_recommendations,
            "explicit_aircraft_request": self.explicit_aircraft_request,
            "structural_first": self.structural_first,
        }


def explicit_aircraft_request(query: str) -> bool:
    return bool(_EXPLICIT_AIRCRAFT_REQUEST_RE.search((query or "").strip()))


def classify_orchestration_response_mode(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> OrchestrationResponseModeResult:
    """
    Classify the user's primary response mode for this orchestration turn.

    Interpretation and structure modes suppress aircraft unless the user explicitly
    requests models, purchase guidance, or a shortlist.
    """
    del history  # reserved for follow-up refinement
    q = (query or "").strip()
    ql = q.lower()
    explicit = explicit_aircraft_request(q)

    scores: Dict[OrchestrationResponseMode, float] = {
        m: 0.0 for m in OrchestrationResponseMode
    }

    if _STRUCTURE_RE.search(ql):
        scores[OrchestrationResponseMode.STRUCTURE_MODE] += 0.92
    if _INTERPRETATION_RE.search(ql):
        scores[OrchestrationResponseMode.INTERPRETATION_MODE] += 0.95

    # Multi-domain / payload planning without an aircraft ask → structure-first
    if re.search(
        r"\b(?:structurally\s+wrong|variable\s+passenger|equipment\s+pallets?|"
        r"single[\s-]aircraft\s+strategy|one\s+aircraft\s+category)\b",
        ql,
    ) and not explicit:
        scores[OrchestrationResponseMode.STRUCTURE_MODE] += 0.88

    if _COMPARISON_RE.search(ql):
        scores[OrchestrationResponseMode.COMPARISON_MODE] += 0.88
    if _BUY_DECISION_RE.search(ql):
        scores[OrchestrationResponseMode.BUY_DECISION_MODE] += 0.85
    if _RECOMMENDATION_RE.search(ql) or explicit:
        scores[OrchestrationResponseMode.RECOMMENDATION_MODE] += 0.80

    best = max(scores, key=lambda k: scores[k])
    best_score = scores[best]

    if best_score < 0.45:
        if explicit:
            best = OrchestrationResponseMode.RECOMMENDATION_MODE
            best_score = 0.55
        elif _STRUCTURE_RE.search(ql):
            best = OrchestrationResponseMode.STRUCTURE_MODE
            best_score = 0.50
        else:
            best = OrchestrationResponseMode.RECOMMENDATION_MODE
            best_score = 0.40

    # Interpretation beats generic recommendation when hierarchy / routing discipline is the ask
    if (
        scores[OrchestrationResponseMode.INTERPRETATION_MODE] >= 0.90
        and scores[OrchestrationResponseMode.INTERPRETATION_MODE]
        >= scores[OrchestrationResponseMode.RECOMMENDATION_MODE]
        and not explicit
    ):
        best = OrchestrationResponseMode.INTERPRETATION_MODE
        best_score = scores[best]

    # Structure beats recommendation for diagnostic / coherence questions
    if (
        scores[OrchestrationResponseMode.STRUCTURE_MODE] >= 0.85
        and scores[OrchestrationResponseMode.STRUCTURE_MODE]
        >= scores[OrchestrationResponseMode.RECOMMENDATION_MODE]
        and not explicit
    ):
        best = OrchestrationResponseMode.STRUCTURE_MODE
        best_score = scores[best]

    if re.search(r"\bwhat\s+actually\s+fits\b", ql):
        best = OrchestrationResponseMode.RECOMMENDATION_MODE
        best_score = max(best_score, 0.90)

    # Explicit aircraft request overrides interpretation/structure unless viability / continuation ask
    if explicit and best in (
        OrchestrationResponseMode.INTERPRETATION_MODE,
        OrchestrationResponseMode.STRUCTURE_MODE,
    ):
        if re.search(r"\b(?:is\s+this\s+viable|what\s+breaks|how\s+should\s+continuation)\b", ql):
            pass  # keep structure / interpretation
        else:
            best = OrchestrationResponseMode.RECOMMENDATION_MODE
            best_score = max(best_score, 0.85)

    # Comparison beats recommendation when both match and no explicit buy ask
    if (
        scores[OrchestrationResponseMode.COMPARISON_MODE] >= 0.85
        and best == OrchestrationResponseMode.RECOMMENDATION_MODE
        and not explicit
    ):
        best = OrchestrationResponseMode.COMPARISON_MODE
        best_score = scores[best]

    # Buy decision beats generic recommendation
    if (
        scores[OrchestrationResponseMode.BUY_DECISION_MODE] >= 0.80
        and best == OrchestrationResponseMode.RECOMMENDATION_MODE
    ):
        best = OrchestrationResponseMode.BUY_DECISION_MODE
        best_score = scores[best]

    signals = [m.value for m, v in scores.items() if v >= 0.55]
    signals.sort(key=lambda k: scores[OrchestrationResponseMode(k)], reverse=True)

    suppress = best in (
        OrchestrationResponseMode.INTERPRETATION_MODE,
        OrchestrationResponseMode.STRUCTURE_MODE,
    ) and not explicit

    structural_first = best in (
        OrchestrationResponseMode.INTERPRETATION_MODE,
        OrchestrationResponseMode.STRUCTURE_MODE,
    )

    return OrchestrationResponseModeResult(
        mode=best,
        confidence=min(0.99, max(0.40, best_score)),
        source="heuristic",
        signals=signals[:5],
        suppresses_aircraft_recommendations=suppress,
        explicit_aircraft_request=explicit,
        structural_first=structural_first,
    )


def apply_orchestration_response_mode_metadata(
    data_used: Dict[str, Any],
    result: OrchestrationResponseModeResult,
) -> None:
    data_used[ORCHESTRATION_RESPONSE_MODE_KEY] = result.to_dict()
    data_used["orchestration_response_mode_value"] = result.mode.value
    data_used["orchestration_suppresses_aircraft"] = result.suppresses_aircraft_recommendations
    data_used["orchestration_structural_first"] = result.structural_first
    if result.suppresses_aircraft_recommendations:
        data_used["recommend_aircraft_gated"] = 0
        data_used["defer_global_shortlist"] = True


def load_orchestration_response_mode(
    data_used: Optional[Dict[str, Any]],
) -> Optional[OrchestrationResponseModeResult]:
    if not isinstance(data_used, dict):
        return None
    raw = data_used.get(ORCHESTRATION_RESPONSE_MODE_KEY)
    if not isinstance(raw, dict):
        return None
    try:
        mode = OrchestrationResponseMode(str(raw.get("mode") or ""))
    except ValueError:
        return None
    return OrchestrationResponseModeResult(
        mode=mode,
        confidence=float(raw.get("confidence") or 0.5),
        source=str(raw.get("source") or "cached"),
        signals=list(raw.get("signals") or []),
        suppresses_aircraft_recommendations=bool(
            raw.get("suppresses_aircraft_recommendations")
        ),
        explicit_aircraft_request=bool(raw.get("explicit_aircraft_request")),
        structural_first=bool(raw.get("structural_first")),
    )


__all__ = [
    "ORCHESTRATION_RESPONSE_MODE_KEY",
    "OrchestrationResponseMode",
    "OrchestrationResponseModeResult",
    "apply_orchestration_response_mode_metadata",
    "classify_orchestration_response_mode",
    "explicit_aircraft_request",
    "load_orchestration_response_mode",
]
