"""
Query-intent classification BEFORE aircraft recommendation.

Routes each turn to a specialized response mode so the system does not emit
purchase-style shortlists for comparisons, critiques, specs-only questions, or gallery asks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

class QueryRecommendationIntent(str, Enum):
    ACQUISITION_RECOMMENDATION = "acquisition_recommendation"
    MISSION_FEASIBILITY = "mission_feasibility"
    AIRCRAFT_COMPARISON = "aircraft_comparison"
    OPERATIONAL_TRADEOFF_ANALYSIS = "operational_tradeoff_analysis"
    OWNERSHIP_ECONOMICS = "ownership_economics"
    PAYLOAD_RANGE_ANALYSIS = "payload_range_analysis"
    VISUALIZATION_REQUEST = "visualization_request"
    AIRCRAFT_CRITIQUE = "aircraft_critique"
    SHORTLIST_RANKING = "shortlist_ranking"


# Intents that run the deterministic aircraft pipeline (ranked shortlist).
_RANKED_PIPELINE_INTENTS: frozenset[QueryRecommendationIntent] = frozenset(
    {
        QueryRecommendationIntent.ACQUISITION_RECOMMENDATION,
        QueryRecommendationIntent.MISSION_FEASIBILITY,
        QueryRecommendationIntent.AIRCRAFT_COMPARISON,
        QueryRecommendationIntent.OPERATIONAL_TRADEOFF_ANALYSIS,
        QueryRecommendationIntent.SHORTLIST_RANKING,
    }
)

# Intents that must not receive acquisition-style ranked shortlists.
_NON_ACQUISITION_INTENTS: frozenset[QueryRecommendationIntent] = frozenset(
    {
        QueryRecommendationIntent.AIRCRAFT_COMPARISON,
        QueryRecommendationIntent.OPERATIONAL_TRADEOFF_ANALYSIS,
        QueryRecommendationIntent.OWNERSHIP_ECONOMICS,
        QueryRecommendationIntent.PAYLOAD_RANGE_ANALYSIS,
        QueryRecommendationIntent.VISUALIZATION_REQUEST,
        QueryRecommendationIntent.AIRCRAFT_CRITIQUE,
    }
)

_VISUALIZATION_RE = re.compile(
    r"\b(?:"
    r"show\s+(?:me\s+)?(?:photos?|pictures?|images?|gallery)|"
    r"(?:interior|cockpit|exterior|cabin)\s+(?:photos?|pictures?|images?|graphic|layout)|"
    r"visuali[sz]e|visuali[sz]ation|"
    r"\bmap\b|range\s+map|rangemap|"
    r"reachable\s+cities?|cities?\s+(?:reachable|within\s+range)|"
    r"compare\s+layouts?|layout\s+comparison|cabin\s+comparison|"
    r"cabin\s+graphic|interior\s+layout|"
    r"just\s+(?:show|see)\s+(?:the\s+)?(?:cabin|interior|cockpit)|"
    r"pics?\s+of|"
    r"what\s+does\s+(?:the\s+)?\w+\s+(?:look|interior)\s+like"
    r")\b",
    re.I,
)

_CRITIQUE_RE = re.compile(
    r"\b(?:"
    r"what\s+(?:aircraft|jet|plane|model)s?\s+(?:should\s+)?(?:i\s+)?avoid|"
    r"what\s+not\s+to\s+bring|"
    r"would\s+you\s+avoid|"
    r"(?:aircraft|jets?|models?)\s+to\s+avoid|"
    r"not\s+(?:a\s+)?(?:good\s+)?(?:buy|purchase|fit)|"
    r"wouldn'?t\s+(?:buy|recommend|choose)|"
    r"steer\s+clear\s+of|"
    r"problems?\s+with|"
    r"issues?\s+with|"
    r"weakness(?:es)?\s+of|"
    r"downsides?\s+of|"
    r"why\s+(?:not|avoid)\s+"
    r")\b",
    re.I,
)

_COMPARISON_RE = re.compile(
    r"\b(?:"
    r"compare|comparison|versus|vs\.?\b|"
    r"which\s+is\s+better|"
    r"head[- ]to[- ]head|"
    r"side[- ]by[- ]side|"
    r"stack\s+up\s+against|"
    r"better\s+than\s+"
    r")\b",
    re.I,
)

_OWNERSHIP_ECONOMICS_RE = re.compile(
    r"\b(?:"
    r"ownership\s+cost|cost\s+of\s+ownership|"
    r"doc\b|fixed\s+cost|"
    r"capital\s+cost|hourly\s+cost|"
    r"ownership\s+economics|"
    r"variable\s+cost|"
    r"cost\s+per\s+(?:hour|nm|mile)|"
    r"economics\s+of\s+owning|"
    r"full\s+ownership\s+vs\s+fractional|"
    r"fractional\s+vs\s+full\s+ownership|"
    r"leaning\s+fractional|fractional\s+for\b|"
    r"overbuying|overbuy\b"
    r")\b",
    re.I,
)

_REGIONAL_MISSION_RE = re.compile(
    r"\b(?:"
    r"east\s+coast|us\s+east\s+coast|"
    r"caribbean|miami\s+to\s+caribbean|"
    r"ski\s+trips?\s+into\s+aspen|into\s+aspen|"
    r"aspen|telluride|mountain\s+airport|"
    r"high[- ]cycle|runway\s+flex"
    r")\b",
    re.I,
)

_ACQUISITION_RE = re.compile(
    r"\b(?:"
    r"what\s+(?:aircraft|jet|plane)\s+should\s+(?:i\s+)?(?:buy|purchase|acquire)|"
    r"which\s+(?:aircraft|jet|plane)\s+(?:should\s+)?(?:i\s+)?(?:buy|purchase)|"
    r"what\s+should\s+i\s+buy|"
    r"acquisition\s+(?:target|budget)|"
    r"shopping\s+for\s+(?:a\s+)?(?:jet|aircraft)|"
    r"in\s+the\s+market\s+for\s+(?:a\s+)?(?:jet|aircraft)"
    r")\b",
    re.I,
)

_MISSION_FEASIBILITY_RE = re.compile(
    r"\b(?:"
    r"can\s+(?:it|this|that|a)\s+(?:fly|make)\s+nonstop|"
    r"nonstop\s+(?:to|from|possible|feasible)|"
    r"feasibility|"
    r"mission\s+feasib|"
    r"can\s+it\s+reach|"
    r"fuel\s+stop|tech[- ]stop|"
    r"westbound|transoceanic|transatlantic|"
    r"runway\s+(?:length|requirement)|"
    r"hot\s+and\s+high|short\s+field|"
    r"operational(?:ly)?\s+(?:possible|feasible)|"
    r"make\s+it\s+(?:to|from)\s+"
    r")\b",
    re.I,
)

_PAYLOAD_RANGE_RE = re.compile(
    r"\b(?:"
    r"payload|baggage|bags|luggage|"
    r"with\s+\d+\s+(?:pax|passengers)|"
    r"passenger[s]?\s+and\s+(?:range|baggage|payload)|"
    r"how\s+far\s+with\s+\d+\s+(?:pax|passengers)|"
    r"range\s+with\s+(?:full\s+)?(?:pax|passengers|baggage)"
    r")\b",
    re.I,
)

_TRADEOFF_RE = re.compile(
    r"\b(?:"
    r"trade[- ]?offs?|"
    r"pros\s+and\s+cons|"
    r"operating\s+trade[- ]?offs?|"
    r"compromise\s+between|"
    r"vs\s+operating\s+cost|"
    r"runway\s+vs\s+range|"
    r"cabin\s+vs\s+(?:range|cost)|"
    r"what\s+give[s]?\s+up"
    r")\b",
    re.I,
)

_SHORTLIST_RE = re.compile(
    r"\b(?:"
    r"recommend|recommendation|"
    r"best\s+(?:jet|aircraft|option)|"
    r"which\s+(?:jet|aircraft)|"
    r"shortlist|top\s+(?:\d+|three|five)|"
    r"what\s+(?:jet|aircraft)\s+fits|"
    r"options\s+for\s+this\s+mission|"
    r"name\s+(?:\d+|three|five)\s+(?:jets?|aircraft|options)"
    r")\b",
    re.I,
)

_EXPLICIT_VIZ_RE = re.compile(
    r"\b(?:"
    r"range\s+map|rangemap|"
    r"reachable\s+cities?|cities?\s+(?:reachable|within\s+range)|"
    r"compare\s+layouts?|layout\s+comparison|cabin\s+comparison|"
    r"cabin\s+graphic|interior\s+layout|"
    r"visuali[sz]e|visuali[sz]ation"
    r")\b",
    re.I,
)

_ROUTE_HINT_RE = re.compile(
    r"\b(?:"
    r"\bto\s+\w+|\bfrom\s+\w+|"
    r"city\s+pair|"
    r"los\s+angeles|new\s+york|london|tokyo|miami|"
    r"nonstop|transatlantic|transpacific"
    r")\b",
    re.I,
)


@dataclass
class QueryRecommendationIntentResult:
    intent: QueryRecommendationIntent
    confidence: float
    source: str = "heuristic"
    signals: List[str] = field(default_factory=list)
    requires_ranked_pipeline: bool = False
    allows_acquisition_framing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": round(float(self.confidence), 4),
            "source": self.source,
            "signals": list(self.signals),
            "requires_ranked_pipeline": self.requires_ranked_pipeline,
            "allows_acquisition_framing": self.allows_acquisition_framing,
        }


def _mentioned_models(query: str) -> List[str]:
    try:
        from services.consultant.recommendation_engine import detect_models_from_text

        return detect_models_from_text(query or "")
    except Exception:
        return []


def is_visualization_query(query: str) -> bool:
    return bool(_VISUALIZATION_RE.search((query or "").lower()))


def _score_intents(ql: str, *, models: List[str], has_route: bool) -> Dict[QueryRecommendationIntent, float]:
    scores: Dict[QueryRecommendationIntent, float] = {i: 0.0 for i in QueryRecommendationIntent}

    if _VISUALIZATION_RE.search(ql):
        scores[QueryRecommendationIntent.VISUALIZATION_REQUEST] += 0.95

    if _CRITIQUE_RE.search(ql):
        scores[QueryRecommendationIntent.AIRCRAFT_CRITIQUE] += 0.92

    if _COMPARISON_RE.search(ql) or (len(models) >= 2 and "compare" in ql):
        scores[QueryRecommendationIntent.AIRCRAFT_COMPARISON] += 0.88
    if len(models) >= 2:
        scores[QueryRecommendationIntent.AIRCRAFT_COMPARISON] += 0.75

    if _OWNERSHIP_ECONOMICS_RE.search(ql) and not _ACQUISITION_RE.search(ql):
        scores[QueryRecommendationIntent.OWNERSHIP_ECONOMICS] += 0.85
    if "fractional" in ql:
        scores[QueryRecommendationIntent.OWNERSHIP_ECONOMICS] += 0.7

    if re.search(
        r"\b(?:lowest|minimize|prioritize)\b.*\b(?:direct\s+)?operating\s+cost\b",
        ql,
    ) or re.search(r"\boperating\s+cost\b.*\b(?:recommend|dispatch|reliability)\b", ql):
        scores[QueryRecommendationIntent.OPERATIONAL_TRADEOFF_ANALYSIS] += 0.88
        scores[QueryRecommendationIntent.OWNERSHIP_ECONOMICS] -= 0.5

    if _TRADEOFF_RE.search(ql):
        scores[QueryRecommendationIntent.OPERATIONAL_TRADEOFF_ANALYSIS] += 0.82

    if re.search(r"\bwhat\s+not\s+to\s+bring\b", ql) and _REGIONAL_MISSION_RE.search(ql):
        scores[QueryRecommendationIntent.SHORTLIST_RANKING] += 0.92
        scores[QueryRecommendationIntent.PAYLOAD_RANGE_ANALYSIS] -= 0.55

    if _REGIONAL_MISSION_RE.search(ql) and (
        _SHORTLIST_RE.search(ql)
        or re.search(r"\b(?:recommend|fits?|what\s+not\s+to\s+bring|actually\s+fits)\b", ql)
    ):
        scores[QueryRecommendationIntent.SHORTLIST_RANKING] += 0.75
        scores[QueryRecommendationIntent.MISSION_FEASIBILITY] += 0.35
        scores[QueryRecommendationIntent.AIRCRAFT_CRITIQUE] -= 0.4

    if _PAYLOAD_RANGE_RE.search(ql) and not _SHORTLIST_RE.search(ql):
        scores[QueryRecommendationIntent.PAYLOAD_RANGE_ANALYSIS] += 0.78

    if _ACQUISITION_RE.search(ql):
        scores[QueryRecommendationIntent.ACQUISITION_RECOMMENDATION] += 0.9

    if _MISSION_FEASIBILITY_RE.search(ql) or (has_route and "nonstop" in ql):
        scores[QueryRecommendationIntent.MISSION_FEASIBILITY] += 0.8

    if _SHORTLIST_RE.search(ql):
        scores[QueryRecommendationIntent.SHORTLIST_RANKING] += 0.72
    if has_route and any(w in ql for w in ("recommend", "best", "which", "shortlist", "options")):
        scores[QueryRecommendationIntent.SHORTLIST_RANKING] += 0.35
        scores[QueryRecommendationIntent.MISSION_FEASIBILITY] += 0.2

    return scores


def classify_query_recommendation_intent(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> QueryRecommendationIntentResult:
    """
    Classify the user query before any ranked aircraft recommendation runs.
    """
    q = (query or "").strip()
    ql = q.lower()

    try:
        from services.orchestration.response_mode_classifier import (
            classify_orchestration_response_mode,
        )

        orm = classify_orchestration_response_mode(q, history=history)
        if orm.suppresses_aircraft_recommendations:
            return QueryRecommendationIntentResult(
                intent=QueryRecommendationIntent.OPERATIONAL_TRADEOFF_ANALYSIS,
                confidence=max(0.88, orm.confidence),
                source="orchestration_response_mode",
                signals=[orm.mode.value],
                requires_ranked_pipeline=True,
                allows_acquisition_framing=False,
            )
    except Exception:
        pass

    models = _mentioned_models(q)
    has_route = bool(_ROUTE_HINT_RE.search(ql))

    scores = _score_intents(ql, models=models, has_route=has_route)
    best_intent = max(scores, key=lambda k: scores[k])
    best_score = scores[best_intent]

    if _EXPLICIT_VIZ_RE.search(ql):
        best_intent = QueryRecommendationIntent.VISUALIZATION_REQUEST
        best_score = max(best_score, scores[QueryRecommendationIntent.VISUALIZATION_REQUEST], 0.95)

    if best_score < 0.45:
        if _SHORTLIST_RE.search(ql) or (has_route and "recommend" in ql):
            best_intent = QueryRecommendationIntent.SHORTLIST_RANKING
            best_score = 0.55
        else:
            best_intent = QueryRecommendationIntent.PAYLOAD_RANGE_ANALYSIS
            best_score = 0.42

    # Tie-break: explicit visualization beats comparison when range map / layouts / etc.
    if (
        scores[QueryRecommendationIntent.VISUALIZATION_REQUEST] >= 0.9
        and best_intent != QueryRecommendationIntent.VISUALIZATION_REQUEST
        and _EXPLICIT_VIZ_RE.search(ql)
    ):
        best_intent = QueryRecommendationIntent.VISUALIZATION_REQUEST
        best_score = scores[best_intent]

    # Tie-break: comparison beats shortlist when multiple models named (not visualization)
    if (
        best_intent != QueryRecommendationIntent.VISUALIZATION_REQUEST
        and len(models) >= 2
        and scores[QueryRecommendationIntent.AIRCRAFT_COMPARISON] >= best_score - 0.15
    ):
        best_intent = QueryRecommendationIntent.AIRCRAFT_COMPARISON
        best_score = scores[best_intent]

    # Critique beats acquisition when both match
    if (
        scores[QueryRecommendationIntent.AIRCRAFT_CRITIQUE] >= 0.85
        and best_intent == QueryRecommendationIntent.ACQUISITION_RECOMMENDATION
    ):
        best_intent = QueryRecommendationIntent.AIRCRAFT_CRITIQUE
        best_score = scores[best_intent]

    signals = [k.value for k, v in scores.items() if v >= 0.55]
    signals.sort(key=lambda k: scores[k], reverse=True)

    requires_pipeline = best_intent in _RANKED_PIPELINE_INTENTS
    allows_acquisition = best_intent in (
        QueryRecommendationIntent.ACQUISITION_RECOMMENDATION,
        QueryRecommendationIntent.SHORTLIST_RANKING,
        QueryRecommendationIntent.MISSION_FEASIBILITY,
    )

    return QueryRecommendationIntentResult(
        intent=best_intent,
        confidence=min(0.99, max(0.45, best_score)),
        source="heuristic",
        signals=signals[:5],
        requires_ranked_pipeline=requires_pipeline,
        allows_acquisition_framing=allows_acquisition,
    )


def requires_ranked_aircraft_pipeline(intent: QueryRecommendationIntent) -> bool:
    return intent in _RANKED_PIPELINE_INTENTS


def allows_acquisition_style_framing(intent: QueryRecommendationIntent) -> bool:
    if intent in _NON_ACQUISITION_INTENTS:
        return False
    return intent in (
        QueryRecommendationIntent.ACQUISITION_RECOMMENDATION,
        QueryRecommendationIntent.SHORTLIST_RANKING,
        QueryRecommendationIntent.MISSION_FEASIBILITY,
    )


def map_to_consultant_fine_intent(intent: QueryRecommendationIntent) -> str:
    """Map to legacy ``consultant_fine_intent`` string for existing routers."""
    mapping = {
        QueryRecommendationIntent.ACQUISITION_RECOMMENDATION: "aircraft_recommendation",
        QueryRecommendationIntent.MISSION_FEASIBILITY: "aviation_mission",
        QueryRecommendationIntent.AIRCRAFT_COMPARISON: "aircraft_comparison",
        QueryRecommendationIntent.OPERATIONAL_TRADEOFF_ANALYSIS: "aviation_mission",
        QueryRecommendationIntent.OWNERSHIP_ECONOMICS: "general_question",
        QueryRecommendationIntent.PAYLOAD_RANGE_ANALYSIS: "aircraft_specs",
        QueryRecommendationIntent.VISUALIZATION_REQUEST: "general_question",
        QueryRecommendationIntent.AIRCRAFT_CRITIQUE: "general_question",
        QueryRecommendationIntent.SHORTLIST_RANKING: "aircraft_recommendation",
    }
    return mapping.get(intent, "aircraft_recommendation")


def consultant_response_mode_for_intent(intent: QueryRecommendationIntent) -> str:
    """Suggested ``consultant_response_mode`` for downstream routers."""
    mapping = {
        QueryRecommendationIntent.ACQUISITION_RECOMMENDATION: "mission_advisory",
        QueryRecommendationIntent.MISSION_FEASIBILITY: "mission_advisory",
        QueryRecommendationIntent.AIRCRAFT_COMPARISON: "comparison_mode",
        QueryRecommendationIntent.OPERATIONAL_TRADEOFF_ANALYSIS: "advisory_mode",
        QueryRecommendationIntent.OWNERSHIP_ECONOMICS: "educational_mode",
        QueryRecommendationIntent.PAYLOAD_RANGE_ANALYSIS: "educational_mode",
        QueryRecommendationIntent.VISUALIZATION_REQUEST: "image_showcase",
        QueryRecommendationIntent.AIRCRAFT_CRITIQUE: "educational_mode",
        QueryRecommendationIntent.SHORTLIST_RANKING: "mission_advisory",
    }
    return mapping.get(intent, "advisory_mode")


def explicit_models_required(intent: QueryRecommendationIntent, query: str) -> bool:
    return intent == QueryRecommendationIntent.AIRCRAFT_COMPARISON and len(_mentioned_models(query)) >= 2


def apply_query_intent_metadata(
    data_used: Dict[str, Any],
    result: QueryRecommendationIntentResult,
) -> None:
    """Persist classification on ``data_used`` for downstream routers."""
    data_used["query_recommendation_intent"] = result.intent.value
    data_used["query_recommendation_intent_confidence"] = result.confidence
    data_used["query_recommendation_intent_source"] = result.source
    data_used["query_recommendation_intent_signals"] = list(result.signals)
    data_used["query_recommendation_requires_pipeline"] = result.requires_ranked_pipeline
    data_used["query_recommendation_allows_acquisition"] = result.allows_acquisition_framing
    mode = consultant_response_mode_for_intent(result.intent)
    data_used["consultant_response_mode_canonical"] = mode
    data_used["consultant_response_mode"] = mode


def build_intent_authority_note(result: QueryRecommendationIntentResult) -> str:
    """Short block for LLM context when pipeline is skipped."""
    lines = [
        "[QUERY INTENT — PRE-RECOMMENDATION]",
        f"Classified intent: {result.intent.value} (confidence {result.confidence:.2f}).",
    ]
    if result.requires_ranked_pipeline:
        lines.append("Ranked aircraft shortlist: produced by deterministic pipeline (see authority block).")
    else:
        lines.append(
            "Do NOT produce a purchase-style ranked shortlist. Answer in "
            f"{result.intent.value.replace('_', ' ')} mode — analysis, comparison, or critique as appropriate."
        )
    return "\n".join(lines)


classifyQueryRecommendationIntent = classify_query_recommendation_intent
