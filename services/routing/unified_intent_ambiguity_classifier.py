"""
Unified intent ambiguity classifier — observability-only semantic drift detection.

Does NOT influence routing, execution_path, or PipelineGate behavior.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from services.routing.unified_intent_router import (
    UnifiedExecutionPath,
    UnifiedIntentRoute,
    UnifiedSecondaryIntent,
    _dedupe_canonical_models,
    _detect_fact_field,
    _detect_market_field,
    _has_capability_or_route_signals,
    _mentioned_models,
    _MODEL_CONFIDENCE_THRESHOLD,
    _CAPABILITY_BORDERLINE_THRESHOLD,
)

from services.comparison.alternative_pipeline_responder import (
    is_alternative_execution_query,
    is_explicit_comparison_query,
)


class AmbiguityType(str, Enum):
    NONE = "none"
    LEXICAL = "lexical"
    INTENT_COLLISION = "intent_collision"
    BORDERLINE_CONFIDENCE = "borderline_confidence"
    UNRESOLVED_OBJECT = "unresolved_object"


class ConfidenceBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Bare tokens that map to multiple catalog families when not fully qualified.
_LEXICAL_AMBIGUOUS_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\blongitude\b", re.I), "longitude"),
    (re.compile(r"\bgulfstream\b(?!\s*(?:g\d|global))", re.I), "gulfstream"),
    (re.compile(r"\bchallenger\b(?!\s*\d)", re.I), "challenger"),
    (re.compile(r"\bcitation\b(?!\s+\w+)", re.I), "citation"),
    (re.compile(r"\blegacy\b(?!\s*\d)", re.I), "legacy"),
    (re.compile(r"\bfalcon\b(?!\s*\d)", re.I), "falcon"),
)


def _confidence_band(route: UnifiedIntentRoute) -> ConfidenceBand:
    conf = float(route.model_confidence)
    if conf >= _MODEL_CONFIDENCE_THRESHOLD:
        return ConfidenceBand.HIGH
    if conf >= _CAPABILITY_BORDERLINE_THRESHOLD:
        return ConfidenceBand.MEDIUM
    return ConfidenceBand.LOW


def _detect_lexical_ambiguity(query: str) -> Optional[str]:
    ql = (query or "").lower()
    hits: List[str] = []
    for pattern, label in _LEXICAL_AMBIGUOUS_PATTERNS:
        if pattern.search(ql):
            hits.append(label)
    if not hits:
        return None
    models = _dedupe_canonical_models(_mentioned_models(query or ""))
    if len(models) == 1:
        return None
    return hits[0] if len(hits) == 1 else ",".join(hits)


def _detect_intent_collision(query: str, route: UnifiedIntentRoute) -> bool:
    ql = (query or "").lower()
    has_capability = _has_capability_or_route_signals(ql)
    has_comparison = bool(is_explicit_comparison_query(query or ""))
    has_alternative = bool(is_alternative_execution_query(query or ""))
    if has_capability and has_comparison:
        return True
    if has_capability and has_alternative:
        return True
    if has_comparison and has_alternative:
        return True
    secondary = route.secondary_intent
    if (
        has_capability
        and secondary == UnifiedSecondaryIntent.AIRCRAFT_COMPARISON_LIKELY
        and not has_comparison
    ):
        return True
    return False


def _detect_unresolved_object(query: str, route: UnifiedIntentRoute) -> bool:
    ql = (query or "").lower()
    expects_model = bool(
        _detect_fact_field(ql)
        or _detect_market_field(ql)
        or _has_capability_or_route_signals(ql)
    )
    if not expects_model:
        return False
    has_mention = bool(route.model) or bool(_mentioned_models(query or ""))
    return has_mention and not route.model and route.execution_path == UnifiedExecutionPath.NONE


def classify_ambiguity(query: str, route: UnifiedIntentRoute) -> "AmbiguityReport":
    """
    Classify routing ambiguity for observability — never mutates ``route``.
    """
    band = _confidence_band(route)
    lexical = _detect_lexical_ambiguity(query or "")
    collision = _detect_intent_collision(query or "", route)
    borderline = (
        bool(route.model)
        and _CAPABILITY_BORDERLINE_THRESHOLD
        <= float(route.model_confidence)
        < _MODEL_CONFIDENCE_THRESHOLD
    )
    unresolved = _detect_unresolved_object(query or "", route)

    if collision:
        return AmbiguityReport(
            is_ambiguous=True,
            ambiguity_type=AmbiguityType.INTENT_COLLISION,
            confidence_band=band,
            recommendation=(
                "Capability and comparison/alternative signals co-present; "
                "verify execution_path matches dominant user intent."
            ),
        )
    if lexical:
        return AmbiguityReport(
            is_ambiguous=True,
            ambiguity_type=AmbiguityType.LEXICAL,
            confidence_band=band,
            recommendation=(
                f"Lexical token '{lexical}' is ambiguous without full model qualification; "
                "prefer compound model names in queries."
            ),
        )
    if unresolved:
        return AmbiguityReport(
            is_ambiguous=True,
            ambiguity_type=AmbiguityType.UNRESOLVED_OBJECT,
            confidence_band=band,
            recommendation=(
                "Model mention detected but router did not resolve a canonical model; "
                "legacy fallback likely."
            ),
        )
    if borderline:
        return AmbiguityReport(
            is_ambiguous=True,
            ambiguity_type=AmbiguityType.BORDERLINE_CONFIDENCE,
            confidence_band=ConfidenceBand.MEDIUM,
            recommendation=(
                f"Model confidence {route.model_confidence:.2f} is borderline "
                f"({_CAPABILITY_BORDERLINE_THRESHOLD}–{_MODEL_CONFIDENCE_THRESHOLD}); "
                "confirm catalog alias or compound name."
            ),
        )

    return AmbiguityReport(
        is_ambiguous=False,
        ambiguity_type=AmbiguityType.NONE,
        confidence_band=band,
        recommendation="No ambiguity detected.",
    )


@dataclass(frozen=True)
class AmbiguityReport:
    is_ambiguous: bool
    ambiguity_type: AmbiguityType
    confidence_band: ConfidenceBand
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "is_ambiguous": self.is_ambiguous,
            "ambiguity_type": self.ambiguity_type.value,
            "confidence_band": self.confidence_band.value,
            "recommendation": self.recommendation,
        }


__all__ = [
    "AmbiguityReport",
    "AmbiguityType",
    "ConfidenceBand",
    "classify_ambiguity",
]
