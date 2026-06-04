"""
Unified intent hardening guard — passive production safety annotations.

Runs after router finalization and before PipelineGate. Observe-only: never blocks
execution or mutates routing decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from services.comparison.alternative_pipeline_responder import (
    is_alternative_execution_query,
    is_explicit_comparison_query,
)
from services.routing.unified_intent_ambiguity_classifier import AmbiguityReport, classify_ambiguity
from services.routing.unified_intent_production_metrics import (
    get_production_metrics,
    record_hardening_event,
)
from services.routing.unified_intent_router import (
    UnifiedExecutionPath,
    UnifiedIntentRoute,
    UnifiedSecondaryIntent,
    _detect_fact_field,
    _detect_market_field,
    _has_capability_or_route_signals,
    _mentioned_models,
    build_unified_intent_shadow,
)

logger = logging.getLogger(__name__)

HARDENING_ROUTING_FAILURE = "HARDENING_ROUTING_FAILURE"


@dataclass(frozen=True)
class HardeningGuardResult:
    routing_failure: bool
    hardening_reason: Optional[str]
    requires_fallback_analysis: bool
    event_code: Optional[str]
    expected_path_category: Optional[str]
    ambiguity_report: AmbiguityReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "routing_failure": self.routing_failure,
            "hardening_reason": self.hardening_reason,
            "requires_fallback_analysis": self.requires_fallback_analysis,
            "event_code": self.event_code,
            "expected_path_category": self.expected_path_category,
            "ambiguity_report": self.ambiguity_report.to_dict(),
        }

    @property
    def hardening_flags(self) -> Dict[str, bool]:
        return {
            "routing_failure": self.routing_failure,
            "ambiguity_detected": self.ambiguity_report.is_ambiguous,
            "fallback_triggered": self.requires_fallback_analysis,
        }


def _has_model_mention(query: str, route: UnifiedIntentRoute) -> bool:
    return bool(route.model) or bool(_mentioned_models(query or ""))


def _expected_path_category(query: str, route: UnifiedIntentRoute) -> Optional[str]:
    """Read-only expectation check — does not alter routing."""
    q = query or ""
    ql = q.lower()

    if is_alternative_execution_query(q) and not is_explicit_comparison_query(q):
        return "alternative"
    if is_explicit_comparison_query(q):
        return "comparison"
    if _has_capability_or_route_signals(ql):
        return "capability"
    market_field = _detect_market_field(ql)
    fact_field = _detect_fact_field(ql)
    if market_field and not fact_field:
        return "market"
    if fact_field and _has_model_mention(q, route):
        return "fact"
    return None


def _router_documented_deferral(
    category: Optional[str],
    route: UnifiedIntentRoute,
    *,
    query: str = "",
) -> bool:
    """
    Router left execution_path NONE with explicit signals — not a hardening failure.

    Hybrid pipeline still handles these; avoids false-positive production warnings.
    """
    signals = set(route.signals or ())
    if category == "capability":
        if "mixed_fact_and_capability" in signals:
            return True
        if route.secondary_intent == UnifiedSecondaryIntent.AIRCRAFT_MISSION_LIKELY:
            return True
    if category == "comparison":
        if "mixed_fact_and_capability" in signals:
            return True
        if route.secondary_intent == UnifiedSecondaryIntent.AIRCRAFT_COMPARISON_LIKELY:
            return True
        if route.execution_path == UnifiedExecutionPath.NONE and is_explicit_comparison_query(
            query or ""
        ):
            return True
    if category == "market" and "no_model" in signals:
        return True
    return False


def _path_satisfied(category: Optional[str], route: UnifiedIntentRoute) -> bool:
    if not category:
        return True
    path = route.execution_path
    mapping = {
        "fact": UnifiedExecutionPath.AIRCRAFT_FACT,
        "market": UnifiedExecutionPath.AIRCRAFT_MARKET_FACT,
        "capability": UnifiedExecutionPath.CAPABILITY,
        "comparison": UnifiedExecutionPath.COMPARISON,
        "alternative": UnifiedExecutionPath.ALTERNATIVE,
    }
    expected = mapping.get(category)
    if expected is None:
        return True
    return path == expected


def evaluate_hardening_guard(
    query: str,
    route: UnifiedIntentRoute,
    *,
    ambiguity_report: Optional[AmbiguityReport] = None,
) -> HardeningGuardResult:
    """
    Validate that classified routes with clear intent signals resolved an execution_path.

    Annotates failures only — never changes ``route`` or gate behavior.
    """
    report = ambiguity_report or classify_ambiguity(query or "", route)
    category = _expected_path_category(query or "", route)
    satisfied = _path_satisfied(category, route)

    documented_deferral = _router_documented_deferral(category, route, query=query or "")
    routing_failure = bool(category) and not satisfied and not documented_deferral
    hardening_reason: Optional[str] = None
    event_code: Optional[str] = None
    requires_fallback = routing_failure or (
        route.execution_path == UnifiedExecutionPath.NONE and report.is_ambiguous
    )

    if documented_deferral and category and not satisfied:
        logger.info(
            "HARDENING_ROUTING_DEFERRED query=%r category=%s path=%s signals=%s",
            (query or "")[:120],
            category,
            route.execution_path.value,
            list(route.signals or ()),
        )
    elif routing_failure:
        event_code = HARDENING_ROUTING_FAILURE
        hardening_reason = (
            f"Expected execution_path for {category} query but router returned "
            f"'{route.execution_path.value}'."
        )
        logger.warning(
            "%s query=%r expected=%s actual=%s reason=%s",
            HARDENING_ROUTING_FAILURE,
            (query or "")[:120],
            category,
            route.execution_path.value,
            hardening_reason,
        )

    capability_without_model = (
        category == "capability"
        and _has_capability_or_route_signals((query or "").lower())
        and not route.model
    )

    record_hardening_event(
        route,
        report,
        routing_failure=routing_failure,
        requires_fallback_analysis=requires_fallback,
        capability_without_model=capability_without_model,
    )

    return HardeningGuardResult(
        routing_failure=routing_failure,
        hardening_reason=hardening_reason,
        requires_fallback_analysis=requires_fallback,
        event_code=event_code,
        expected_path_category=category,
        ambiguity_report=report,
    )


def attach_hardening_layer(
    data_used: Dict[str, Any],
    *,
    query: str,
    route: UnifiedIntentRoute,
    qri_intent: str,
    enforce_fact: bool = False,
) -> HardeningGuardResult:
    """
    Run ambiguity classification + hardening guard and merge into data_used/shadow.

    Passive only — does not block pipeline gate or handler execution.
    """
    result = evaluate_hardening_guard(query or "", route)

    data_used["unified_intent_hardening"] = result.to_dict()
    data_used["unified_intent_production_metrics"] = get_production_metrics()

    shadow = data_used.get("unified_intent_shadow")
    if isinstance(shadow, dict):
        shadow["hardening_flags"] = result.hardening_flags
        data_used["unified_intent_shadow"] = shadow
    else:
        data_used["unified_intent_shadow"] = build_unified_intent_shadow(
            route,
            qri_intent,
            enforce_fact=enforce_fact,
            hardening_flags=result.hardening_flags,
        )

    return result


__all__ = [
    "HARDENING_ROUTING_FAILURE",
    "HardeningGuardResult",
    "attach_hardening_layer",
    "evaluate_hardening_guard",
]
