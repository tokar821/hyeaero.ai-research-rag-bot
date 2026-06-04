"""Deterministic ambiguity detection and canonicalization before reasoning."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from services.broker_reasoning.comparison_soft_resolution import soft_resolve_comparison
from services.broker_reasoning.intent_expander import _detect_manufacturer, _detect_reference_model
from services.intent_collapse.canonical_intent_frame import (
    AircraftScopeType,
    PrimaryIntent,
)


@dataclass
class AmbiguityResolution:
    flags: List[str] = field(default_factory=list)
    clarification_request: Optional[str] = None
    resolved_models: List[str] = field(default_factory=list)
    comparison_action: str = "none"
    confidence_penalty: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flags": list(self.flags),
            "clarification_request": self.clarification_request,
            "resolved_models": list(self.resolved_models),
            "comparison_action": self.comparison_action,
            "confidence_penalty": self.confidence_penalty,
        }


_CHEAP_GULFSTREAM_RE = re.compile(r"(?is)\b(?:cheap|cheapest|affordable)\b.*\bgulfstream\b|\bgulfstream\b.*\b(?:cheap|cheapest)\b")
_VS_RE = re.compile(r"\b(?:vs\.?|versus)\b", re.I)
_VALUATION_RE = re.compile(
    r"(?is)\b(?:worth|valuation|value\s+of|apprais|good\s+deal|fair\s+price|overpay|realistic)\b",
)
_LISTING_RE = re.compile(r"(?is)\b(?:saw|found|listing|asking|for\s+\d+\s*m)\b")


def resolve_ambiguity(
    query: str,
    *,
    primary_intent: str,
    adversarial: Optional[Dict[str, Any]] = None,
    budget_cap_musd: Optional[float] = None,
) -> AmbiguityResolution:
    """
  Resolve comparison pairs and shorthand; emit flags or clarification before reasoning.
    """
    q = (query or "").strip()
    adv = adversarial if isinstance(adversarial, dict) else {}
    flags: List[str] = []
    penalty = 0.0
    clarification: Optional[str] = None
    resolved: List[str] = []

    adv_models = []
    for m in adv.get("resolved_models") or []:
        if isinstance(m, dict) and m.get("canonical_model"):
            adv_models.append(str(m["canonical_model"]))

    if _CHEAP_GULFSTREAM_RE.search(q):
        flags.append("PRICING_UNCLEAR")
        flags.append("ENTRY_LEVEL_SCOPE_REQUIRED")
        penalty += 0.05

    if budget_cap_musd is None and primary_intent in (
        PrimaryIntent.BUY.value,
        PrimaryIntent.DISCOVERY.value,
    ):
        if re.search(r"(?is)\b(?:cheap|budget|affordable|under|below)\b", q):
            flags.append("BUDGET_VAGUE")
            penalty += 0.1
        elif re.search(r"(?is)\bwhat\s+should\s+i\s+buy\b", q):
            flags.append("MISSING_BUDGET_IN_QUERY")
            penalty += 0.15

    comparison_action = "none"
    if primary_intent == PrimaryIntent.COMPARE.value:
        soft = soft_resolve_comparison(q)
        if soft is not None:
            comparison_action = soft.action
            resolved = list(soft.models)
            if soft.action == "clarify":
                flags.append("COMPARISON_AMBIGUOUS")
                penalty += 0.25
                clarification = (
                    "I need the exact models to compare - say both names or use verified shorthand "
                    "(e.g. Longitude vs Praetor 600)."
                )
            elif soft.action == "auto_with_note":
                flags.append("COMPARISON_SOFT_RESOLVED")
            else:
                flags.append("COMPARISON_RESOLVED")
        else:
            flags.append("COMPARISON_AMBIGUOUS")
            penalty += 0.3
            clarification = "Which two aircraft should I compare?"

        if budget_cap_musd is not None:
            flags.append("BUDGET_CONSTRAINT_ON_COMPARE")

    if primary_intent == PrimaryIntent.VALUATION.value:
        ref = _detect_reference_model(q)
        if not ref and not adv_models:
            flags.append("MISSING_AIRCRAFT_MODEL")
            penalty += 0.2
            clarification = "Which aircraft (and ideally year or ask) should I value?"

    if primary_intent == PrimaryIntent.BUY.value and not _VS_RE.search(q):
        ref = _detect_reference_model(q)
        if not ref and not adv_models and not _detect_manufacturer(q):
            if _LISTING_RE.search(q):
                flags.append("LISTING_MODEL_UNCLEAR")
            elif re.search(r"(?is)\b(?:buy|get|find)\b", q):
                flags.append("MISSING_AIRCRAFT_MODEL")

    if adv.get("budget_feasibility") == "INFEASIBLE":
        flags.append("ADVERSARIAL_BUDGET_INFEASIBLE")

    return AmbiguityResolution(
        flags=flags,
        clarification_request=clarification,
        resolved_models=resolved or adv_models[:2],
        comparison_action=comparison_action,
        confidence_penalty=penalty,
    )


__all__ = ["AmbiguityResolution", "resolve_ambiguity"]
