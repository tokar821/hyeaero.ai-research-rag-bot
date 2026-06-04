"""Single-authority contract — each truth domain owned by exactly one layer."""

from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet, Optional


class TruthDomain(str, Enum):
    INTENT = "INTENT"
    CONSTRAINTS = "CONSTRAINTS"
    EVALUATION = "EVALUATION"
    RECOMMENDATION = "RECOMMENDATION"


# Layer name → domains it may *decide* (not merely annotate).
LAYER_AUTHORITY: Dict[str, FrozenSet[TruthDomain]] = {
    "intent_collapse": frozenset({TruthDomain.INTENT}),
    "adversarial": frozenset({TruthDomain.CONSTRAINTS}),
    "market_reality": frozenset({TruthDomain.CONSTRAINTS}),
    "broker_reasoning": frozenset({TruthDomain.EVALUATION}),
    "broker_decision": frozenset({TruthDomain.EVALUATION}),
    "executive_broker": frozenset({TruthDomain.RECOMMENDATION}),
    "truth_compression": frozenset(),  # presentation only
    "conversation": frozenset(),  # tone only
}


DOMAIN_OWNER: Dict[TruthDomain, str] = {
    TruthDomain.INTENT: "intent_collapse",
    TruthDomain.CONSTRAINTS: "adversarial + market_reality",
    TruthDomain.EVALUATION: "broker_reasoning",
    TruthDomain.RECOMMENDATION: "executive_broker",
}


# Phrases only the owning layer may introduce in client-facing prose.
FORBIDDEN_PHRASES_BY_LAYER: Dict[str, FrozenSet[str]] = {
    "broker_reasoning": frozenset({"my primary recommendation would be"}),
    "broker_decision": frozenset({"my primary recommendation would be"}),
    "market_reality": frozenset({"my primary recommendation would be"}),
    "client_context": frozenset({"my primary recommendation would be"}),
}


def owner_for_domain(domain: TruthDomain) -> str:
    return DOMAIN_OWNER.get(domain, "unknown")


def layer_may_decide(layer: str, domain: TruthDomain) -> bool:
    return domain in LAYER_AUTHORITY.get(layer, frozenset())


def violating_layer_for_phrase(text: str, *, speaking_layer: str) -> Optional[str]:
    """Return layer name if ``speaking_layer`` used another layer's exclusive phrasing."""
    low = (text or "").lower()
    if speaking_layer == "executive_broker":
        return None
    for phrase in FORBIDDEN_PHRASES_BY_LAYER.get(speaking_layer, frozenset()):
        if phrase in low:
            return "executive_broker"
    return None


__all__ = [
    "DOMAIN_OWNER",
    "FORBIDDEN_PHRASES_BY_LAYER",
    "LAYER_AUTHORITY",
    "TruthDomain",
    "layer_may_decide",
    "owner_for_domain",
    "violating_layer_for_phrase",
]
