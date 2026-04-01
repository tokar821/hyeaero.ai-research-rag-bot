"""Intent enums and classification result (shared types)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class AviationIntent(str, Enum):
    """Fine-grained Ask Consultant / aviation intents (API-friendly string values)."""

    AIRCRAFT_COMPARISON = "aircraft_comparison"
    AIRCRAFT_SPECS = "aircraft_specs"
    MISSION_FEASIBILITY = "mission_feasibility"
    REGISTRATION_LOOKUP = "registration_lookup"
    SERIAL_LOOKUP = "serial_lookup"
    AIRCRAFT_FOR_SALE = "aircraft_for_sale"
    MARKET_PRICE = "market_price"
    OPERATOR_LOOKUP = "operator_lookup"
    GENERAL_QUESTION = "general_question"


class ConsultantIntent(str, Enum):
    """Coarse routing bucket (retrieval ranking, legacy policy hooks)."""

    REGISTRATION_LOOKUP = "registration_lookup"
    AIRCRAFT_IDENTITY = "aircraft_identity"
    MARKET_PRICING = "market_pricing"
    TECHNICAL_SPEC = "technical_spec"
    GENERAL_AVIATION = "general_aviation"
    UNKNOWN = "unknown"


class ConsultantQueryKind(str, Enum):
    """High-level query type for routing, logging, and prompt shaping (before / alongside RAG)."""

    MISSION = "mission"
    SPEC = "spec"
    COMPARISON = "comparison"
    OWNERSHIP = "ownership"
    MARKET = "market"
    LISTINGS = "listings"
    GENERAL = "general"


def query_kind_from_aviation_intent(
    av: Optional[AviationIntent],
) -> ConsultantQueryKind:
    """Map fine aviation intent to user-facing query class (mission / spec / …)."""
    if av is None:
        return ConsultantQueryKind.GENERAL
    if av == AviationIntent.MISSION_FEASIBILITY:
        return ConsultantQueryKind.MISSION
    if av == AviationIntent.AIRCRAFT_SPECS:
        return ConsultantQueryKind.SPEC
    if av == AviationIntent.AIRCRAFT_COMPARISON:
        return ConsultantQueryKind.COMPARISON
    if av in (
        AviationIntent.REGISTRATION_LOOKUP,
        AviationIntent.SERIAL_LOOKUP,
        AviationIntent.OPERATOR_LOOKUP,
    ):
        return ConsultantQueryKind.OWNERSHIP
    if av == AviationIntent.MARKET_PRICE:
        return ConsultantQueryKind.MARKET
    if av == AviationIntent.AIRCRAFT_FOR_SALE:
        return ConsultantQueryKind.LISTINGS
    return ConsultantQueryKind.GENERAL


@dataclass
class IntentClassification:
    """Output of :func:`rag.intent.classifier.classify_consultant_intent`."""

    primary: ConsultantIntent
    source: str
    confidence: float = 1.0
    notes: str = ""
    aviation_intent: Optional[AviationIntent] = None
    query_kind: ConsultantQueryKind = ConsultantQueryKind.GENERAL

    def asdict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "primary": self.primary.value,
            "source": self.source,
            "confidence": self.confidence,
            "notes": self.notes,
            "intent": self.aviation_intent.value if self.aviation_intent else self.primary.value,
            "query_kind": self.query_kind.value,
        }
        if self.aviation_intent is not None:
            d["aviation_intent"] = self.aviation_intent.value
        return d
