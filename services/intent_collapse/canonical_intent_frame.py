"""Canonical broker intent — single pre-reasoning interpretation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from services.intent_collapse.mission_frame_builder import MissionFrame


class PrimaryIntent(str, Enum):
    BUY = "BUY"
    COMPARE = "COMPARE"
    VALUATION = "VALUATION"
    DISCOVERY = "DISCOVERY"


class AircraftScopeType(str, Enum):
    OPEN = "OPEN"
    ENTRY_LEVEL_GULFSTREAM_SCOPE = "ENTRY_LEVEL_GULFSTREAM_SCOPE"
    MANUFACTURER_FAMILY = "MANUFACTURER_FAMILY"
    EXPLICIT_MODELS = "EXPLICIT_MODELS"
    COMPARISON_PAIR = "COMPARISON_PAIR"


@dataclass
class BudgetFrame:
    min_musd: Optional[float] = None
    max_musd: Optional[float] = None
    cap_musd: Optional[float] = None
    tier_hint: str = "UNKNOWN"
    unknown: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_musd": self.min_musd,
            "max_musd": self.max_musd,
            "cap_musd": self.cap_musd,
            "tier_hint": self.tier_hint,
            "unknown": self.unknown,
        }


@dataclass
class AircraftScopeFrame:
    scope_type: str
    manufacturer: Optional[str] = None
    models: List[str] = field(default_factory=list)
    price_sensitive: bool = False
    entry_level_only: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope_type": self.scope_type,
            "manufacturer": self.manufacturer,
            "models": list(self.models),
            "price_sensitive": self.price_sensitive,
            "entry_level_only": self.entry_level_only,
        }


@dataclass
class CanonicalIntentFrame:
    primary_intent: str
    mission: MissionFrame
    budget: BudgetFrame
    aircraft_scope: AircraftScopeFrame
    confidence: float
    ambiguity_flags: List[str] = field(default_factory=list)
    clarification_request: Optional[str] = None
    normalized_query: str = ""
    raw_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_intent": self.primary_intent,
            "mission": self.mission.to_dict(),
            "budget": self.budget.to_dict(),
            "aircraft_scope": self.aircraft_scope.to_dict(),
            "confidence": self.confidence,
            "ambiguity_flags": list(self.ambiguity_flags),
            "clarification_request": self.clarification_request,
            "normalized_query": self.normalized_query,
            "raw_query": self.raw_query,
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> Optional[CanonicalIntentFrame]:
        if not isinstance(raw, dict):
            return None
        mission = MissionFrame.from_dict(raw.get("mission"))
        budget_raw = raw.get("budget") or {}
        scope_raw = raw.get("aircraft_scope") or {}
        return cls(
            primary_intent=str(raw.get("primary_intent") or PrimaryIntent.DISCOVERY.value),
            mission=mission or MissionFrame(),
            budget=BudgetFrame(
                min_musd=budget_raw.get("min_musd"),
                max_musd=budget_raw.get("max_musd"),
                cap_musd=budget_raw.get("cap_musd"),
                tier_hint=str(budget_raw.get("tier_hint") or "UNKNOWN"),
                unknown=bool(budget_raw.get("unknown", True)),
            ),
            aircraft_scope=AircraftScopeFrame(
                scope_type=str(scope_raw.get("scope_type") or AircraftScopeType.OPEN.value),
                manufacturer=scope_raw.get("manufacturer"),
                models=list(scope_raw.get("models") or []),
                price_sensitive=bool(scope_raw.get("price_sensitive")),
                entry_level_only=bool(scope_raw.get("entry_level_only")),
            ),
            confidence=float(raw.get("confidence") or 0.5),
            ambiguity_flags=list(raw.get("ambiguity_flags") or []),
            clarification_request=raw.get("clarification_request"),
            normalized_query=str(raw.get("normalized_query") or ""),
            raw_query=str(raw.get("raw_query") or ""),
        )


__all__ = [
    "AircraftScopeFrame",
    "AircraftScopeType",
    "BudgetFrame",
    "CanonicalIntentFrame",
    "PrimaryIntent",
]
