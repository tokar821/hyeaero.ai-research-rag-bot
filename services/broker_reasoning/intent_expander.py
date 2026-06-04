"""Expand conversational requests into broker-understandable intent structures."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class IntentCategory(str, Enum):
    MANUFACTURER_FAMILY = "manufacturer_family"
    REFERENCE_MODEL = "reference_model"
    EXPLICIT_MODEL = "explicit_model"
    MISSION_BUDGET = "mission_budget"
    COMPARISON = "comparison"
    ACQUISITION = "acquisition"
    VALUATION = "valuation"
    UNKNOWN = "unknown"


@dataclass
class ExpandedIntent:
    category: IntentCategory
    manufacturer: Optional[str] = None
    reference_model: Optional[str] = None
    acquisition_focus: bool = False
    price_sensitivity: str = "normal"  # low | normal | high
    budget_sensitive: bool = False
    alternative_search: bool = False
    constraint: Optional[str] = None
    intent_hint: Optional[str] = None
    raw_signals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "manufacturer": self.manufacturer,
            "reference_model": self.reference_model,
            "acquisition_focus": self.acquisition_focus,
            "price_sensitivity": self.price_sensitivity,
            "budget_sensitive": self.budget_sensitive,
            "alternative_search": self.alternative_search,
            "constraint": self.constraint,
            "intent_hint": self.intent_hint,
            "raw_signals": list(self.raw_signals),
        }


_CHEAP_RE = re.compile(r"\b(?:cheap|cheapest|affordable|budget|lowest[- ]cost|inexpensive)\b", re.I)
_BUDGET_RE = re.compile(r"\b(?:budget|under|below|around|about|for)\s+\$?\s*\d", re.I)
_BUY_RE = re.compile(r"\b(?:buy|purchase|acquire|get|should\s+i\s+buy|what\s+should\s+i\s+buy)\b", re.I)
_LIKE_CHEAPER_RE = re.compile(
    r"(?is)\b(?:something\s+)?like\s+(?:a\s+)?(?P<ref>.+?)\s+but\s+cheaper\b",
)
_UNDER_BUDGET_RE = re.compile(
    r"(?is)\b(?:like\s+(?:a\s+)?(?P<ref2>.+?)\s+)?under\s+\$?\s*(?P<amt>\d+(?:\.\d+)?)\s*(?P<unit>m|mm|million|mil)?\b",
)
_MANUFACTURER_PATTERNS: Dict[str, re.Pattern[str]] = {
    "Gulfstream": re.compile(r"\b(?:gulfstream|g\s*\d{3})\b", re.I),
    "Dassault": re.compile(r"\b(?:dassault|falcon)\b", re.I),
    "Bombardier": re.compile(r"\b(?:bombardier|challenger|global|learjet)\b", re.I),
    "Cessna": re.compile(r"\b(?:citation|cessna|latitude|longitude)\b", re.I),
    "Embraer": re.compile(r"\b(?:embraer|phenom|praetor|legacy)\b", re.I),
}

_REFERENCE_MODEL_PATTERNS: List[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\blongitude\b", re.I), "Citation Longitude"),
    (re.compile(r"\blatitude\b", re.I), "Citation Latitude"),
    (re.compile(r"\bg650\b|\bgulfstream\s+g650\b", re.I), "Gulfstream G650"),
    (re.compile(r"\bg700\b|\bgulfstream\s+g700\b", re.I), "Gulfstream G700"),
    (re.compile(r"\bchallenger\s*350\b", re.I), "Challenger 350"),
    (re.compile(r"\bchallenger\b", re.I), "Challenger 350"),
    (re.compile(r"\bphenom\s*300\b|\bphenom\b", re.I), "Phenom 300"),
    (re.compile(r"\bfalcon\s*8x\b", re.I), "Falcon 8X"),
    (re.compile(r"\bfalcon\b", re.I), "Falcon 2000"),
]


def _detect_manufacturer(query: str) -> Optional[str]:
    for name, pat in _MANUFACTURER_PATTERNS.items():
        if pat.search(query or ""):
            return name
    return None


def _detect_reference_model(query: str) -> Optional[str]:
    q = query or ""
    for pat, model in _REFERENCE_MODEL_PATTERNS:
        if pat.search(q):
            return model
    return None


def expand_intent(query: str, *, data_used: Optional[Dict[str, Any]] = None) -> ExpandedIntent:
    """Expand a conversational query into structured broker intent."""
    del data_used
    q = (query or "").strip()
    low = q.lower()
    signals: List[str] = []

    cheap = bool(_CHEAP_RE.search(q))
    budget = bool(_BUDGET_RE.search(q))
    buy = bool(_BUY_RE.search(q))
    manufacturer = _detect_manufacturer(q)
    reference = _detect_reference_model(q)

    m_like = _LIKE_CHEAPER_RE.search(q)
    if m_like:
        ref_raw = (m_like.group("ref") or "").strip()
        ref_model = _detect_reference_model(ref_raw) or ref_raw.title()
        signals.append("like_cheaper")
        return ExpandedIntent(
            category=IntentCategory.REFERENCE_MODEL,
            reference_model=ref_model,
            acquisition_focus=True,
            price_sensitivity="high",
            budget_sensitive=True,
            alternative_search=True,
            constraint="lower_acquisition_cost",
            intent_hint="alternative_search",
            raw_signals=signals,
        )

    m_under = _UNDER_BUDGET_RE.search(q)
    if m_under and m_under.group("ref2"):
        ref_model = _detect_reference_model(m_under.group("ref2") or "") or reference
        signals.append("reference_under_budget")
        return ExpandedIntent(
            category=IntentCategory.REFERENCE_MODEL,
            reference_model=ref_model,
            acquisition_focus=True,
            budget_sensitive=True,
            price_sensitivity="high",
            constraint="budget_cap",
            intent_hint="alternative_search",
            raw_signals=signals,
        )

    if cheap and manufacturer == "Gulfstream":
        signals.append("cheap_gulfstream")
        return ExpandedIntent(
            category=IntentCategory.MANUFACTURER_FAMILY,
            manufacturer="Gulfstream",
            acquisition_focus=True,
            price_sensitivity="high",
            budget_sensitive=budget,
            intent_hint="manufacturer_discovery",
            raw_signals=signals,
        )

    if budget and manufacturer:
        signals.append("budget_manufacturer")
        return ExpandedIntent(
            category=IntentCategory.MANUFACTURER_FAMILY,
            manufacturer=manufacturer,
            acquisition_focus=True,
            budget_sensitive=True,
            price_sensitivity="high" if cheap else "normal",
            intent_hint="manufacturer_discovery",
            raw_signals=signals,
        )

    if re.search(r"\b(?:best|what)\s+(?:jet|aircraft)\b", low) and budget:
        signals.append("mission_budget")
        return ExpandedIntent(
            category=IntentCategory.MISSION_BUDGET,
            acquisition_focus=True,
            budget_sensitive=True,
            intent_hint="budget_discovery",
            raw_signals=signals,
        )

    if buy and reference:
        signals.append("buy_reference")
        return ExpandedIntent(
            category=IntentCategory.ACQUISITION,
            reference_model=reference,
            acquisition_focus=True,
            budget_sensitive=budget,
            raw_signals=signals,
        )

    if re.search(r"\b(?:vs\.?|versus|compare)\b", low):
        signals.append("comparison")
        return ExpandedIntent(
            category=IntentCategory.COMPARISON,
            intent_hint="comparison",
            raw_signals=signals,
        )

    if re.search(r"\b(?:worth|valuation|value\s+of|how\s+much)\b", low):
        signals.append("valuation")
        return ExpandedIntent(
            category=IntentCategory.VALUATION,
            reference_model=reference,
            intent_hint="valuation",
            raw_signals=signals,
        )

    if manufacturer and (cheap or budget):
        return ExpandedIntent(
            category=IntentCategory.MANUFACTURER_FAMILY,
            manufacturer=manufacturer,
            acquisition_focus=True,
            price_sensitivity="high" if cheap else "normal",
            budget_sensitive=budget,
            raw_signals=signals,
        )

    return ExpandedIntent(
        category=IntentCategory.UNKNOWN,
        manufacturer=manufacturer,
        reference_model=reference,
        acquisition_focus=buy,
        budget_sensitive=budget,
        price_sensitivity="high" if cheap else "normal",
        raw_signals=signals,
    )


__all__ = ["ExpandedIntent", "IntentCategory", "expand_intent"]
