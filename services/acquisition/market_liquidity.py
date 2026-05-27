"""Market liquidity signals for acquisition advisory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LiquidityAssessment:
    model: str
    liquidity_tier: str  # strong | moderate | thin | unknown
    days_on_market_hint: Optional[int]
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "liquidity_tier": self.liquidity_tier,
            "days_on_market_hint": self.days_on_market_hint,
            "notes": self.notes,
        }


_LIQUIDITY_HINTS = {
    "g650": ("strong", 90, "Active ULR market; serial-specific pricing varies."),
    "g650er": ("strong", 90, "Active ULR market; serial-specific pricing varies."),
    "global 7500": ("moderate", 120, "ULR segment; fewer listings than G650 family."),
    "challenger 350": ("strong", 75, "High super-mid transaction volume."),
    "citation latitude": ("strong", 70, "Strong midsize liquidity."),
}


def assess_market_liquidity(model: str, *, year: Optional[int] = None) -> LiquidityAssessment:
    key = (model or "").strip().lower()
    tier, dom, notes = _LIQUIDITY_HINTS.get(key, ("unknown", None, "Liquidity not cataloged."))
    if year and year < 2005:
        notes = f"{notes} Older vintage may extend marketing time."
    return LiquidityAssessment(model=model, liquidity_tier=tier, days_on_market_hint=dom, notes=notes)
