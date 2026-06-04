"""Estimate buyer vs seller leverage from liquidity, inventory, and temporal read."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from services.market_intelligence.liquidity_scoring import LiquidityBand, LiquidityScore
from services.market_intelligence.listing_analytics import MarketSnapshot
from services.market_reality.inventory_pressure_detector import InventoryPressure


class BuyerLeverage(str, Enum):
    BUYER_FRIENDLY = "BUYER_FRIENDLY"
    BALANCED = "BALANCED"
    SELLER_FRIENDLY = "SELLER_FRIENDLY"


def analyze_buyer_leverage(
    *,
    liquidity: Optional[LiquidityScore],
    inventory_pressure: str,
    temporal_timing: Optional[str] = None,
    price_confidence: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deterministic leverage read — does not alter liquidity or temporal engines.
    """
    score = 0  # positive = buyer friendly

    if liquidity is not None:
        if liquidity.band in (LiquidityBand.THIN, LiquidityBand.MODERATE):
            score += 1
        elif liquidity.band == LiquidityBand.HIGH:
            score -= 1

    if inventory_pressure == InventoryPressure.OVERSUPPLY.value:
        score += 2
    elif inventory_pressure == InventoryPressure.LOW_INVENTORY.value:
        score -= 2

    if temporal_timing in ("LATE_CYCLE", "MID_CYCLE"):
        score += 1
    elif temporal_timing == "EARLY_CYCLE":
        score -= 1

    if price_confidence in ("UNUSUALLY_CHEAP",):
        score += 1
    elif price_confidence in ("UNUSUALLY_EXPENSIVE",):
        score -= 1

    if score >= 2:
        leverage = BuyerLeverage.BUYER_FRIENDLY
        summary = "Buyers have reasonable negotiating room — inventory and liquidity support pushing on price."
    elif score <= -2:
        leverage = BuyerLeverage.SELLER_FRIENDLY
        summary = "Sellers are holding firm — limited inventory or strong demand limits discounting."
    else:
        leverage = BuyerLeverage.BALANCED
        summary = "Balanced market — price will hinge on tail-specific condition and motivation."

    return {
        "leverage": leverage.value,
        "summary": summary,
        "score": score,
    }


def leverage_from_snapshot(snapshot: MarketSnapshot, liquidity: LiquidityScore) -> Dict[str, Any]:
    from services.market_reality.inventory_pressure_detector import detect_inventory_pressure

    inv = detect_inventory_pressure(snapshot)
    return analyze_buyer_leverage(
        liquidity=liquidity,
        inventory_pressure=inv["pressure"],
    )


__all__ = ["BuyerLeverage", "analyze_buyer_leverage", "leverage_from_snapshot"]
