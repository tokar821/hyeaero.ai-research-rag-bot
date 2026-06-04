"""Aircraft market intelligence (Phase 24)."""

from services.market.aircraft_market_intelligence_engine import (
    MarketIntelligenceReport,
    analyze_inventory_trend,
    analyze_market_liquidity,
    analyze_price_trend,
    attach_market_intelligence_if_enabled,
    build_market_intelligence,
    evaluate_purchase_timing,
    evaluate_replacement_risk,
    evaluate_sale_timing,
    market_intelligence_enabled,
)

__all__ = [
    "MarketIntelligenceReport",
    "analyze_inventory_trend",
    "analyze_market_liquidity",
    "analyze_price_trend",
    "attach_market_intelligence_if_enabled",
    "build_market_intelligence",
    "evaluate_purchase_timing",
    "evaluate_replacement_risk",
    "evaluate_sale_timing",
    "market_intelligence_enabled",
]
