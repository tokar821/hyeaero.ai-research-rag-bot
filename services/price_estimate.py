"""Predictive pricing: estimate fair market value and time-to-sale from historical data.

Uses Hye Aero's sale history (aircraft_sales) and listing data. Placeholder returns
heuristic-based estimate; can be replaced with regression/sentiment model later.
"""

import logging
from typing import Dict, Any, Optional

from database.postgres_client import PostgresClient

logger = logging.getLogger(__name__)


def estimate_value(
    db: PostgresClient,
    manufacturer: Optional[str] = None,
    model: Optional[str] = None,
    year: Optional[int] = None,
    flight_hours: Optional[float] = None,
    flight_cycles: Optional[int] = None,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return estimated market value, range, confidence, and time-to-sale placeholder.
    """
    try:
        conditions = ["1=1"]
        params: list = []
        if manufacturer:
            conditions.append("manufacturer ILIKE %s")
            params.append(f"%{manufacturer}%")
        if model:
            conditions.append("model ILIKE %s")
            params.append(f"%{model}%")
        if year is not None:
            conditions.append("manufacturer_year = %s")
            params.append(year)
        if region and region.lower() != "global":
            conditions.append("(based_country ILIKE %s OR registration_country ILIKE %s)")
            params.append(f"%{region}%")
            params.append(f"%{region}%")

        where = " AND ".join(conditions)
        params.append(50)

        # Recent sales for same model/year band
        sales_query = f"""
            SELECT sold_price, ask_price, days_on_market, manufacturer_year, airframe_total_time
            FROM aircraft_sales
            WHERE {where}
            AND sold_price IS NOT NULL AND sold_price > 0
            ORDER BY date_sold DESC NULLS LAST
            LIMIT %s
        """
        sales = db.execute_query(sales_query, tuple(params))

        if not sales:
            return {
                "estimated_value_millions": None,
                "range_low_millions": None,
                "range_high_millions": None,
                "confidence_pct": 0,
                "market_demand": "Unknown",
                "vs_average_pct": None,
                "time_to_sale_days": None,
                "breakdown": [],
                "error": None,
                "message": "No comparable sales found in database. Add more historical data for accurate estimates.",
            }

        prices = [float(s["sold_price"]) for s in sales if s.get("sold_price")]
        if not prices:
            return {
                "estimated_value_millions": None,
                "range_low_millions": None,
                "range_high_millions": None,
                "confidence_pct": 0,
                "market_demand": "Unknown",
                "vs_average_pct": None,
                "time_to_sale_days": None,
                "breakdown": [],
                "error": None,
                "message": "No sale prices in comparables.",
            }

        avg_price = sum(prices) / len(prices)
        low = min(prices)
        high = max(prices)
        # Simple adjustment by flight_hours if provided
        if flight_hours is not None and flight_hours > 0:
            avg_hrs = sum(float(s.get("airframe_total_time") or 0) for s in sales) / max(len(sales), 1)
            if avg_hrs > 0:
                hr_factor = 1 - 0.02 * ((flight_hours - avg_hrs) / 1000)  # rough discount per 1000h
                hr_factor = max(0.7, min(1.2, hr_factor))
                avg_price *= hr_factor
                low *= 0.95
                high *= 1.05

        avg_m = round(avg_price / 1_000_000, 1)
        low_m = round(low / 1_000_000, 1)
        high_m = round(high / 1_000_000, 1)
        days_list = [s.get("days_on_market") for s in sales if s.get("days_on_market") is not None]
        avg_days = int(sum(days_list) / len(days_list)) if days_list else None

        return {
            "estimated_value_millions": avg_m,
            "range_low_millions": low_m,
            "range_high_millions": high_m,
            "confidence_pct": min(95, 70 + len(prices)),
            "market_demand": "High" if len(prices) >= 10 else "Moderate" if len(prices) >= 3 else "Low",
            "vs_average_pct": None,
            "time_to_sale_days": avg_days,
            "breakdown": [
                {"label": "Base (comparable sales)", "value_millions": round(avg_price / 1_000_000, 2)},
            ],
            "error": None,
            "message": f"Based on {len(prices)} comparable sale(s).",
        }
    except Exception as e:
        logger.exception("Price estimate failed")
        return {
            "estimated_value_millions": None,
            "range_low_millions": None,
            "range_high_millions": None,
            "confidence_pct": 0,
            "market_demand": "Unknown",
            "vs_average_pct": None,
            "time_to_sale_days": None,
            "breakdown": [],
            "error": str(e),
            "message": None,
        }
