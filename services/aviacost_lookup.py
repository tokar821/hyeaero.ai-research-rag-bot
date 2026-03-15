"""Lookup aviacost_aircraft_details by manufacturer + model.

Used by Price Estimator, PhlyData Owner details, and other features that need
operating cost and pre-owned price reference data by aircraft type.
"""

import logging
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from database.postgres_client import PostgresClient

logger = logging.getLogger(__name__)


def lookup_aviacost(
    db: "PostgresClient",
    manufacturer: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Look up aviacost_aircraft_details by manufacturer and/or model.
    Returns a dict with variable_cost_per_hour, average_pre_owned_price, name, etc.
    Returns None if no match or if neither manufacturer nor model provided.
    """
    mfr = (manufacturer or "").strip()
    mdl = (model or "").strip()
    if not mfr and not mdl:
        return None

    # If only model (e.g. "Eclipse EA500"), try splitting for manufacturer
    if not mfr and mdl:
        parts = mdl.split(None, 1)
        if len(parts) >= 2:
            mfr, mdl = parts[0], parts[1]

    try:
        conditions = []
        params: list = []
        if mdl:
            conditions.append("name ILIKE %s")
            params.append(f"%{mdl}%")
        if mfr:
            conditions.append("(manufacturer_name ILIKE %s OR name ILIKE %s)")
            params.extend([f"%{mfr}%", f"%{mfr}%"])
        if not conditions:
            return None
        where = " AND ".join(conditions)
        params.append(1)
        q = f"""
            SELECT name, manufacturer_name, category_name,
                   variable_cost_per_hour, average_pre_owned_price,
                   fuel_gallons_per_hour, normal_cruise_speed_kts,
                   seats_full_range_nm, typical_passenger_capacity_max,
                   years_in_production
            FROM aviacost_aircraft_details
            WHERE {where}
            ORDER BY LENGTH(COALESCE(name,''))
            LIMIT %s
        """
        rows = db.execute_query(q, tuple(params))
        if not rows:
            return None
        row = dict(rows[0])

        # Format for API consumers
        out: Dict[str, Any] = {
            "name": row.get("name"),
            "manufacturer_name": row.get("manufacturer_name"),
            "category_name": row.get("category_name"),
            "variable_cost_per_hour": float(row["variable_cost_per_hour"]) if row.get("variable_cost_per_hour") is not None else None,
            "average_pre_owned_price": float(row["average_pre_owned_price"]) if row.get("average_pre_owned_price") is not None else None,
            "fuel_gallons_per_hour": float(row["fuel_gallons_per_hour"]) if row.get("fuel_gallons_per_hour") is not None else None,
            "normal_cruise_speed_kts": float(row["normal_cruise_speed_kts"]) if row.get("normal_cruise_speed_kts") is not None else None,
            "seats_full_range_nm": float(row["seats_full_range_nm"]) if row.get("seats_full_range_nm") is not None else None,
            "typical_passenger_capacity_max": row.get("typical_passenger_capacity_max"),
            "years_in_production": row.get("years_in_production"),
        }
        return out
    except Exception as e:
        logger.warning("aviacost lookup failed: %s", e)
        return None
