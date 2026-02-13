"""Market comparison: query listings by model, age, hours, region from Hye Aero data."""

import logging
from typing import List, Dict, Any, Optional

from database.postgres_client import PostgresClient

logger = logging.getLogger(__name__)

# Region dropdown value -> search terms for location/based_at/country (e.g. "Europe" includes UK, England, London)
REGION_SEARCH_TERMS: Dict[str, List[str]] = {
    "europe": [
        "europe", "eu", "uk", "united kingdom", "england", "scotland", "wales", "london",
        "france", "germany", "spain", "italy", "netherlands", "belgium", "switzerland",
        "austria", "ireland", "portugal", "greece", "poland", "sweden", "norway", "denmark",
        "finland", "czech", "hungary", "romania", "bulgaria", "croatia", "slovakia", "slovenia",
        "luxembourg", "malta", "cyprus", "estonia", "latvia", "lithuania", "iceland",
    ],
    "north america": [
        "north america", "usa", "united states", "u.s.", "us ", " u.s ", "america",
        "canada", "mexico", "california", "texas", "florida", "new york", "nevada", "arizona",
        "georgia", "illinois", "ohio", "colorado", "washington", "ontario", "quebec",
    ],
    "asia pacific": [
        "asia", "pacific", "australia", "japan", "china", "singapore", "hong kong",
        "uae", "dubai", "india", "south korea", "new zealand", "thailand", "malaysia",
        "indonesia", "philippines", "vietnam", "taiwan",
    ],
}


def run_comparison(
    db: PostgresClient,
    models: List[str],
    region: Optional[str] = None,
    max_hours: Optional[float] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Compare aircraft listings by model(s), optional region, max hours, year range.
    Returns side-by-side style rows from aircraft_listings + aircraft (manufacturer, model, year).
    """
    if not models:
        return {"rows": [], "summary": "No models selected.", "error": None}

    # Build WHERE: match any of the models (manufacturer + model from aircraft)
    # aircraft_listings joins aircraft on aircraft_id; aircraft has manufacturer, model, manufacturer_year
    conditions = []
    params: list = []

    # Model filter: e.g. "Phenom 300" -> match aircraft.model ILIKE %Phenom% AND aircraft.model ILIKE %300%
    model_conditions = []
    for m in models:
        m = (m or "").strip()
        if not m:
            continue
        # Simple: match model string in aircraft.model (or manufacturer || ' ' || model)
        model_conditions.append("(a.manufacturer ILIKE %s OR a.model ILIKE %s OR (a.manufacturer || ' ' || a.model) ILIKE %s)")
        like = f"%{m}%"
        params.extend([like, like, like])
    if model_conditions:
        conditions.append("(" + " OR ".join(model_conditions) + ")")

    if region and region.lower() != "global":
        terms = REGION_SEARCH_TERMS.get(region.lower().strip())
        if terms:
            # Match if any location field contains any of the region's search terms (e.g. Europe -> UK, England, London, ...)
            placeholders = []
            for _ in terms:
                placeholders.append("(l.location ILIKE %s OR l.based_at ILIKE %s OR a.based_country ILIKE %s OR a.registration_country ILIKE %s)")
            conditions.append("(" + " OR ".join(placeholders) + ")")
            for t in terms:
                r = f"%{t}%"
                params.extend([r, r, r, r])
        else:
            # Fallback: single term (region name as-is)
            conditions.append("(l.location ILIKE %s OR l.based_at ILIKE %s OR a.based_country ILIKE %s OR a.registration_country ILIKE %s)")
            r = f"%{region}%"
            params.extend([r, r, r, r])

    if max_hours is not None and max_hours > 0:
        conditions.append("(l.airframe_total_time IS NULL OR l.airframe_total_time <= %s)")
        params.append(max_hours)

    if min_year is not None:
        conditions.append("a.manufacturer_year >= %s")
        params.append(min_year)
    if max_year is not None:
        conditions.append("a.manufacturer_year <= %s")
        params.append(max_year)

    where_sql = " AND ".join(conditions) if conditions else "1=1"
    params.append(limit)

    query = f"""
        SELECT
            l.id AS listing_id,
            l.source_platform,
            l.listing_status,
            l.ask_price,
            l.sold_price,
            l.airframe_total_time,
            l.airframe_total_cycles,
            l.location,
            l.based_at,
            l.days_on_market,
            l.avionics_description,
            a.manufacturer,
            a.model,
            a.manufacturer_year
        FROM aircraft_listings l
        LEFT JOIN aircraft a ON l.aircraft_id = a.id
        WHERE {where_sql}
        ORDER BY l.updated_at DESC NULLS LAST, l.created_at DESC
        LIMIT %s
    """
    try:
        rows = db.execute_query(query, tuple(params))
        # Serialize decimals/dates for JSON
        out = []
        for r in rows:
            row = dict(r)
            for k, v in row.items():
                if hasattr(v, "isoformat"):
                    row[k] = v.isoformat()
                elif hasattr(v, "__float__") and not isinstance(v, (int, bool)):
                    try:
                        row[k] = float(v)
                    except (TypeError, ValueError):
                        pass
            out.append(row)
        summary = f"Found {len(out)} comparable listings."
        return {"rows": out, "summary": summary, "error": None}
    except Exception as e:
        logger.exception("Market comparison query failed")
        return {"rows": [], "summary": "", "error": str(e)}
