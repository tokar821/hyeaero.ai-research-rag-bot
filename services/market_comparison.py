"""Market comparison: query listings by model, age, hours, region from Hye Aero data."""

import logging
from typing import List, Dict, Any, Optional

from database.postgres_client import PostgresClient

logger = logging.getLogger(__name__)


def _build_comparison_consultant_summary(
    *,
    rows: List[Dict[str, Any]],
    models: List[str],
    region: str,
) -> str:
    """
    Human-style market snapshot (Part 6): lead with a verdict, then mission fit / tradeoffs / usage —
    not a raw spec dump. Numbers come only from ``rows`` (no invented pricing).
    """
    if not rows:
        return "No comparable listings matched these filters — widen region, year band, or model selection."

    n = len(rows)
    model_lbl = ", ".join(m.strip() for m in models if (m or "").strip()) or "selected models"
    asks: List[float] = []
    years: List[int] = []
    hours: List[float] = []
    for r in rows:
        ap = r.get("ask_price")
        if ap is not None:
            try:
                asks.append(float(ap))
            except (TypeError, ValueError):
                pass
        y = r.get("manufacturer_year")
        if y is not None:
            try:
                years.append(int(y))
            except (TypeError, ValueError):
                pass
        h = r.get("airframe_total_time")
        if h is not None:
            try:
                hours.append(float(h))
            except (TypeError, ValueError):
                pass

    verdict = (
        f"Across {n} current listing snapshot(s) for **{model_lbl}** in **{region}**, the market shows a spread of "
        f"asking levels and equipment — useful for a quick sanity check, not a final bid."
    )
    parts = [verdict]

    if asks:
        lo, hi = min(asks), max(asks)
        parts.append(
            f"**Asking range (synced listings only):** about **${lo:,.0f}** to **${hi:,.0f}** USD — verify each ad; "
            "status and equipment differ, and some rows may omit ask while still being instructive for positioning."
        )
    else:
        parts.append(
            "**Asking prices:** many rows have no stored ask in this ingest — treat pricing as **unknown** unless "
            "you open the individual listing."
        )

    if years:
        parts.append(
            f"**Vintage spread:** roughly **{min(years)}–{max(years)}** delivery years in this slice — newer metal "
            "usually trades mission flexibility (range/payload) against capital cost; older units can be strong buys "
            "when programs and documentation are tight."
        )

    if hours:
        parts.append(
            f"**Time on airframe:** about **{min(hours):,.0f}–{max(hours):,.0f}** hours among these rows — higher-time "
            "jets often imply different operating economics (reserves, upcoming checks); lower-time examples skew to "
            "premium pricing if pedigree is clean."
        )

    parts.append(
        "**How to read this:** use the table for *mission fit* (range/cabin class implied by model), *tradeoffs* "
        "(year vs price vs hours), and *real-world usage* (typical missions for the class). Follow up with a broker "
        "on specific serials before committing."
    )
    return " ".join(parts)


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
        summary = _build_comparison_consultant_summary(
            rows=out,
            models=models,
            region=region or "Global",
        )
        return {"rows": out, "summary": summary, "error": None}
    except Exception as e:
        logger.exception("Market comparison query failed")
        return {"rows": [], "summary": "", "error": str(e)}
