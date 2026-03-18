"""AircraftPost fleet lookup for enrichment.

Used by Price Estimator and other features that want fleet context (hours/landings/base/etc.)
by make/model (and optionally serial number).
"""

import logging
from typing import Dict, Any, Optional, TYPE_CHECKING, List

if TYPE_CHECKING:
    from database.postgres_client import PostgresClient

logger = logging.getLogger(__name__)


def lookup_aircraftpost_fleet(
    db: "PostgresClient",
    manufacturer: Optional[str] = None,
    model: Optional[str] = None,
    serial: Optional[str] = None,
    limit_matches: int = 3,
    max_summary_sample: int = 5000,
) -> Optional[Dict[str, Any]]:
    """
    Look up aircraftpost_fleet_aircraft by make_model_name (ILIKE manufacturer + model)
    and optionally by serial_number. Returns a dict with:
      - matches: top N recent rows (small sample for UI)
      - fleet_summary: counts + percentile stats for airframe_hours / total_landings

    Returns None if manufacturer+model are not provided.
    """
    mfr = (manufacturer or "").strip()
    mdl = (model or "").strip()
    ser = (serial or "").strip()

    # Price Estimator sends the dropdown value as a single string in `model`
    # (e.g. "Embraer Phenom 100") with manufacturer often empty. Split it.
    if not mfr and mdl:
        parts = mdl.split(None, 1)
        if len(parts) >= 2:
            mfr, mdl = parts[0].strip(), parts[1].strip()

    if not mfr or not mdl:
        return None

    like_mfr = f"%{mfr}%"
    like_mdl = f"%{mdl}%"

    where = ["make_model_name ILIKE %s", "make_model_name ILIKE %s"]
    params: List[Any] = [like_mfr, like_mdl]
    if ser:
        where.append("serial_number = %s")
        params.append(ser)
    where_sql = " AND ".join(where)

    try:
        # Small set of recent sample matches for UI
        matches = db.execute_query(
            f"""
            SELECT id, aircraft_entity_id, make_model_id, make_model_name,
                   serial_number, registration_number,
                   mfr_year, eis_date, country_code, base_code, owner_url,
                   airframe_hours, total_landings, prior_owners, for_sale, passengers,
                   engine_program_type, apu_program, ingestion_date
            FROM aircraftpost_fleet_aircraft
            WHERE {where_sql}
            ORDER BY ingestion_date DESC, updated_at DESC
            LIMIT %s
            """,
            tuple(params + [max(1, int(limit_matches))]),
        )

        # Summary stats (percentiles) for fleet context.
        # We limit sample size to avoid heavy scans on large tables; ordering by updated_at is a practical proxy.
        summary_rows = db.execute_query(
            f"""
            WITH filtered AS (
                SELECT airframe_hours, total_landings, base_code, country_code, for_sale
                FROM aircraftpost_fleet_aircraft
                WHERE {where_sql}
                ORDER BY updated_at DESC
                LIMIT %s
            )
            SELECT
                COUNT(*) AS total_records,
                COUNT(airframe_hours) AS records_with_hours,
                COUNT(total_landings) AS records_with_landings,
                AVG(airframe_hours)::float AS avg_airframe_hours,
                AVG(total_landings)::float AS avg_total_landings,
                PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY airframe_hours) AS p10_airframe_hours,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY airframe_hours) AS p50_airframe_hours,
                PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY airframe_hours) AS p90_airframe_hours,
                PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY total_landings) AS p10_total_landings,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY total_landings) AS p50_total_landings,
                PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY total_landings) AS p90_total_landings,
                SUM(CASE WHEN for_sale IS TRUE THEN 1 ELSE 0 END) AS for_sale_count
            FROM filtered
            """,
            tuple(params + [max(100, int(max_summary_sample))]),
        )
        summary = dict(summary_rows[0]) if summary_rows else {}

        # Top bases/countries (from the same limited filtered set)
        top_bases = db.execute_query(
            f"""
            WITH filtered AS (
                SELECT base_code
                FROM aircraftpost_fleet_aircraft
                WHERE {where_sql}
                ORDER BY updated_at DESC
                LIMIT %s
            )
            SELECT base_code, COUNT(*) AS n
            FROM filtered
            WHERE base_code IS NOT NULL AND TRIM(base_code) <> ''
            GROUP BY base_code
            ORDER BY n DESC
            LIMIT 5
            """,
            tuple(params + [max(100, int(max_summary_sample))]),
        )
        top_countries = db.execute_query(
            f"""
            WITH filtered AS (
                SELECT country_code
                FROM aircraftpost_fleet_aircraft
                WHERE {where_sql}
                ORDER BY updated_at DESC
                LIMIT %s
            )
            SELECT country_code, COUNT(*) AS n
            FROM filtered
            WHERE country_code IS NOT NULL AND TRIM(country_code) <> ''
            GROUP BY country_code
            ORDER BY n DESC
            LIMIT 5
            """,
            tuple(params + [max(100, int(max_summary_sample))]),
        )

        total_records = int(summary.get("total_records") or 0)
        if total_records == 0 and not matches:
            return {"matches": [], "fleet_summary": {"total_records": 0}}

        fleet_summary: Dict[str, Any] = {
            "manufacturer": mfr,
            "model": mdl,
            "serial": ser or None,
            "total_records": total_records,
            "records_with_hours": int(summary.get("records_with_hours") or 0),
            "records_with_landings": int(summary.get("records_with_landings") or 0),
            "for_sale_count": int(summary.get("for_sale_count") or 0),
            "for_sale_rate": (
                float(summary.get("for_sale_count") or 0) / float(total_records)
                if total_records else None
            ),
            "airframe_hours": {
                "avg": summary.get("avg_airframe_hours"),
                "p10": summary.get("p10_airframe_hours"),
                "p50": summary.get("p50_airframe_hours"),
                "p90": summary.get("p90_airframe_hours"),
            },
            "total_landings": {
                "avg": summary.get("avg_total_landings"),
                "p10": summary.get("p10_total_landings"),
                "p50": summary.get("p50_total_landings"),
                "p90": summary.get("p90_total_landings"),
            },
            "top_bases": [dict(r) for r in (top_bases or [])],
            "top_countries": [dict(r) for r in (top_countries or [])],
            "note": f"Summary computed from up to {max_summary_sample} recent fleet records.",
        }

        return {"matches": [dict(r) for r in (matches or [])], "fleet_summary": fleet_summary}
    except Exception as e:
        logger.warning("aircraftpost fleet lookup failed: %s", e)
        return None

