#!/usr/bin/env python3
"""PostgreSQL data quality report for RAG source tables.

Usage (from backend/):
  python runners/rag_data_quality_report.py

Requires POSTGRES_CONNECTION_STRING or POSTGRES_* in .env.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config_loader import get_config
from database.postgres_client import PostgresClient
from rag.aircraft_normalization import normalized_type_key


def _table_columns(db: PostgresClient, table: str) -> set[str]:
    rows = db.execute_query(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        """,
        (table,),
    )
    return {str(r["column_name"]) for r in rows}


def _pct(num: float, den: float) -> str:
    if den <= 0:
        return "n/a"
    return f"{100.0 * num / den:.1f}%"


def main() -> int:
    cfg = get_config()
    cs = cfg.postgres_connection_string
    if not cs:
        host = cfg.postgres_host
        db = cfg.postgres_database
        user = cfg.postgres_user
        pw = cfg.postgres_password or ""
        port = cfg.postgres_port or 5432
        if not all([host, db, user]):
            print("ERROR: No PostgreSQL config (POSTGRES_CONNECTION_STRING or POSTGRES_HOST/DATABASE/USER).")
            return 1
        cs = f"postgresql://{user}:{pw}@{host}:{port}/{db}"

    db = PostgresClient(cs)
    try:
        db.execute_query("SELECT 1")
    except Exception as e:
        print(f"ERROR: Database connection failed: {e}")
        return 1

    print("=== RAG data quality report ===\n")

    tables = [
        "aircraft",
        "aircraft_listings",
        "aircraft_sales",
        "faa_registrations",
        "aviacost_aircraft_details",
        "phlydata_aircraft",
    ]
    for t in tables:
        try:
            r = db.execute_query(f"SELECT COUNT(*) AS c FROM public.{t}")
            print(f"  {t}: {int(r[0]['c']):,} rows")
        except Exception as e:
            print(f"  {t}: MISSING OR ERROR — {e}")
    print()

    # --- aircraft ---
    print("### aircraft")
    try:
        r = db.execute_query("SELECT COUNT(*) AS c FROM public.aircraft")[0]["c"]
        tot = int(r)
        for col in (
            "serial_number",
            "registration_number",
            "manufacturer",
            "model",
            "manufacturer_year",
            "type_engine",
        ):
            q = f"""
            SELECT COUNT(*) AS c FROM public.aircraft
            WHERE {col} IS NULL OR TRIM(COALESCE({col}::text, '')) = ''
            """
            nulls = int(db.execute_query(q)[0]["c"])
            print(f"  missing {col}: {nulls:,} ({_pct(nulls, tot)})")

        dup_reg = db.execute_query(
            """
            SELECT registration_number, COUNT(*) AS n
            FROM public.aircraft
            WHERE registration_number IS NOT NULL AND TRIM(registration_number) <> ''
            GROUP BY registration_number
            HAVING COUNT(*) > 1
            ORDER BY n DESC
            LIMIT 15
            """
        )
        print(f"  duplicate registration_number groups: {len(dup_reg)} (show top 15)")
        for row in dup_reg[:5]:
            print(f"    {row['registration_number']}: {row['n']} rows")

        dup_serial = db.execute_query(
            """
            SELECT serial_number, COUNT(*) AS n
            FROM public.aircraft
            WHERE serial_number IS NOT NULL AND TRIM(serial_number) <> ''
            GROUP BY serial_number
            HAVING COUNT(*) > 1
            ORDER BY n DESC
            LIMIT 15
            """
        )
        print(f"  duplicate serial_number groups: {len(dup_serial)} (show top 5)")
        for row in dup_serial[:5]:
            print(f"    {row['serial_number']}: {row['n']} rows")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()

    # --- listings: join + optional denormalized mfr/model ---
    print("### aircraft_listings (join consistency)")
    try:
        lc = _table_columns(db, "aircraft_listings")
        j = db.execute_query(
            """
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN al.aircraft_id IS NULL THEN 1 ELSE 0 END) AS missing_aircraft_id
            FROM public.aircraft_listings al
            """
        )[0]
        tot = int(j["total"] or 0)
        miss = int(j["missing_aircraft_id"] or 0)
        print(f"  total listings: {tot:,}; aircraft_id NULL: {miss:,} ({_pct(miss, tot)})")
        has_list_mfr = "manufacturer" in lc and "model" in lc
        if not has_list_mfr:
            print(
                "  NOTE: aircraft_listings has no manufacturer/model columns in this DB — "
                "RAG listing text cannot print make/model unless synced to aircraft or raw_data; "
                "entity_extractors may emit weaker embeddings for listings without aircraft_id."
            )
        else:
            mism = db.execute_query(
                """
                SELECT COUNT(*) AS c
                FROM public.aircraft_listings al
                JOIN public.aircraft a ON a.id = al.aircraft_id
                WHERE al.manufacturer IS NOT NULL AND a.manufacturer IS NOT NULL
                  AND TRIM(al.manufacturer) <> '' AND TRIM(a.manufacturer) <> ''
                  AND UPPER(TRIM(al.manufacturer)) <> UPPER(TRIM(a.manufacturer))
                """
            )[0]["c"]
            print(
                f"  listing.manufacturer differs from aircraft.manufacturer (when both set): {int(mism):,}"
            )

            mism_m = db.execute_query(
                """
                SELECT COUNT(*) AS c
                FROM public.aircraft_listings al
                JOIN public.aircraft a ON a.id = al.aircraft_id
                WHERE al.model IS NOT NULL AND a.model IS NOT NULL
                  AND TRIM(al.model) <> '' AND TRIM(a.model) <> ''
                  AND UPPER(TRIM(al.model)) <> UPPER(TRIM(a.model))
                """
            )[0]["c"]
            print(f"  listing.model differs from aircraft.model (when both set): {int(mism_m):,}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()

    # --- sales ---
    print("### aircraft_sales")
    try:
        r = db.execute_query("SELECT COUNT(*) AS c FROM public.aircraft_sales")[0]["c"]
        tot = int(r)
        for col in ("serial_number", "registration_number", "manufacturer", "model"):
            nulls = int(
                db.execute_query(
                    f"""
                SELECT COUNT(*) AS c FROM public.aircraft_sales
                WHERE {col} IS NULL OR TRIM(COALESCE({col}::text, '')) = ''
                """
                )[0]["c"]
            )
            print(f"  missing {col}: {nulls:,} ({_pct(nulls, tot)})")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()

    # --- faa ---
    print("### faa_registrations")
    try:
        r = db.execute_query("SELECT COUNT(*) AS c FROM public.faa_registrations")[0]["c"]
        tot = int(r)
        for col in ("n_number", "serial_number", "type_aircraft", "type_engine"):
            nulls = int(
                db.execute_query(
                    f"""
                SELECT COUNT(*) AS c FROM public.faa_registrations
                WHERE {col} IS NULL OR TRIM(COALESCE({col}::text, '')) = ''
                """
                )[0]["c"]
            )
            print(f"  missing {col}: {nulls:,} ({_pct(nulls, tot)})")

        dup_n = db.execute_query(
            """
            SELECT n_number, COUNT(DISTINCT ingestion_date) AS versions
            FROM public.faa_registrations
            WHERE n_number IS NOT NULL AND TRIM(n_number) <> ''
            GROUP BY n_number
            HAVING COUNT(*) > 1
            ORDER BY COUNT(*) DESC
            LIMIT 5
            """
        )
        print(f"  sample N-numbers with multiple rows (ingest history): {len(dup_n)} shown")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()

    # --- aviacost ---
    print("### aviacost_aircraft_details")
    try:
        r = db.execute_query("SELECT COUNT(*) AS c FROM public.aviacost_aircraft_details")[0]["c"]
        tot = int(r)
        for col in ("name", "manufacturer_name", "engine_model", "powerplant"):
            nulls = int(
                db.execute_query(
                    f"""
                SELECT COUNT(*) AS c FROM public.aviacost_aircraft_details
                WHERE {col} IS NULL OR TRIM(COALESCE({col}::text, '')) = ''
                """
                )[0]["c"]
            )
            print(f"  missing {col}: {nulls:,} ({_pct(nulls, tot)})")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()

    # --- phlydata ---
    print("### phlydata_aircraft")
    try:
        r = db.execute_query("SELECT COUNT(*) AS c FROM public.phlydata_aircraft")[0]["c"]
        tot = int(r)
        for col in ("serial_number", "registration_number", "manufacturer", "model"):
            nulls = int(
                db.execute_query(
                    f"""
                SELECT COUNT(*) AS c FROM public.phlydata_aircraft
                WHERE {col} IS NULL OR TRIM(COALESCE({col}::text, '')) = ''
                """
                )[0]["c"]
            )
            print(f"  missing {col}: {nulls:,} ({_pct(nulls, tot)})")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()

    # --- G650 family strings (raw) ---
    print("### Gulfstream G650 naming variants (raw substring search)")
    lc_g6 = _table_columns(db, "aircraft_listings")
    if "manufacturer" in lc_g6 and "model" in lc_g6:
        listing_g650_union = """
    UNION ALL
    SELECT 'aircraft_listings' AS src, manufacturer, model, COUNT(*) AS n
    FROM public.aircraft_listings
    WHERE (COALESCE(manufacturer,'') || ' ' || COALESCE(model,'')) ~* 'gulfstream.*650|g\\s*-?\\s*650'
       OR COALESCE(model,'') ~* '^g\\s*-?\\s*650'
    GROUP BY manufacturer, model
    """
    else:
        listing_g650_union = """
    UNION ALL
    SELECT 'aircraft_listings_join_aircraft' AS src, a.manufacturer, a.model, COUNT(*) AS n
    FROM public.aircraft_listings al
    JOIN public.aircraft a ON a.id = al.aircraft_id
    WHERE (COALESCE(a.manufacturer,'') || ' ' || COALESCE(a.model,'')) ~* 'gulfstream.*650|g\\s*-?\\s*650'
       OR COALESCE(a.model,'') ~* '^g\\s*-?\\s*650'
    GROUP BY a.manufacturer, a.model
    """
    pattern_sql = """
    SELECT 'aircraft' AS src, manufacturer, model, COUNT(*) AS n
    FROM public.aircraft
    WHERE (COALESCE(manufacturer,'') || ' ' || COALESCE(model,'')) ~* 'gulfstream.*650|g\\s*-?\\s*650'
       OR COALESCE(model,'') ~* '^g\\s*-?\\s*650'
    GROUP BY manufacturer, model
    """ + listing_g650_union + """
    UNION ALL
    SELECT 'aircraft_sales' AS src, manufacturer, model, COUNT(*) AS n
    FROM public.aircraft_sales
    WHERE (COALESCE(manufacturer,'') || ' ' || COALESCE(model,'')) ~* 'gulfstream.*650|g\\s*-?\\s*650'
       OR COALESCE(model,'') ~* '^g\\s*-?\\s*650'
    GROUP BY manufacturer, model
    UNION ALL
    SELECT 'aviacost_aircraft_details' AS src, manufacturer_name, name, COUNT(*) AS n
    FROM public.aviacost_aircraft_details
    WHERE (COALESCE(manufacturer_name,'') || ' ' || COALESCE(name,'')) ~* 'gulfstream.*650|g\\s*-?\\s*650'
       OR COALESCE(name,'') ~* '^g\\s*-?\\s*650'
    GROUP BY manufacturer_name, name
    """
    try:
        phly_union = """
    UNION ALL
    SELECT 'phlydata_aircraft' AS src, manufacturer, model, COUNT(*) AS n
    FROM public.phlydata_aircraft
    WHERE (COALESCE(manufacturer,'') || ' ' || COALESCE(model,'')) ~* 'gulfstream.*650|g\\s*-?\\s*650'
       OR COALESCE(model,'') ~* '^g\\s*-?\\s*650'
    GROUP BY manufacturer, model
    """
        full_sql = pattern_sql + " ORDER BY src, n DESC"
        try:
            rows = db.execute_query(pattern_sql + phly_union + " ORDER BY src, n DESC")
        except Exception:
            rows = db.execute_query(full_sql)
        type_keys: dict[str, set[str]] = {}
        for row in rows:
            m = row.get("manufacturer") or row.get("manufacturer_name")
            mo = row.get("model") or row.get("name")
            tk = normalized_type_key(str(m) if m else None, str(mo) if mo else None)
            type_keys.setdefault(tk, set()).add(f"{row['src']}: mfr={m!r} model={mo!r} (n={row['n']})")

        print("  Raw manufacturer/model groups (G650-related):")
        for row in rows[:40]:
            print(f"    {row['src']}: mfr={row.get('manufacturer') or row.get('manufacturer_name')!r} "
                  f"model={row.get('model') or row.get('name')!r} count={row['n']}")
        if len(rows) > 40:
            print(f"    ... ({len(rows) - 40} more groups)")
        print("  After normalization (type_key -> sources):")
        for tk, lines in sorted(type_keys.items(), key=lambda x: (-len(x[1]), x[0]))[:20]:
            print(f"    {tk!r}:")
            for line in sorted(lines):
                print(f"      {line}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n=== End report ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
