"""
Columns required by Ask Consultant SQL paths (``public.*`` tables).

Validated against a live Postgres instance (see ``tests/test_consultant_sql_schema.py``).
If migrations rename columns, update these frozensets and the corresponding queries
in :mod:`rag.consultant_market_lookup`, :mod:`rag.phlydata_consultant_lookup`, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, FrozenSet

if TYPE_CHECKING:
    from database.postgres_client import PostgresClient

# public.aircraft — :func:`rag.phlydata_consultant_lookup.lookup_aircraft_master_rows`
AIRCRAFT: FrozenSet[str] = frozenset(
    {
        "id",
        "serial_number",
        "registration_number",
        "manufacturer",
        "model",
        "manufacturer_year",
        "delivery_year",
        "category",
        "aircraft_status",
        "condition",
        "registration_country",
        "based_country",
        "updated_at",
    }
)

# public.aircraft_listings — listing join in :mod:`rag.consultant_market_lookup`
AIRCRAFT_LISTINGS: FrozenSet[str] = frozenset(
    {
        "id",
        "aircraft_id",
        "source_platform",
        "source_listing_id",
        "listing_status",
        "ask_price",
        "sold_price",
        "listing_url",
        "seller",
        "seller_broker",
        "location",
        "date_listed",
        "date_sold",
        "created_at",
        "updated_at",
    }
)

# public.aircraft_sales — comps query in consultant_market_lookup
AIRCRAFT_SALES: FrozenSet[str] = frozenset(
    {
        "manufacturer",
        "model",
        "manufacturer_year",
        "sold_price",
        "ask_price",
        "date_sold",
        "airframe_total_time",
        "based_country",
        "registration_country",
    }
)

# public.faa_master + public.faa_aircraft_reference — :mod:`services.faa_master_lookup`
FAA_MASTER: FrozenSet[str] = frozenset(
    {
        "n_number",
        "serial_number",
        "mfr_mdl_code",
        "registrant_name",
        "street",
        "street2",
        "city",
        "state",
        "zip_code",
        "region",
        "county",
        "country",
        "year_mfr",
        "type_aircraft",
        "type_engine",
        "certification",
        "status_code",
        "ingestion_date",
    }
)

FAA_AIRCRAFT_REFERENCE: FrozenSet[str] = frozenset(
    {
        "code",
        "manufacturer",
        "model",
    }
)


def table_columns(db: "PostgresClient", table: str) -> FrozenSet[str]:  # noqa: F821
    rows = db.execute_query(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        """,
        (table,),
    )
    return frozenset(str(r["column_name"]) for r in rows)


def verify_consultant_sql_columns(db: "PostgresClient") -> list[str]:
    """Return human-readable problems; empty list means all required columns exist."""
    checks = [
        ("aircraft", AIRCRAFT),
        ("aircraft_listings", AIRCRAFT_LISTINGS),
        ("aircraft_sales", AIRCRAFT_SALES),
        ("faa_master", FAA_MASTER),
        ("faa_aircraft_reference", FAA_AIRCRAFT_REFERENCE),
    ]
    problems: list[str] = []
    for table, required in checks:
        try:
            have = table_columns(db, table)
        except Exception as e:
            problems.append(f"{table}: cannot introspect ({e})")
            continue
        missing = sorted(required - have)
        if missing:
            problems.append(f"{table}: missing columns {missing}")
    return problems
