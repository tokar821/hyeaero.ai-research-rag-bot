"""
Columns for ``public.phlydata_aircraft`` (internal CSV → Postgres).

Canonical typed columns are listed in ``PHLYDATA_AIRCRAFT_DB_COLUMNS``. The table may also contain
one TEXT column per extra CSV header (``csv_*``). Use :func:`phlydata_aircraft_select_sql` with a
database client to SELECT all non-PK data columns dynamically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, FrozenSet, Optional

if TYPE_CHECKING:
    from database.postgres_client import PostgresClient

# Core typed columns (always created by ETL). Extra CSV fields become additional ``csv_*`` TEXT columns.
PHLYDATA_AIRCRAFT_DB_COLUMNS: tuple[str, ...] = (
    "serial_number",
    "registration_number",
    "manufacturer",
    "model",
    "manufacturer_year",
    "delivery_year",
    "category",
    "aircraft_status",
    "transaction_status",
    "ask_price",
    "take_price",
    "sold_price",
    "airframe_total_time",
    "apu_total_time",
    "prop_total_time",
    "engine_program",
    "engine_program_deferment",
    "engine_program_deferment_amount",
    "apu_program",
    "apu_program_deferment",
    "apu_program_deferment_amount",
    "airframe_program",
    "maintenance_tracking_program",
    "registration_country",
    "based_country",
    "number_of_passengers",
    "date_listed",
    "interior_year",
    "exterior_year",
    "seller_broker",
    "buyer_broker",
    "seller",
    "buyer",
    "source_updated_at",
    "updated_by",
    "has_damage",
    "feature_source",
    "features",
    "next_inspections",
)

# Do not expose legacy JSON blob if it still exists on old DBs.
_DEFAULT_EXCLUDE: FrozenSet[str] = frozenset({"csv_extra"})


def fetch_phlydata_aircraft_data_columns(db: Any) -> list[str]:
    """All data columns for ``phlydata_aircraft`` except ``aircraft_id``, in table ordinal order."""
    rows = db.execute_query(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'phlydata_aircraft'
          AND column_name NOT IN ('aircraft_id')
        ORDER BY ordinal_position
        """
    )
    return [str(r["column_name"]) for r in rows]


def phlydata_aircraft_select_sql(
    *,
    table_alias: str = "",
    include_cast_id: bool = True,
    db: Optional["PostgresClient"] = None,
    exclude_columns: Optional[FrozenSet[str]] = None,
) -> str:
    """
    SELECT list: optional ``id`` (text UUID) + every data column (canonical + dynamic ``csv_*``).

    Pass ``db`` so extra PostgreSQL columns are included. If ``db`` is omitted, only canonical
    columns from ``PHLYDATA_AIRCRAFT_DB_COLUMNS`` are selected (tests / offline use).
    """
    ex = _DEFAULT_EXCLUDE | (exclude_columns or frozenset())
    if db is not None:
        cols = [c for c in fetch_phlydata_aircraft_data_columns(db) if c not in ex]
    else:
        cols = [c for c in PHLYDATA_AIRCRAFT_DB_COLUMNS if c not in ex]
    p = f"{table_alias}." if table_alias else ""
    parts: list[str] = []
    if include_cast_id:
        parts.append(f"CAST({p}aircraft_id AS TEXT) AS id")
    parts.extend(f"{p}{c}" for c in cols)
    return ",\n            ".join(parts)


def phlydata_aircraft_api_null_payload(db: Optional["PostgresClient"] = None) -> dict:
    """Default JSON shape when no row matches (keys align with dynamic SELECT when db is set)."""
    if db is not None:
        cols = [c for c in fetch_phlydata_aircraft_data_columns(db) if c not in _DEFAULT_EXCLUDE]
        return {"id": None, **{c: None for c in cols}}
    return {"id": None, **{c: None for c in PHLYDATA_AIRCRAFT_DB_COLUMNS}}
