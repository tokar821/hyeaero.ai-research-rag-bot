"""
Columns for ``public.phlydata_aircraft`` (internal CSV → Postgres).

This table is **Hye Aero’s internal PhlyData** only — the product’s canonical aircraft export in Postgres.
It is **not** marketplace ingest (that is ``aircraft_listings`` / Controller / exchanges).

**Standard semantics (what Ask Consultant and the Phly tab treat as authoritative):**

- **aircraft_status** — for-sale / disposition / inventory wording exactly as in the Phly export.
- **ask_price** — internal asking price from that export (same column name in the DB).

There is **no separate transaction_status** in the normal Hye Aero PhlyData shape; if a column with that
name exists on some deployments, treat it as **legacy / optional** (historic ETL or old spreadsheets).

``PHLYDATA_AIRCRAFT_DB_COLUMNS`` lists typed columns the ETL *may* create; for the live table use
:func:`fetch_phlydata_aircraft_data_columns` so missing columns are omitted from SELECT/API shapes.

--------------------------------------------------------------------
Identity columns: ``registration_number`` and ``serial_number`` (TEXT, **no CHECK**)
--------------------------------------------------------------------

Postgres stores **exactly what the Phly CSV/export wrote**. Consultant lookup uses
``TRIM(UPPER(registration_number))`` / ``serial_number`` equality, ``ILIKE`` patterns, and (for
civil tail-shaped tokens) a **hyphen-collapsed** variant so ``TC-KEA`` matches a row stored as ``TCKEA``
and vice versa — see :mod:`rag.phlydata_consultant_lookup`.

**registration_number (tail / civil mark)** — shapes to expect in exports:

1. **U.S. civil** — ``N`` + digits/letters, typically ``N118CX``, ``N12345`` (compact, no hyphen).
2. **International hyphenated** — ``PREFIX-SUFFIX`` e.g. ``PR-CCA``, ``TC-KEA``, ``C-GUGU``, ``G-CIVG``,
   ``FL-1185``, ``XA-ABC`` (format depends on state of registry).
3. **International compact** — same mark **without** hyphen (spreadsheet column); lookup must try both.
4. **Whitespace** — occasional leading/trailing spaces; SQL TRIM handles display match.
5. **Wrong column** — rarely the N-number or a fragment may appear in ``serial_number`` instead;
   :func:`rag.phlydata_consultant_lookup.lookup_phlydata_aircraft_rows` matches tokens against **both** columns.

**serial_number (MSN / factory / alternate id)** — shapes to expect:

1. **Plain numeric** — ``61033``, ``5500123``.
2. **OEM hyphenated** — ``525-0444``, ``510-0010``, ``550-0123`` (Cessna/Bombardier-style).
3. **Alphanumeric** — ``525B0044``, ``172S11842``, type codes embedded in MSN.
4. **Leading zeros** — ``0000171``, ``0011``.
5. **Prefixes in raw text** — some sources use ``SN-6201`` / ``S/N``; Pinecone metadata normalization
   strips those for vectors; **SQL lookup uses the raw cell** unless the user query token matches after trim.

**registration_country** — ISO-ish or free-text country for the tail; use with tail format when debugging.

To **measure what your deployment actually stores**, run :data:`PHLY_IDENTITY_SAMPLE_SQL` against Postgres
(adapt limits). For full distribution of patterns, group by regex classes in a one-off analysis query.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, FrozenSet, Optional

if TYPE_CHECKING:
    from database.postgres_client import PostgresClient

# Ad-hoc audit: run in psql or a notebook to see real registration/serial shapes in your environment.
PHLY_IDENTITY_SAMPLE_SQL = """
SELECT
  TRIM(registration_number) AS reg_raw,
  TRIM(serial_number) AS sn_raw,
  registration_country,
  manufacturer,
  model
FROM public.phlydata_aircraft
WHERE TRIM(COALESCE(registration_number, '')) <> ''
   OR TRIM(COALESCE(serial_number, '')) <> ''
ORDER BY source_updated_at DESC NULLS LAST, aircraft_id
LIMIT 200;
"""

PHLY_IDENTITY_PATTERN_COUNTS_SQL = """
-- Optional: coarse buckets for registration_number (edit regex list for your fleet).
SELECT
  CASE
    WHEN reg ~ '^N[0-9A-Z]{1,6}$' THEN 'us_n_compact'
    WHEN reg ~ '^[A-Z]{1,3}-[A-Z0-9]{2,16}$' THEN 'intl_hyphenated'
    WHEN reg ~ '^[A-Z0-9]{4,12}$' AND reg !~ '^N' THEN 'intl_or_compact_alnum'
    ELSE 'other_reg'
  END AS reg_bucket,
  COUNT(*) AS n
FROM (
  SELECT TRIM(UPPER(COALESCE(registration_number, ''))) AS reg
  FROM public.phlydata_aircraft
  WHERE TRIM(COALESCE(registration_number, '')) <> ''
) t
GROUP BY 1
ORDER BY n DESC;

SELECT
  CASE
    WHEN sn ~ '^[0-9]+$' THEN 'sn_digits_only'
    WHEN sn ~ '^[0-9]{2,5}-[0-9]{3,6}$' THEN 'sn_oem_hyphen_numeric'
    WHEN sn ~ '[A-Z]' AND sn ~ '[0-9]' THEN 'sn_mixed_alnum'
    ELSE 'other_sn'
  END AS sn_bucket,
  COUNT(*) AS n
FROM (
  SELECT TRIM(UPPER(COALESCE(serial_number, ''))) AS sn
  FROM public.phlydata_aircraft
  WHERE TRIM(COALESCE(serial_number, '')) <> ''
) t
GROUP BY 1
ORDER BY n DESC;
"""

# Core typed columns (always created by ETL). Extra CSV fields become additional ``csv_*`` TEXT columns.
# For-sale → aircraft_status. Internal ask → ask_price. transaction_status is legacy/optional only.
PHLYDATA_AIRCRAFT_DB_COLUMNS: tuple[str, ...] = (
    "serial_number",
    "registration_number",
    "manufacturer",
    "model",
    "manufacturer_year",
    "delivery_year",
    "category",
    "aircraft_status",
    "transaction_status",  # optional; omitted when not present in information_schema
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
