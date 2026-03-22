"""PhlyData owner lookup against ``faa_master`` (FAA MASTER CSV ingested to PostgreSQL).

``faa_master.n_number`` may be stored as ``277G`` (FAA CSV style) or ``N277G`` (with prefix).
Tier 1 compares a **canonical tail** in SQL: ``REGEXP_REPLACE(TRIM(UPPER(n_number)), '^N', '')`` so
client ``N277G`` + DB ``N277G`` / ``277G`` all align. Serial uses normalized variants (leading zeros).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from services.aircraftpost_lookup import serial_variants_for_lookup

if TYPE_CHECKING:
    from database.postgres_client import PostgresClient

logger = logging.getLogger(__name__)

_SELECT_FM = """
    SELECT
      fm.registrant_name, fm.street, fm.street2, fm.city, fm.state, fm.zip_code,
      fm.region, fm.county, fm.country
"""


def serial_keys_for_faa_master_match(serial: str) -> List[str]:
    """Keys aligned with ``_FM_SERIAL_NORM``: trim/upper then strip spaces, hyphens, dots (per variant)."""
    seen: set[str] = set()
    out: List[str] = []

    def add(k: str) -> None:
        if k and k not in seen:
            seen.add(k)
            out.append(k)

    for v in serial_variants_for_lookup((serial or "").strip()):
        k = (v or "").strip().upper()
        norm = re.sub(r"[\s\-.]+", "", k)
        add(norm)
    return out


# SQL expr: normalize serial like Python ``re.sub(r'[\\s\\-.]+', '', ...)`` (FAA padding)
_FM_SERIAL_NORM = (
    "REPLACE(REPLACE(REPLACE(TRIM(UPPER(COALESCE(fm.serial_number, ''))), ' ', ''), '-', ''), '.', '')"
)

# Canonical US tail: trim/upper, strip one leading N from column (matches N277G and 277G).
_FM_N_TAIL = (
    "REGEXP_REPLACE(TRIM(UPPER(COALESCE(fm.n_number, ''))), '^N', '')"
)


def registration_tail_canonical(reg: Optional[str]) -> Optional[str]:
    """
    Canonical form for ``faa_master`` tail matching — must stay in sync with ``_FM_N_TAIL`` in SQL.

    Examples: ``N277G`` / ``n-277g`` → ``277G``; ``277G`` → ``277G``.
    """
    if not reg or not str(reg).strip():
        return None
    t = str(reg).strip().upper()
    t = re.sub(r"[\s\-.]+", "", t)
    if t.startswith("N") and len(t) > 1:
        t = t[1:]
    return t or None


def _safe_select(db: "PostgresClient", sql: str, params: tuple) -> List[Dict[str, Any]]:
    try:
        return db.execute_query(sql, params)
    except Exception as e:
        msg = str(e).lower()
        if "faa_master" in msg and ("does not exist" in msg or "undefinedtable" in msg):
            logger.warning("faa_master not available (%s); skipping FAA MASTER owner lookup.", e)
            return []
        raise


def fetch_faa_master_owner_rows(
    db: "PostgresClient",
    *,
    serial: str,
    model: Optional[str],
    registration: Optional[str],
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Tier 1: ``n_number`` (``N277G`` / ``277G``) + normalized ``serial_number`` when tail is known.
    Tier 1b: ``n_number`` only when tier 1 misses (e.g. PhlyData serial ≠ FAA serial for same tail).
    Tier 2: ``serial_number`` + ``mfr_mdl_code`` via ``faa_aircraft_reference`` when model is provided.
    Tier 3: ``serial_number`` only when model is **not** provided.

    Returns ``(rows, match_kind)`` where ``match_kind`` is ``n_number_serial``, ``n_number_only``,
    ``serial_model``, ``serial_only``, or ``None``.
    """
    serial = (serial or "").strip()
    if not serial:
        return [], None

    mdl = (model or "").strip() or None
    tail_key = registration_tail_canonical(registration)
    ser_keys = serial_keys_for_faa_master_match(serial)
    if not ser_keys:
        ser_keys = [re.sub(r"[\s\-.]+", "", serial.strip().upper())]

    if tail_key:
        rows = _safe_select(
            db,
            _SELECT_FM
            + f"""
    FROM faa_master fm
    WHERE {_FM_N_TAIL} = %s
      AND {_FM_SERIAL_NORM} = ANY(%s)
    ORDER BY fm.ingestion_date DESC NULLS LAST
    LIMIT 10
    """,
            (tail_key, ser_keys),
        )
        if rows:
            return rows, "n_number_serial"

        # Tail is authoritative for US registry; PhlyData serial may not match FAA serial for same tail.
        rows = _safe_select(
            db,
            _SELECT_FM
            + f"""
    FROM faa_master fm
    WHERE {_FM_N_TAIL} = %s
    ORDER BY fm.ingestion_date DESC NULLS LAST
    LIMIT 1
    """,
            (tail_key,),
        )
        if rows:
            return rows, "n_number_only"

    if mdl:
        rows = _safe_select(
            db,
            _SELECT_FM
            + f"""
    FROM faa_master fm
    WHERE {_FM_SERIAL_NORM} = ANY(%s)
      AND fm.mfr_mdl_code IN (
        SELECT far.code FROM faa_aircraft_reference far
        WHERE far.model ILIKE %s
      )
    ORDER BY fm.ingestion_date DESC NULLS LAST
    LIMIT 10
    """,
            (ser_keys, f"%{mdl}%"),
        )
        return rows, ("serial_model" if rows else None)

    rows = _safe_select(
        db,
        _SELECT_FM
        + f"""
    FROM faa_master fm
    WHERE {_FM_SERIAL_NORM} = ANY(%s)
    ORDER BY fm.ingestion_date DESC NULLS LAST
    LIMIT 3
    """,
        (ser_keys,),
    )
    return rows, "serial_only" if rows else ([], None)
