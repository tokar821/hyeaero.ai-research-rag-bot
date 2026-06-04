"""Normalized market snapshots from Controller, AircraftExchange, and AircraftPost ingest."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from database.postgres_client import PostgresClient

logger = logging.getLogger(__name__)

PREFERRED_PLATFORMS = frozenset(
    {
        "controller",
        "aircraftexchange",
        "aircraft_exchange",
        "aircraftpost",
    }
)

_INACTIVE_STATUS = frozenset(
    {
        "sold",
        "withdrawn",
        "expired",
        "inactive",
        "off_market",
        "closed",
    }
)

MAX_LISTING_FETCH = 200
DEFAULT_STALE_DAYS = 90


@dataclass(frozen=True)
class MarketSnapshot:
    model: str
    active_listing_count: int
    median_ask_price: Optional[float]
    low_ask_price: Optional[float]
    high_ask_price: Optional[float]
    median_year: Optional[int]
    average_days_on_market: Optional[float]
    last_refresh: Optional[str]
    listing_sources: tuple[str, ...] = field(default_factory=tuple)
    stale: bool = False
    insufficient_reason: Optional[str] = None


def _parse_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_int(v: Any) -> Optional[int]:
    f = _parse_float(v)
    if f is None:
        return None
    y = int(f)
    if 1950 <= y <= 2035:
        return y
    return None


def _row_platform(row: Dict[str, Any]) -> str:
    return (row.get("source_platform") or "").strip().lower()


def _is_active_listing(row: Dict[str, Any]) -> bool:
    st = (row.get("listing_status") or "").strip().lower()
    if st in _INACTIVE_STATUS:
        return False
    if st in ("for_sale", "active", "available", "listed", "on market", "on_market"):
        return True
    if not st:
        return row.get("ask_price") is not None
    return st not in _INACTIVE_STATUS


def _model_match_sql_patterns(model: str) -> List[str]:
    m = (model or "").strip()
    if not m:
        return []
    return [f"%{m}%"]


def fetch_listing_rows(
    db: "PostgresClient",
    model: str,
    *,
    limit: int = MAX_LISTING_FETCH,
) -> List[Dict[str, Any]]:
    """Pull listing rows for a model from preferred marketplace platforms."""
    patterns = _model_match_sql_patterns(model)
    if not patterns:
        return []

    plat_placeholders = ", ".join(["%s"] * len(PREFERRED_PLATFORMS))
    model_conds = []
    params: List[Any] = list(PREFERRED_PLATFORMS)
    for p in patterns:
        model_conds.append(
            "(a.manufacturer ILIKE %s OR a.model ILIKE %s OR (a.manufacturer || ' ' || a.model) ILIKE %s)"
        )
        params.extend([p, p, p])

    params.append(limit)
    query = f"""
        SELECT
            l.id AS listing_id,
            l.source_platform,
            l.listing_status,
            l.ask_price,
            l.days_on_market,
            l.updated_at,
            l.created_at,
            l.date_listed,
            a.manufacturer,
            a.model,
            a.manufacturer_year
        FROM aircraft_listings l
        LEFT JOIN aircraft a ON l.aircraft_id = a.id
        WHERE LOWER(COALESCE(l.source_platform, '')) IN ({plat_placeholders})
          AND ({' OR '.join(model_conds)})
          AND l.ask_price IS NOT NULL
        ORDER BY l.updated_at DESC NULLS LAST, l.created_at DESC
        LIMIT %s
    """
    try:
        rows = db.execute_query(query, tuple(params))
    except Exception:
        logger.exception("market_intelligence listing fetch failed for %s", model)
        return []

    out: List[Dict[str, Any]] = []
    for r in rows or []:
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
    return out


def fetch_aircraftpost_for_sale_count(db: "PostgresClient", model: str) -> int:
    """Supplemental active-for-sale count from AircraftPost fleet (no ask required)."""
    m = (model or "").strip()
    if not m:
        return 0
    parts = m.split(None, 1)
    if len(parts) >= 2:
        like_mfr, like_mdl = f"%{parts[0]}%", f"%{parts[1]}%"
    else:
        like_mfr, like_mdl = f"%{m}%", f"%{m}%"
    try:
        rows = db.execute_query(
            """
            SELECT COUNT(*) AS n
            FROM aircraftpost_fleet_aircraft
            WHERE make_model_name ILIKE %s
              AND make_model_name ILIKE %s
              AND for_sale IS TRUE
            """,
            (like_mfr, like_mdl),
        )
        if rows:
            return int(rows[0].get("n") or 0)
    except Exception:
        logger.debug("aircraftpost for_sale count unavailable for %s", model, exc_info=True)
    return 0


def _filter_and_normalize_rows(
    model: str,
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    from rag.consultant_market_lookup import (
        filter_listings_sane_ask_prices,
        prioritize_and_deduplicate_listing_rows,
    )

    active = [r for r in rows if _is_active_listing(r) and _row_platform(r) in PREFERRED_PLATFORMS]
    deduped = prioritize_and_deduplicate_listing_rows(active)
    return filter_listings_sane_ask_prices(deduped)


def _median_int(values: List[int]) -> Optional[int]:
    if not values:
        return None
    return int(median(values))


def _latest_refresh_iso(rows: Sequence[Dict[str, Any]]) -> Optional[str]:
    best = 0.0
    for r in rows:
        for k in ("updated_at", "created_at", "date_listed"):
            v = r.get(k)
            if v is None:
                continue
            if hasattr(v, "timestamp"):
                best = max(best, float(v.timestamp()))
            elif isinstance(v, str) and len(v) >= 10:
                try:
                    ts = datetime.fromisoformat(v[:19].replace("Z", "+00:00")).timestamp()
                    best = max(best, ts)
                except ValueError:
                    continue
    if best <= 0:
        return None
    return datetime.fromtimestamp(best, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_stale(refresh_iso: Optional[str], *, max_age_days: int = DEFAULT_STALE_DAYS) -> bool:
    if not refresh_iso:
        return True
    try:
        dt = datetime.fromisoformat(refresh_iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
        return age > max_age_days
    except ValueError:
        return True


def build_market_snapshot(
    model: str,
    rows: Sequence[Dict[str, Any]],
    *,
    aircraftpost_for_sale: int = 0,
    min_listings: int = 3,
    stale_days: int = DEFAULT_STALE_DAYS,
) -> MarketSnapshot:
    """Aggregate listing rows into a normalized ``MarketSnapshot``."""
    clean = _filter_and_normalize_rows(model, list(rows))
    asks: List[float] = []
    years: List[int] = []
    doms: List[float] = []
    sources: set[str] = set()

    for r in clean:
        ask = _parse_float(r.get("ask_price"))
        if ask is not None and ask > 0:
            asks.append(ask)
        y = _parse_int(r.get("manufacturer_year"))
        if y is not None:
            years.append(y)
        dom = _parse_float(r.get("days_on_market"))
        if dom is not None and dom >= 0:
            doms.append(dom)
        plat = _row_platform(r)
        if plat:
            sources.add(plat)

    active_count = len(asks)
    if aircraftpost_for_sale > active_count:
        active_count = aircraftpost_for_sale

    refresh = _latest_refresh_iso(clean)
    stale = _is_stale(refresh, max_age_days=stale_days)

    insufficient: Optional[str] = None
    if len(asks) < min_listings:
        insufficient = "too_few_listings"
    elif stale:
        insufficient = "stale_market"

    return MarketSnapshot(
        model=(model or "").strip(),
        active_listing_count=active_count,
        median_ask_price=float(median(asks)) if asks else None,
        low_ask_price=min(asks) if asks else None,
        high_ask_price=max(asks) if asks else None,
        median_year=_median_int(years),
        average_days_on_market=(sum(doms) / len(doms)) if doms else None,
        last_refresh=refresh,
        listing_sources=tuple(sorted(sources)),
        stale=stale,
        insufficient_reason=insufficient,
    )


def load_market_snapshot(db: "PostgresClient", model: str) -> MarketSnapshot:
    """Fetch listings and build a snapshot for ``model``."""
    rows = fetch_listing_rows(db, model)
    ap_count = fetch_aircraftpost_for_sale_count(db, model)
    return build_market_snapshot(model, rows, aircraftpost_for_sale=ap_count)
