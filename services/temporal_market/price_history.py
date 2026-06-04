"""Listing price time series from Controller + AircraftExchange ingest."""

from __future__ import annotations

import logging
from collections import defaultdict
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
    }
)

MAX_HISTORY_ROWS = 500


@dataclass(frozen=True)
class PriceHistorySeries:
    model: str
    timestamps: tuple[str, ...]
    prices: tuple[float, ...]
    listing_sources: tuple[str, ...]
    last_updated: Optional[str]
    dom_samples: tuple[float, ...] = field(default_factory=tuple)
    point_count: int = 0
    insufficient_history: bool = True


def _parse_ts(v: Any) -> Optional[float]:
    if v is None:
        return None
    if hasattr(v, "timestamp"):
        try:
            return float(v.timestamp())
        except Exception:
            return None
    if isinstance(v, str) and len(v) >= 10:
        try:
            return datetime.fromisoformat(v[:19].replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rows_from_db(db: "PostgresClient", model: str) -> List[Dict[str, Any]]:
    from services.market_intelligence.listing_analytics import _model_match_sql_patterns

    patterns = _model_match_sql_patterns(model)
    if not patterns:
        return []

    plat_ph = ", ".join(["%s"] * len(PREFERRED_PLATFORMS))
    model_conds = []
    params: List[Any] = list(PREFERRED_PLATFORMS)
    for p in patterns:
        model_conds.append(
            "(a.manufacturer ILIKE %s OR a.model ILIKE %s OR (a.manufacturer || ' ' || a.model) ILIKE %s)"
        )
        params.extend([p, p, p])
    params.append(MAX_HISTORY_ROWS)

    query = f"""
        SELECT
            l.ask_price,
            l.updated_at,
            l.created_at,
            l.date_listed,
            l.source_platform,
            l.days_on_market
        FROM aircraft_listings l
        LEFT JOIN aircraft a ON l.aircraft_id = a.id
        WHERE LOWER(COALESCE(l.source_platform, '')) IN ({plat_ph})
          AND ({' OR '.join(model_conds)})
          AND l.ask_price IS NOT NULL
        ORDER BY COALESCE(l.updated_at, l.created_at, l.date_listed) ASC NULLS LAST
        LIMIT %s
    """
    try:
        rows = db.execute_query(query, tuple(params))
        return [dict(r) for r in rows or []]
    except Exception:
        logger.exception("price_history fetch failed for %s", model)
        return []


def _aggregate_daily_points(rows: Sequence[Dict[str, Any]]) -> tuple[List[float], List[float], List[str], List[float]]:
    """Bucket by UTC day → median ask; collect DOM samples."""
    by_day: Dict[str, List[float]] = defaultdict(list)
    doms: List[float] = []
    sources: List[str] = []

    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            ask = float(r.get("ask_price"))
        except (TypeError, ValueError):
            continue
        if ask <= 0:
            continue
        ts = _parse_ts(r.get("updated_at")) or _parse_ts(r.get("created_at")) or _parse_ts(r.get("date_listed"))
        if ts is None:
            continue
        day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        by_day[day].append(ask)
        plat = (r.get("source_platform") or "").strip().lower()
        if plat:
            sources.append(plat)
        dom = r.get("days_on_market")
        try:
            if dom is not None:
                doms.append(float(dom))
        except (TypeError, ValueError):
            pass

    days_sorted = sorted(by_day.keys())
    timestamps: List[float] = []
    prices: List[float] = []
    for day in days_sorted:
        pts = by_day[day]
        if not pts:
            continue
        dt = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        timestamps.append(dt.timestamp())
        prices.append(float(median(pts)))

    iso_ts = [_iso(t) for t in timestamps]
    return timestamps, prices, iso_ts, doms


def collect_price_history(
    db: Optional["PostgresClient"],
    model: str,
    *,
    listing_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> PriceHistorySeries:
    """
    Build a daily median ask series for ``model``.

  Never invents prices — empty when no rows.
    """
    rows: List[Dict[str, Any]] = []
    if listing_rows:
        rows = [dict(r) for r in listing_rows if isinstance(r, dict)]
    elif db is not None:
        rows = _rows_from_db(db, model)

    ts_floats, prices, iso_ts, doms = _aggregate_daily_points(rows)
    last_updated = iso_ts[-1] if iso_ts else None
    n = len(prices)
    return PriceHistorySeries(
        model=(model or "").strip(),
        timestamps=tuple(iso_ts),
        prices=tuple(prices),
        listing_sources=tuple(
            sorted(
                {
                    (r.get("source_platform") or "").strip().lower()
                    for r in rows
                    if (r.get("source_platform") or "").strip()
                }
            )
        ),
        last_updated=last_updated,
        dom_samples=tuple(doms[:200]),
        point_count=n,
        insufficient_history=n < 5,
    )
