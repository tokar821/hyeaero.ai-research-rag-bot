"""
Logging of user questions to Ask Consultant (PostgreSQL).

On by default whenever the API uses Postgres. Set ``CONSULTANT_QUERY_ANALYTICS_ENABLED=0`` (or
``false`` / ``no`` / ``off``) to stop inserting new rows; admin list/delete APIs still work.

Privacy: stores the raw ``query`` string and optional client hints; use a retention policy and
appropriate notices to users if required in your jurisdiction.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from database.postgres_client import PostgresClient

logger = logging.getLogger(__name__)

_MAX_QUERY_CHARS = 16000
_MAX_UA_CHARS = 512
_MAX_IP_CHARS = 200
_MAX_SEARCH_CHARS = 500


def _env_truthy(key: str) -> bool:
    return (os.getenv(key) or "").strip().lower() in ("1", "true", "yes")


def consultant_query_analytics_enabled() -> bool:
    """Unset defaults to **on**. Explicit ``0``/``false``/``no``/``off`` turns inserts off."""
    v = (os.getenv("CONSULTANT_QUERY_ANALYTICS_ENABLED") or "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True


@dataclass
class ConsultantQueryListFilters:
    """URL-friendly filter set for list + count (ISO dates ``YYYY-MM-DD``, UTC day bounds)."""

    date_from: Optional[str] = None
    date_to: Optional[str] = None
    endpoint: Optional[str] = None  # "sync" | "stream"
    q: Optional[str] = None  # ILIKE substring on query_text
    user_id: Optional[int] = None  # logged-in user who asked (FK app_users)


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    if not value or not str(value).strip():
        return None
    s = str(value).strip()[:10]
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _escape_ilike_literal(fragment: str) -> str:
    """Escape ``%``, ``_``, ``\\`` for use in ILIKE ... ESCAPE '\\'."""
    return (
        (fragment or "")
        .replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _where_clause_for_filters(filters: ConsultantQueryListFilters) -> Tuple[str, tuple]:
    parts: List[str] = []
    params: List[Any] = []

    d0 = _parse_iso_date(filters.date_from)
    if d0 is not None:
        start = datetime(d0.year, d0.month, d0.day, tzinfo=timezone.utc)
        parts.append("l.created_at >= %s")
        params.append(start)

    d1 = _parse_iso_date(filters.date_to)
    if d1 is not None:
        end_day = datetime(d1.year, d1.month, d1.day, tzinfo=timezone.utc) + timedelta(days=1)
        parts.append("l.created_at < %s")
        params.append(end_day)

    ep = (filters.endpoint or "").strip().lower()
    if ep in ("sync", "stream"):
        parts.append("l.endpoint = %s")
        params.append(ep)

    raw_q = (filters.q or "").strip()
    if raw_q:
        chunk = _escape_ilike_literal(raw_q[:_MAX_SEARCH_CHARS])
        parts.append("l.query_text ILIKE %s ESCAPE '\\'")
        params.append(f"%{chunk}%")

    if filters.user_id is not None:
        try:
            uid = int(filters.user_id)
        except (TypeError, ValueError):
            uid = -1
        if uid >= 1:
            parts.append("l.user_id = %s")
            params.append(uid)

    if not parts:
        return "TRUE", tuple()
    return " AND ".join(parts), tuple(params)


def record_consultant_query(
    db: PostgresClient,
    *,
    query: str,
    endpoint: str,
    history_turn_count: int = 0,
    client_ip: Optional[str] = None,
    user_agent: Optional[str] = None,
    user_id: Optional[int] = None,
) -> None:
    """
    Insert one row. Silently no-ops if analytics disabled via env. Raises only if caller should surface DB errors.
    """
    if not consultant_query_analytics_enabled():
        return
    q = (query or "").strip()
    if not q:
        return
    q = q[:_MAX_QUERY_CHARS]
    ep = (endpoint or "unknown")[:16]
    ua = (user_agent or "")[:_MAX_UA_CHARS] or None
    ip = (client_ip or "")[:_MAX_IP_CHARS] or None
    htc = max(0, min(10_000, int(history_turn_count)))
    uid = None
    if user_id is not None:
        try:
            uid = int(user_id)
        except (TypeError, ValueError):
            uid = None
        if uid is not None and uid < 1:
            uid = None
    db.execute_update(
        """
        INSERT INTO consultant_query_log
            (query_text, endpoint, history_turn_count, client_ip, user_agent, user_id)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (q, ep, htc, ip, ua, uid),
    )


def list_consultant_queries(
    db: PostgresClient,
    *,
    limit: int = 50,
    offset: int = 0,
    filters: Optional[ConsultantQueryListFilters] = None,
) -> List[Dict[str, Any]]:
    """Return rows newest first (for admin API); optional ``filters`` narrow the set."""
    lim = max(1, min(500, int(limit)))
    off = max(0, min(100_000, int(offset)))
    fl = filters or ConsultantQueryListFilters()
    where_sql, where_params = _where_clause_for_filters(fl)
    rows = db.execute_query(
        f"""
        SELECT l.id, l.created_at, l.query_text, l.endpoint, l.history_turn_count, l.client_ip, l.user_agent
        FROM consultant_query_log l
        WHERE {where_sql}
        ORDER BY l.created_at DESC, l.id DESC
        LIMIT %s OFFSET %s
        """,
        (*where_params, lim, off),
    )
    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        ts = d.get("created_at")
        if ts is not None and hasattr(ts, "isoformat"):
            d["created_at"] = ts.isoformat()
        out.append(d)
    return out


def count_consultant_queries(
    db: PostgresClient, *, filters: Optional[ConsultantQueryListFilters] = None
) -> int:
    fl = filters or ConsultantQueryListFilters()
    where_sql, where_params = _where_clause_for_filters(fl)
    r = db.execute_query(
        f"SELECT COUNT(*)::bigint AS c FROM consultant_query_log l WHERE {where_sql}",
        where_params,
    )
    if not r:
        return 0
    return int(r[0].get("c") or 0)


def delete_consultant_query_by_id(db: PostgresClient, row_id: int) -> int:
    """Delete one row by primary key; returns rows deleted (0 or 1)."""
    rid = int(row_id)
    if rid < 1:
        return 0
    return db.execute_update("DELETE FROM consultant_query_log WHERE id = %s", (rid,))


def delete_consultant_queries_by_ids(db: PostgresClient, ids: List[int]) -> int:
    """Delete up to 500 rows; invalid ids skipped."""
    clean: List[int] = []
    for x in ids[:500]:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if i >= 1:
            clean.append(i)
    if not clean:
        return 0
    ph = ",".join(["%s"] * len(clean))
    return db.execute_update(
        f"DELETE FROM consultant_query_log WHERE id IN ({ph})",
        tuple(clean),
    )
