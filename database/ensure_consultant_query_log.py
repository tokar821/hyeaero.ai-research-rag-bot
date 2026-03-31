"""
Apply ``ensure_consultant_query_log.sql`` on API startup when PostgreSQL is configured.

Safe to call repeatedly; statements are idempotent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from database.postgres_client import PostgresClient
from database.split_sql import sql_statements

logger = logging.getLogger(__name__)

_MIGRATION = Path(__file__).resolve().parent / "migrations" / "ensure_consultant_query_log.sql"


def apply_consultant_query_log_schema(db: PostgresClient) -> bool:
    """Create ``consultant_query_log`` and indexes if missing."""
    if not _MIGRATION.is_file():
        logger.error("Migration file missing: %s", _MIGRATION)
        return False
    sql_text = _MIGRATION.read_text(encoding="utf-8")
    statements = sql_statements(sql_text)
    for i, stmt in enumerate(statements, 1):
        try:
            db.execute_update(stmt)
        except Exception as e:
            msg = str(e).lower()
            if "already exists" in msg or "duplicate" in msg:
                logger.debug("Statement %s skipped (already applied): %s", i, e)
                continue
            logger.error("Migration statement %s failed: %s\n%s", i, e, stmt[:200])
            return False
    logger.info("consultant_query_log schema ensured (%s statements)", len(statements))
    return True
