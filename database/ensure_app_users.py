"""Apply ``ensure_app_users.sql`` and optional bootstrap super admin."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from database.postgres_client import PostgresClient
from database.split_sql import sql_statements

logger = logging.getLogger(__name__)

_MIGRATION = Path(__file__).resolve().parent / "migrations" / "ensure_app_users.sql"


def apply_app_users_schema(db: PostgresClient) -> bool:
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
                logger.debug("Statement %s skipped: %s", i, e)
                continue
            logger.error("Migration statement %s failed: %s\n%s", i, e, stmt[:220])
            return False
    logger.info("app_users schema ensured (%s statements)", len(statements))
    _ensure_consultant_query_log_user_fk(db)
    return True


def _ensure_consultant_query_log_user_fk(db: PostgresClient) -> None:
    """Add ``user_id`` to ``consultant_query_log`` when that table already exists (analytics enabled)."""
    try:
        rows = db.execute_query(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'consultant_query_log'
            LIMIT 1
            """
        )
        if not rows:
            return
        db.execute_update(
            """
            ALTER TABLE consultant_query_log
            ADD COLUMN IF NOT EXISTS user_id BIGINT NULL REFERENCES app_users(id) ON DELETE SET NULL
            """
        )
        db.execute_update(
            "CREATE INDEX IF NOT EXISTS consultant_query_log_user_id_idx ON consultant_query_log (user_id)"
        )
    except Exception as e:
        logger.warning("consultant_query_log.user_id migration skipped: %s", e)
