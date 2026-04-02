"""
Ensure ``public`` tables used by Ask Consultant SQL still expose expected columns.

Skips when Postgres is not configured (local unit runs without DB).
"""

from __future__ import annotations

import pytest

from database.consultant_sql_columns import verify_consultant_sql_columns


def test_consultant_sql_columns_match_database():
    try:
        from config.config_loader import get_config
        from database.postgres_client import PostgresClient

        cfg = get_config()
        cs = cfg.postgres_connection_string
        if not cs:
            cs = "postgresql://{}:{}@{}:{}/{}".format(
                cfg.postgres_user,
                cfg.postgres_password or "",
                cfg.postgres_host,
                cfg.postgres_port or 5432,
                cfg.postgres_database,
            )
        db = PostgresClient(cs)
        db.execute_query("SELECT 1")
    except Exception:
        pytest.skip("PostgreSQL not available for schema check")

    problems = verify_consultant_sql_columns(db)
    assert not problems, "; ".join(problems)
