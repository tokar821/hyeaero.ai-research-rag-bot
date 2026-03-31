"""Shared Postgres handle for dependencies that must not import ``api.main`` (circular imports)."""

from __future__ import annotations

from typing import Optional

from database.postgres_client import PostgresClient

_shared: Optional[PostgresClient] = None


def register_postgres_client(client: PostgresClient) -> None:
    global _shared
    _shared = client


def get_registered_postgres_client() -> Optional[PostgresClient]:
    return _shared
