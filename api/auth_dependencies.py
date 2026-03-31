"""FastAPI dependencies: JWT bearer user + role gates."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import HTTPException, Request

from config.config_loader import Config
from database.postgres_client import PostgresClient
from api.runtime_db import get_registered_postgres_client, register_postgres_client
from services.app_user_service import (
    ROLE_ADMIN,
    ROLE_SUPER,
    STATUS_ACTIVE,
    get_user_by_id,
)
from services.auth_jwt import decode_access_token


def get_db_for_auth() -> PostgresClient:
    db = get_registered_postgres_client()
    if db is not None:
        return db
    c = Config.from_env()
    if not c.postgres_connection_string:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured")
    db = PostgresClient(c.postgres_connection_string)
    register_postgres_client(db)
    return db


def get_current_user_optional(request: Request) -> Optional[Dict[str, Any]]:
    auth = (request.headers.get("Authorization") or "").strip()
    if not auth.lower().startswith("bearer "):
        return None
    token = auth[7:].strip()
    if not token:
        return None
    payload = decode_access_token(token)
    if not payload:
        return None
    try:
        uid = int(payload.get("sub"))
    except (TypeError, ValueError):
        return None
    db = get_registered_postgres_client()
    if db is None:
        return None
    row = get_user_by_id(db, uid)
    if not row or row.get("status") != STATUS_ACTIVE:
        return None
    return row


def require_authenticated_user(request: Request) -> Dict[str, Any]:
    u = get_current_user_optional(request)
    if not u:
        raise HTTPException(status_code=401, detail="Authentication required")
    return u


def require_staff_user(request: Request) -> Dict[str, Any]:
    u = require_authenticated_user(request)
    if u.get("role") not in (ROLE_ADMIN, ROLE_SUPER):
        raise HTTPException(status_code=403, detail="Admin access required")
    return u


def require_super_admin_user(request: Request) -> Dict[str, Any]:
    u = require_authenticated_user(request)
    if u.get("role") != ROLE_SUPER:
        raise HTTPException(status_code=403, detail="Super admin access required")
    return u


def assert_staff_can_manage_target(viewer: Dict[str, Any], target: Dict[str, Any]) -> None:
    """Admin cannot modify or delete the super admin account (super may manage all)."""
    if target.get("role") == ROLE_SUPER and viewer.get("role") != ROLE_SUPER:
        raise HTTPException(status_code=403, detail="Cannot manage the super admin account")
