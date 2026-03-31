"""Persistence and helpers for ``app_users``."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from database.postgres_client import PostgresClient
from services.auth_password import hash_password, verify_password

logger = logging.getLogger(__name__)

ROLE_USER = "user"
ROLE_ADMIN = "admin"
ROLE_SUPER = "super_admin"

STATUS_PENDING = "pending"
STATUS_ACTIVE = "active"
STATUS_REJECTED = "rejected"

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def user_row_public(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "email": row["email"],
        "full_name": row["full_name"],
        "country": row["country"],
        "role": row["role"],
        "status": row["status"],
        "created_at": row["created_at"].isoformat() if hasattr(row.get("created_at"), "isoformat") else row.get("created_at"),
        "updated_at": row["updated_at"].isoformat() if hasattr(row.get("updated_at"), "isoformat") else row.get("updated_at"),
    }


def count_super_admins(db: PostgresClient) -> int:
    r = db.execute_query("SELECT COUNT(*)::bigint AS c FROM app_users WHERE role = %s", (ROLE_SUPER,))
    return int(r[0]["c"]) if r else 0


def get_user_by_id(db: PostgresClient, user_id: int) -> Optional[Dict[str, Any]]:
    rows = db.execute_query(
        """
        SELECT id, email, full_name, country, password_hash, role, status, created_at, updated_at
        FROM app_users WHERE id = %s LIMIT 1
        """,
        (int(user_id),),
    )
    return dict(rows[0]) if rows else None


def get_user_by_email(db: PostgresClient, email: str) -> Optional[Dict[str, Any]]:
    em = normalize_email(email)
    rows = db.execute_query(
        """
        SELECT id, email, full_name, country, password_hash, role, status, created_at, updated_at
        FROM app_users WHERE lower(trim(email)) = %s LIMIT 1
        """,
        (em,),
    )
    return dict(rows[0]) if rows else None


def create_user(
    db: PostgresClient,
    *,
    email: str,
    full_name: str,
    country: str,
    password: Optional[str] = None,
    password_hash: Optional[str] = None,
    role: str = ROLE_USER,
    status: str = STATUS_PENDING,
) -> Dict[str, Any]:
    em = normalize_email(email)
    ph = password_hash if password_hash else hash_password(password or "")
    rows = db.execute_query(
        """
        INSERT INTO app_users (email, full_name, country, password_hash, role, status)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id, email, full_name, country, role, status, created_at, updated_at
        """,
        (em, full_name.strip(), country.strip(), ph, role, status),
    )
    if not rows:
        raise RuntimeError("insert returned no row")
    return user_row_public(dict(rows[0]))


def try_login(
    db: PostgresClient, email: str, password: str
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Check email + password and account status.

    Returns ``(outcome, row)`` where ``outcome`` is ``ok``, ``invalid``, ``pending``, or
    ``rejected``; ``row`` is set only when outcome is ``ok``.
    """
    row = get_user_by_email(db, email)
    if not row:
        return ("invalid", None)
    if not verify_password(password, str(row.get("password_hash") or "")):
        return ("invalid", None)
    status = str(row.get("status") or "")
    if status == STATUS_ACTIVE:
        return ("ok", row)
    if status == STATUS_PENDING:
        return ("pending", None)
    if status == STATUS_REJECTED:
        return ("rejected", None)
    return ("invalid", None)


def authenticate(db: PostgresClient, email: str, password: str) -> Optional[Dict[str, Any]]:
    """Return user row only if credentials are valid and status is **active**."""
    outcome, row = try_login(db, email, password)
    return row if outcome == "ok" else None


def list_users(
    db: PostgresClient,
    *,
    viewer_role: str,
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    lim = max(1, min(500, limit))
    off = max(0, offset)
    where = ["1=1"]
    params: List[Any] = []
    if viewer_role != ROLE_SUPER:
        where.append("role != %s")
        params.append(ROLE_SUPER)
    if status in (STATUS_PENDING, STATUS_ACTIVE, STATUS_REJECTED):
        where.append("status = %s")
        params.append(status)
    wh = " AND ".join(where)
    params.extend([lim, off])
    rows = db.execute_query(
        f"""
        SELECT id, email, full_name, country, role, status, created_at, updated_at
        FROM app_users WHERE {wh}
        ORDER BY id DESC
        LIMIT %s OFFSET %s
        """,
        tuple(params),
    )
    out = []
    for r in rows:
        out.append(user_row_public(dict(r)))
    return out


def count_users(db: PostgresClient, *, viewer_role: str, status: Optional[str] = None) -> int:
    where = ["1=1"]
    params: List[Any] = []
    if viewer_role != ROLE_SUPER:
        where.append("role != %s")
        params.append(ROLE_SUPER)
    if status in (STATUS_PENDING, STATUS_ACTIVE, STATUS_REJECTED):
        where.append("status = %s")
        params.append(status)
    wh = " AND ".join(where)
    r = db.execute_query(f"SELECT COUNT(*)::bigint AS c FROM app_users WHERE {wh}", tuple(params))
    return int(r[0]["c"]) if r else 0


def update_user_fields(
    db: PostgresClient,
    user_id: int,
    *,
    full_name: Optional[str] = None,
    country: Optional[str] = None,
    role: Optional[str] = None,
    status: Optional[str] = None,
) -> bool:
    sets: List[str] = []
    params: List[Any] = []
    if full_name is not None:
        sets.append("full_name = %s")
        params.append(full_name.strip())
    if country is not None:
        sets.append("country = %s")
        params.append(country.strip())
    if role is not None:
        sets.append("role = %s")
        params.append(role)
    if status is not None:
        sets.append("status = %s")
        params.append(status)
    if not sets:
        return True
    sets.append("updated_at = NOW()")
    params.append(int(user_id))
    sql = f"UPDATE app_users SET {', '.join(sets)} WHERE id = %s"
    n = db.execute_update(sql, tuple(params))
    return n > 0


def set_password(db: PostgresClient, user_id: int, new_plain: str) -> bool:
    ph = hash_password(new_plain)
    n = db.execute_update(
        "UPDATE app_users SET password_hash = %s, updated_at = NOW() WHERE id = %s",
        (ph, int(user_id)),
    )
    return n > 0


def delete_user_by_id(db: PostgresClient, user_id: int) -> int:
    return db.execute_update("DELETE FROM app_users WHERE id = %s", (int(user_id),))


def bootstrap_super_admin_from_env(db: PostgresClient) -> None:
    """Create the single super admin when none exists, using env credentials."""
    if count_super_admins(db) >= 1:
        return
    email = normalize_email(
        os.getenv("SUPER_ADMIN_EMAIL") or os.getenv("BOOTSTRAP_SUPER_ADMIN_EMAIL") or ""
    )
    password = (
        os.getenv("SUPER_ADMIN_PASSWORD") or os.getenv("BOOTSTRAP_SUPER_ADMIN_PASSWORD") or ""
    ).strip()
    full_name = (
        os.getenv("SUPER_ADMIN_FULL_NAME")
        or os.getenv("BOOTSTRAP_SUPER_ADMIN_FULL_NAME")
        or "Super Admin"
    ).strip() or "Super Admin"
    country = (
        os.getenv("SUPER_ADMIN_COUNTRY") or os.getenv("BOOTSTRAP_SUPER_ADMIN_COUNTRY") or "US"
    ).strip() or "US"
    if not email or not _EMAIL_RE.match(email):
        logger.info(
            "Super admin bootstrap skipped (set SUPER_ADMIN_EMAIL and SUPER_ADMIN_PASSWORD, or BOOTSTRAP_* aliases)"
        )
        return
    if len(password) < 8:
        logger.warning("Super admin bootstrap skipped (password must be at least 8 chars)")
        return
    try:
        create_user(
            db,
            email=email,
            full_name=full_name,
            country=country,
            password=password,
            role=ROLE_SUPER,
            status=STATUS_ACTIVE,
        )
        logger.info("Bootstrap super admin created for %s", email)
    except Exception as e:
        logger.error("Bootstrap super admin failed: %s", e)
