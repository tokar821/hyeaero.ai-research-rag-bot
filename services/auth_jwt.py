"""JWT access tokens for dashboard users."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt


def _secret() -> str:
    return (os.getenv("JWT_SECRET") or "").strip()


def jwt_configured() -> bool:
    return len(_secret()) >= 16


def create_access_token(*, user_id: int, email: str, role: str) -> str:
    secret = _secret()
    if len(secret) < 16:
        raise RuntimeError("JWT_SECRET must be set (min 16 characters)")
    try:
        days = int((os.getenv("JWT_EXPIRE_DAYS") or "7").strip())
    except ValueError:
        days = 7
    days = max(1, min(90, days))
    exp = datetime.now(timezone.utc) + timedelta(days=days)
    payload: Dict[str, Any] = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "exp": exp,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    secret = _secret()
    if not secret:
        return None
    try:
        return jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
