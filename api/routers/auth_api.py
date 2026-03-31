"""Public auth: register, login, me."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field

from api.auth_dependencies import get_current_user_optional, require_authenticated_user
from api.auth_dependencies import get_db_for_auth
from services.app_user_service import (
    ROLE_USER,
    STATUS_ACTIVE,
    STATUS_PENDING,
    create_user,
    get_user_by_email,
    normalize_email,
    try_login,
    user_row_public,
)
from services.auth_jwt import create_access_token, jwt_configured

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterBody(BaseModel):
    email: EmailStr
    full_name: str = Field(..., min_length=1, max_length=255)
    country: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=8, max_length=256)


class LoginBody(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserPublicResponse(BaseModel):
    id: int
    email: str
    full_name: str
    country: str
    role: str
    status: str
    created_at: str
    updated_at: str


def _strip_user(u: Dict[str, Any]) -> UserPublicResponse:
    data = user_row_public(u)
    return UserPublicResponse(**data)


@router.post("/register", response_model=UserPublicResponse)
def auth_register(body: RegisterBody):
    if not jwt_configured():
        raise HTTPException(
            status_code=503,
            detail="Registration unavailable (set JWT_SECRET on the server, min 16 characters)",
        )
    db = get_db_for_auth()
    if get_user_by_email(db, body.email):
        raise HTTPException(status_code=409, detail="An account with this email already exists")
    try:
        row = create_user(
            db,
            email=normalize_email(str(body.email)),
            full_name=body.full_name,
            country=body.country,
            password=body.password,
            role=ROLE_USER,
            status=STATUS_PENDING,
        )
        return UserPublicResponse(**row)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=TokenResponse)
def auth_login(body: LoginBody):
    if not jwt_configured():
        raise HTTPException(status_code=503, detail="Login unavailable (JWT_SECRET not configured)")
    db = get_db_for_auth()
    outcome, row = try_login(db, normalize_email(str(body.email)), body.password)
    if outcome == "pending":
        raise HTTPException(
            status_code=403,
            detail="Account pending activation",
        )
    if outcome == "rejected":
        raise HTTPException(
            status_code=403,
            detail="Account access denied",
        )
    if outcome != "ok" or not row:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(
        user_id=int(row["id"]),
        email=str(row["email"]),
        role=str(row["role"]),
    )
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserPublicResponse)
def auth_me(request: Request, user: dict = Depends(require_authenticated_user)):
    return _strip_user(user)


@router.get("/session", response_model=UserPublicResponse)
def auth_session_optional(request: Request):
    """Returns 401 if no valid Bearer token (for frontend boot)."""
    u = get_current_user_optional(request)
    if not u:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return _strip_user(u)
