"""Super admin–only endpoints (admin roster)."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth_dependencies import get_db_for_auth, require_super_admin_user
from services.app_user_service import ROLE_ADMIN, user_row_public

router = APIRouter(prefix="/api/super-admin", tags=["super-admin"])


class UserPublicResponse(BaseModel):
    id: int
    email: str
    full_name: str
    country: str
    role: str
    status: str
    created_at: str
    updated_at: str


class AdminUserListResponse2(BaseModel):
    total: int
    items: List[UserPublicResponse]


@router.get("/admins", response_model=AdminUserListResponse2)
def list_admin_accounts(_viewer: dict = Depends(require_super_admin_user)):
    """Users with role ``admin`` (super admin manages this roster)."""
    db = get_db_for_auth()
    rows = db.execute_query(
        """
        SELECT id, email, full_name, country, role, status, created_at, updated_at
        FROM app_users
        WHERE role = %s
        ORDER BY id DESC
        """,
        (ROLE_ADMIN,),
    )
    items = [UserPublicResponse(**user_row_public(dict(r))) for r in rows]
    return AdminUserListResponse2(total=len(items), items=items)
