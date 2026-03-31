"""Admin user management (role admin or super_admin)."""

from __future__ import annotations

from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field

from api.auth_dependencies import assert_staff_can_manage_target, get_db_for_auth, require_staff_user
from services.app_user_service import (
    ROLE_SUPER,
    STATUS_ACTIVE,
    STATUS_PENDING,
    STATUS_REJECTED,
    count_users,
    create_user,
    delete_user_by_id,
    get_user_by_id,
    get_user_by_email,
    list_users,
    normalize_email,
    set_password,
    update_user_fields,
    user_row_public,
)

router = APIRouter(prefix="/api/admin/users", tags=["admin-users"])


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


class AdminCreateUserBody(BaseModel):
    email: EmailStr
    full_name: str = Field(..., min_length=1, max_length=255)
    country: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=8, max_length=256)
    role: Literal["user", "admin"] = "user"
    status: Literal["pending", "active", "rejected"] = "active"


class AdminPatchUserBody(BaseModel):
    full_name: Optional[str] = Field(None, min_length=1, max_length=255)
    country: Optional[str] = Field(None, min_length=1, max_length=100)
    role: Optional[Literal["user", "admin"]] = None
    status: Optional[Literal["pending", "active", "rejected"]] = None


class AdminResetPasswordBody(BaseModel):
    new_password: str = Field(..., min_length=8, max_length=256)


@router.get("", response_model=AdminUserListResponse2)
def admin_list_users(
    viewer: dict = Depends(require_staff_user),
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
):
    db = get_db_for_auth()
    st = status if status in (STATUS_PENDING, STATUS_ACTIVE, STATUS_REJECTED) else None
    items = list_users(db, viewer_role=viewer["role"], limit=limit, offset=offset, status=st)
    total = count_users(db, viewer_role=viewer["role"], status=st)
    return AdminUserListResponse2(total=total, items=[UserPublicResponse(**x) for x in items])


@router.post("", response_model=UserPublicResponse)
def admin_create_user(body: AdminCreateUserBody, viewer: dict = Depends(require_staff_user)):
    db = get_db_for_auth()
    if get_user_by_email(db, str(body.email)):
        raise HTTPException(status_code=409, detail="Email already registered")
    # Only super_admin can create a user that is already active (or rejected). Admins add pending accounts.
    effective_status = (
        body.status if viewer.get("role") == ROLE_SUPER else STATUS_PENDING
    )
    row = create_user(
        db,
        email=normalize_email(str(body.email)),
        full_name=body.full_name,
        country=body.country,
        password=body.password,
        role=body.role,
        status=effective_status,
    )
    return UserPublicResponse(**row)


@router.patch("/{user_id:int}", response_model=UserPublicResponse)
def admin_patch_user(
    user_id: int,
    body: AdminPatchUserBody,
    viewer: dict = Depends(require_staff_user),
):
    db = get_db_for_auth()
    target = get_user_by_id(db, user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    assert_staff_can_manage_target(viewer, target)
    if body.role is not None and target["role"] == ROLE_SUPER:
        raise HTTPException(status_code=400, detail="Cannot change super admin role via API")
    if body.status is not None and viewer.get("role") != ROLE_SUPER:
        raise HTTPException(
            status_code=403,
            detail="Only the super admin can change account status (activate, reject, or set pending)",
        )
    ok = update_user_fields(
        db,
        user_id,
        full_name=body.full_name,
        country=body.country,
        role=body.role,
        status=body.status,
    )
    if not ok:
        raise HTTPException(status_code=400, detail="No changes applied")
    fresh = get_user_by_id(db, user_id)
    if not fresh:
        raise HTTPException(status_code=404, detail="User not found")
    return UserPublicResponse(**user_row_public(fresh))


@router.post("/{user_id:int}/reset-password")
def admin_reset_password(
    user_id: int,
    body: AdminResetPasswordBody,
    viewer: dict = Depends(require_staff_user),
):
    db = get_db_for_auth()
    target = get_user_by_id(db, user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    assert_staff_can_manage_target(viewer, target)
    set_password(db, user_id, body.new_password)
    return {"ok": True}


@router.delete("/{user_id:int}")
def admin_delete_user(user_id: int, viewer: dict = Depends(require_staff_user)):
    db = get_db_for_auth()
    target = get_user_by_id(db, user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    assert_staff_can_manage_target(viewer, target)
    if int(viewer["id"]) == int(user_id):
        raise HTTPException(status_code=400, detail="You cannot delete your own account here")
    if target["role"] == ROLE_SUPER:
        raise HTTPException(status_code=400, detail="Delete the super admin only after promoting another (not supported via API)")
    n = delete_user_by_id(db, user_id)
    return {"deleted": n}
