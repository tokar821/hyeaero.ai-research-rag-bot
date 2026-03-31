import services.app_user_service as aus


def test_try_login_pending_correct_password(monkeypatch):
    monkeypatch.setattr(
        aus,
        "get_user_by_email",
        lambda db, e: {
            "id": 1,
            "email": e,
            "password_hash": "h",
            "status": aus.STATUS_PENDING,
        },
    )
    monkeypatch.setattr(aus, "verify_password", lambda p, h: True)
    outcome, row = aus.try_login(None, "a@b.com", "secret")
    assert outcome == "pending"
    assert row is None


def test_try_login_active_correct_password(monkeypatch):
    row = {
        "id": 2,
        "email": "a@b.com",
        "password_hash": "h",
        "status": aus.STATUS_ACTIVE,
        "role": "user",
    }
    monkeypatch.setattr(aus, "get_user_by_email", lambda db, e: dict(row, email=e))
    monkeypatch.setattr(aus, "verify_password", lambda p, h: True)
    outcome, got = aus.try_login(None, "a@b.com", "secret")
    assert outcome == "ok"
    assert got["id"] == 2


def test_try_login_wrong_password(monkeypatch):
    monkeypatch.setattr(
        aus,
        "get_user_by_email",
        lambda db, e: {"id": 1, "password_hash": "h", "status": aus.STATUS_ACTIVE},
    )
    monkeypatch.setattr(aus, "verify_password", lambda p, h: False)
    outcome, row = aus.try_login(None, "a@b.com", "bad")
    assert outcome == "invalid"
    assert row is None
