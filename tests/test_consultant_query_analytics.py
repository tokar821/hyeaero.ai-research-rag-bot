"""Unit tests for consultant query logging toggles (no PostgreSQL required)."""

from unittest.mock import MagicMock

from services.consultant_query_analytics import (
    consultant_query_analytics_enabled,
    record_consultant_query,
)


def test_analytics_enabled_when_unset(monkeypatch):
    monkeypatch.delenv("CONSULTANT_QUERY_ANALYTICS_ENABLED", raising=False)
    assert consultant_query_analytics_enabled() is True


def test_analytics_enabled_when_explicit_truthy(monkeypatch):
    monkeypatch.setenv("CONSULTANT_QUERY_ANALYTICS_ENABLED", "1")
    assert consultant_query_analytics_enabled() is True


def test_analytics_disabled_when_explicit_off(monkeypatch):
    monkeypatch.setenv("CONSULTANT_QUERY_ANALYTICS_ENABLED", "0")
    assert consultant_query_analytics_enabled() is False


def test_record_no_op_when_disabled(monkeypatch):
    monkeypatch.setenv("CONSULTANT_QUERY_ANALYTICS_ENABLED", "0")
    db = MagicMock()
    assert record_consultant_query(db, query="who owns N12345", endpoint="sync") is None
    db.execute_query.assert_not_called()
    db.execute_update.assert_not_called()


def test_record_inserts_when_enabled(monkeypatch):
    monkeypatch.setenv("CONSULTANT_QUERY_ANALYTICS_ENABLED", "1")
    db = MagicMock()
    db.execute_query.return_value = [{"id": 42}]
    rid = record_consultant_query(
        db,
        query="  market for citation  ",
        endpoint="stream",
        history_turn_count=4,
        client_ip="203.0.113.1",
        user_agent="pytest",
        user_email="u@example.com",
        user_full_name="User Name",
    )
    db.execute_query.assert_called_once()
    assert rid == 42
    params = db.execute_query.call_args[0][1]
    assert "market for citation" in params[0]
    assert "u@example.com" in params


def test_record_skips_empty_query(monkeypatch):
    monkeypatch.delenv("CONSULTANT_QUERY_ANALYTICS_ENABLED", raising=False)
    db = MagicMock()
    assert record_consultant_query(db, query="   ", endpoint="sync") is None
    db.execute_query.assert_not_called()


def test_where_clause_endpoint_and_search():
    from services.consultant_query_analytics import (
        ConsultantQueryListFilters,
        _where_clause_for_filters,
    )

    sql, params = _where_clause_for_filters(ConsultantQueryListFilters())
    assert sql == "TRUE"
    assert params == tuple()

    sql, params = _where_clause_for_filters(
        ConsultantQueryListFilters(endpoint="stream", q="50% off")
    )
    assert "endpoint = %s" in sql
    assert "answer_text" in sql
    assert "ILIKE" in sql
    assert "stream" in params
    assert sum(1 for p in params if isinstance(p, str) and "50" in p and "off" in p) >= 2
