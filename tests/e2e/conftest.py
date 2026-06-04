"""Phase 48 e2e — broker certification report hook (shared with other e2e tests)."""

from __future__ import annotations

import pytest

from tests.e2e.broker_certification_helpers import (
    get_certification_recorder,
    reset_certification_recorder,
)


@pytest.fixture(scope="session", autouse=True)
def _broker_cert_session_recorder():
    reset_certification_recorder()
    yield
    get_certification_recorder().write_report()


def pytest_sessionfinish(session, exitstatus):
    try:
        get_certification_recorder().write_report()
    except Exception:
        pass
