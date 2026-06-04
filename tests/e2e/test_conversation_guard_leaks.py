"""Regression tests — conversation guard must not short-circuit hard deterministic queries."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.deterministic

from tests.conftest import run_retrieval


def test_guard_does_not_short_circuit_alternatives_to_longitude(mock_svc, enable_intent_lock, disable_fine_intent_llm):
    kind, payload = run_retrieval("Alternatives to Longitude", svc=mock_svc)
    du = payload.get("data_used") or {}
    assert kind == "professional"
    assert isinstance(du.get("intent_lock"), dict)
    assert du.get("authority_dispatch_kind") == "alternative"


def test_guard_does_not_short_circuit_alternatives_to_fake_aircraft(mock_svc, enable_intent_lock, disable_fine_intent_llm):
    kind, payload = run_retrieval("Alternatives to FakeJet9000", svc=mock_svc)
    du = payload.get("data_used") or {}
    assert kind == "professional"
    assert isinstance(du.get("intent_lock"), dict)
    assert du.get("authority_dispatch_safety_fallback") == "alternative"
