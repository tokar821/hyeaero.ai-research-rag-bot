"""Ownership simulator — charter hours parsing."""

from __future__ import annotations

from services.orchestration.ownership_simulator import _parse_annual_hours


def test_charter_around_hours_annually():
    q = "We currently charter around 300 hours annually and are debating ownership."
    assert _parse_annual_hours(q) == 300


def test_hours_per_year():
    q = "We fly 250 hours per year."
    assert _parse_annual_hours(q) == 250
