"""Phase 34.4 — deterministic comparison catalog alias resolution."""

from __future__ import annotations

import pytest

from services.aircraft.aircraft_authority_service import resolve_aircraft_alias
from services.comparison.aircraft_registry_lock import lock_comparison_aircraft, resolve_to_registry_name
from services.comparison.comparison_pipeline_v2_responder import _resolve_compare_models
from services.consultant.recommendation_engine import detect_models_from_text

pytestmark = pytest.mark.deterministic


@pytest.mark.parametrize(
    "token,expected",
    [
        ("G700", "Gulfstream G700"),
        ("Gulfstream G700", "Gulfstream G700"),
        ("Longitude", "Citation Longitude"),
        ("Citation Longitude", "Citation Longitude"),
        ("Cessna Citation Longitude", "Citation Longitude"),
    ],
)
def test_resolve_aircraft_alias_canonical(token: str, expected: str) -> None:
    assert resolve_aircraft_alias(token) == expected
    assert resolve_to_registry_name(token) == expected


def test_g650_vs_g700_compare_models_length_two() -> None:
    models = _resolve_compare_models("G650 vs G700")
    assert len(models) == 2
    assert "Gulfstream G650" in models
    assert "Gulfstream G700" in models


def test_g650_vs_longitude_compare_models_length_two() -> None:
    models = _resolve_compare_models("G650 vs Longitude")
    assert len(models) == 2
    assert "Gulfstream G650" in models
    assert "Citation Longitude" in models


def test_lock_comparison_aircraft_rejects_unknown_only() -> None:
    lock = lock_comparison_aircraft(["G700", "G650"])
    assert len(lock.canonical) == 2
    assert not lock.rejected


def test_detect_models_includes_longitude_shorthand() -> None:
    found = detect_models_from_text("G650 vs Longitude")
    assert "Citation Longitude" in found or "Gulfstream G650" in found
    assert len(found) >= 2
