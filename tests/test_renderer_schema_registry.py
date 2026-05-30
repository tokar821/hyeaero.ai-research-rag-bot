"""Renderer schema registry validation tests."""

from __future__ import annotations

from services.rendering.renderer_schema_registry import (
    RENDERER_SCHEMA_REGISTRY,
    validate_envelope,
)


def test_registry_covers_all_modes():
    assert "explicit_comparison" in RENDERER_SCHEMA_REGISTRY
    assert "named_aircraft_capability" in RENDERER_SCHEMA_REGISTRY
    assert "strategic_fleet_analysis" in RENDERER_SCHEMA_REGISTRY
    assert "network_structure" in RENDERER_SCHEMA_REGISTRY
    assert "recommendation_request" in RENDERER_SCHEMA_REGISTRY


def test_valid_comparison_envelope():
    env = {
        "mode": "explicit_comparison",
        "component": "comparison_table_v2",
        "payload": {
            "aircraft": [
                {"name": "Global 7500", "mission_fit_score": 0.8},
                {"name": "Falcon 8X", "mission_fit_score": 0.7},
            ],
            "comparison_rows": [
                {
                    "aircraft_id": "Global 7500",
                    "label": "Global 7500",
                    "cabin": "ULR",
                    "range": 7700,
                    "operating_economics": "high",
                    "field_performance": "strong",
                    "verdict": "strong",
                },
                {
                    "aircraft_id": "Falcon 8X",
                    "label": "Falcon 8X",
                    "cabin": "ULR",
                    "range": 6450,
                    "operating_economics": "high",
                    "field_performance": "marginal",
                    "verdict": "good",
                },
            ],
            "verdict": {"best_overall": "Global 7500"},
        },
        "meta": {},
    }
    assert validate_envelope(env).ok


def test_invalid_comparison_missing_rows_fails():
    env = {
        "mode": "explicit_comparison",
        "component": "comparison_table_v2",
        "payload": {
            "aircraft": [{"name": "A"}, {"name": "B"}],
            "verdict": {},
        },
        "meta": {},
    }
    assert not validate_envelope(env).ok


def test_insufficient_comparison_payload_valid():
    env = {
        "mode": "explicit_comparison",
        "component": "comparison_table_v2",
        "payload": {"status": "INSUFFICIENT_DATA", "reason": "missing set"},
        "meta": {},
    }
    assert validate_envelope(env).ok
