"""Renderer envelope structure tests."""

from __future__ import annotations

import json

from services.rendering.renderer_payload_v2 import (
    RendererEnvelopeV2,
    comparison_rows_from_aircraft,
    renderer_failure_envelope,
)
from services.rendering.renderer_schema_registry import validate_envelope


def test_envelope_has_required_keys():
    env = RendererEnvelopeV2(
        mode="explicit_comparison",
        component="comparison_table_v2",
        payload={"status": "INSUFFICIENT_DATA", "reason": "test"},
        meta={"schema_version": "v2"},
    )
    d = env.to_dict()
    assert set(d.keys()) == {"mode", "component", "payload", "meta"}


def test_fail_closed_error_envelope():
    env = renderer_failure_envelope("INSUFFICIENT_RENDER_DATA")
    vr = validate_envelope(env.to_dict())
    assert vr.ok
    assert env.mode == "error"
    assert env.payload["reason"] == "INSUFFICIENT_RENDER_DATA"


def test_comparison_rows_deterministic():
    aircraft = [
        {
            "name": "Global 7500",
            "category": "ULR",
            "mission_fit_score": 0.82,
            "cost_band": "ultra",
            "winter_westbound_capability": True,
        },
        {
            "name": "Falcon 8X",
            "category": "ULR",
            "mission_fit_score": 0.71,
            "cost_band": "high",
            "winter_westbound_capability": "conditional",
        },
    ]
    rows = comparison_rows_from_aircraft(aircraft)
    assert len(rows) == 2
    assert rows[0]["aircraft_id"] == "Global 7500"
    assert rows[0]["verdict_class"] == "strong"
    assert "|" not in json.dumps(rows)
