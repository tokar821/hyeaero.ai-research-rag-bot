"""Broker formatting tests (Priority 7)."""

from __future__ import annotations

from services.rendering.broker_response_format import (
    format_comparison_intelligence_block,
    format_continuity_acknowledgment,
)
from services.rendering.prose_renderer_v2 import render_comparison_prose


def test_comparison_intelligence_in_prose():
    payload = {
        "aircraft": [{"name": "A"}, {"name": "B"}],
        "comparison_rows": [
            {
                "label": "Falcon 8X",
                "aircraft_id": "Falcon 8X",
                "cabin": "ULR",
                "range": 5600,
                "operating_economics": "high",
                "field_performance": "marginal",
                "verdict": "conditional",
                "maintenance_ecosystem": "trijet_specialist_network",
                "dispatch_maturity": "strong_ulr_with_westbound_caveats",
                "cabin_usability": "trijet_sleeping_berth_bias",
                "airport_flexibility": "moderate_field",
            },
            {
                "label": "Global 7500",
                "aircraft_id": "Global 7500",
                "cabin": "ULR",
                "range": 6600,
                "operating_economics": "high",
                "field_performance": "strong",
                "verdict": "strong",
                "maintenance_ecosystem": "strong_oem_program",
                "dispatch_maturity": "flagship_dispatch_maturity",
                "cabin_usability": "boardroom_ulr",
                "airport_flexibility": "ulr_runway_bias",
            },
        ],
        "verdict": {},
        "intelligence_dimensions": ["dispatch_maturity"],
    }
    text = render_comparison_prose(payload)
    assert "Operational tradeoffs" in text
    assert "Dispatch maturity" in text


def test_continuity_ack():
    ack = format_continuity_acknowledgment(
        {"reference_aircraft": "Gulfstream G650ER", "network_phrase": "prior_network"}
    )
    assert "G650ER" in ack
