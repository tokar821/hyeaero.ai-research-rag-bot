"""
Renderer payload v2 — strict response envelope for frontend contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Component identifiers (stable UI contract)
COMPONENT_COMPARISON_TABLE_V2 = "comparison_table_v2"
COMPONENT_CAPABILITY_VERDICT_V2 = "capability_verdict_v2"
COMPONENT_STRATEGIC_ANALYSIS_V2 = "strategic_analysis_v2"
COMPONENT_NETWORK_TOPOLOGY_V2 = "network_topology_v2"
COMPONENT_BROKER_RECOMMENDATION_V2 = "broker_recommendation_v2"
COMPONENT_RENDERER_FAILURE = "renderer_failure"

VALID_MODES = frozenset(
    {
        "explicit_comparison",
        "named_aircraft_capability",
        "strategic_fleet_analysis",
        "network_structure",
        "recommendation_request",
        "error",
    }
)


@dataclass
class RendererEnvelopeV2:
    mode: str
    component: str
    payload: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "component": self.component,
            "payload": dict(self.payload),
            "meta": dict(self.meta),
        }


def renderer_failure_envelope(reason: str = "INSUFFICIENT_RENDER_DATA") -> RendererEnvelopeV2:
    return RendererEnvelopeV2(
        mode="error",
        component=COMPONENT_RENDERER_FAILURE,
        payload={"reason": reason},
        meta={"fail_closed": True},
    )


def comparison_rows_from_aircraft(aircraft: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministic comparison rows for UI — full dimension contract, no markdown."""
    rows: List[Dict[str, Any]] = []
    for ac in aircraft:
        if not isinstance(ac, dict):
            continue
        name = str(ac.get("name") or "").strip()
        if not name:
            continue
        cost_band = ac.get("cost_band")
        verdict_class = _fit_class_from_score(ac.get("mission_fit_score"))
        rows.append(
            {
                "aircraft_id": name,
                "label": name,
                "cabin": ac.get("category"),
                "range": ac.get("range_nm"),
                "operating_economics": cost_band,
                "field_performance": ac.get("winter_westbound_capability"),
                "verdict": verdict_class,
                "category": ac.get("category"),
                "range_nm": ac.get("range_nm"),
                "seats": ac.get("seats"),
                "mission_fit_score": ac.get("mission_fit_score"),
                "cost_band": cost_band,
                "winter_westbound_capability": ac.get("winter_westbound_capability"),
                "verdict_class": verdict_class,
            }
        )
    return rows


def _fit_class_from_score(score: Any) -> str:
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "unknown"
    if s >= 0.75:
        return "strong"
    if s >= 0.6:
        return "conditional"
    return "weak"


__all__ = [
    "RendererEnvelopeV2",
    "COMPONENT_BROKER_RECOMMENDATION_V2",
    "COMPONENT_CAPABILITY_VERDICT_V2",
    "COMPONENT_COMPARISON_TABLE_V2",
    "COMPONENT_NETWORK_TOPOLOGY_V2",
    "COMPONENT_RENDERER_FAILURE",
    "COMPONENT_STRATEGIC_ANALYSIS_V2",
    "VALID_MODES",
    "comparison_rows_from_aircraft",
    "renderer_failure_envelope",
]
