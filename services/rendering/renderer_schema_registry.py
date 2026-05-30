"""
Renderer schema registry — validate envelopes before returning to clients.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from services.rendering.renderer_payload_v2 import (
    COMPONENT_BROKER_RECOMMENDATION_V2,
    COMPONENT_CAPABILITY_VERDICT_V2,
    COMPONENT_COMPARISON_TABLE_V2,
    COMPONENT_NETWORK_TOPOLOGY_V2,
    COMPONENT_RENDERER_FAILURE,
    COMPONENT_STRATEGIC_ANALYSIS_V2,
    VALID_MODES,
)

_BANNED_PROSE_MARKERS = re.compile(
    r"(?:##\s*STRATEGIC|##\s*NETWORK|##\s*Ranked|\|\s*Rank\s*\|)",
    re.I,
)

_BANNED_CAPABILITY_MARKERS = re.compile(
    r"\b(?:shortlist|recommend(?:ation)?|alternatives?|substitute)\b",
    re.I,
)


@dataclass(frozen=True)
class SchemaValidationResult:
    ok: bool
    reason: str = ""


def _require_keys(payload: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k not in payload:
            return f"missing key: {k}"
    return None


def _validate_comparison(payload: Dict[str, Any]) -> SchemaValidationResult:
    if payload.get("status") == "INSUFFICIENT_DATA":
        if not str(payload.get("reason") or "").strip():
            return SchemaValidationResult(False, "insufficient comparison missing reason")
        return SchemaValidationResult(True)

    if payload.get("comparison_type") == "strategy_vs_strategy":
        err = _require_keys(payload, ["strategies", "comparison_rows", "verdict"])
        if err:
            return SchemaValidationResult(False, err)
        return SchemaValidationResult(True)

    err = _require_keys(payload, ["aircraft", "comparison_rows", "verdict"])
    if err:
        return SchemaValidationResult(False, err)
    aircraft = payload.get("aircraft")
    rows = payload.get("comparison_rows")
    if not isinstance(aircraft, list) or len(aircraft) < 2:
        return SchemaValidationResult(False, "comparison requires >=2 aircraft")
    if not isinstance(rows, list) or len(rows) != len(aircraft):
        return SchemaValidationResult(False, "comparison_rows length mismatch")
    for ac in aircraft:
        if not str((ac or {}).get("name") or "").strip():
            return SchemaValidationResult(False, "aircraft name required")
    comparison_rows = payload.get("comparison_rows")
    if not isinstance(comparison_rows, list):
        return SchemaValidationResult(False, "comparison_rows must be list")
    for row in comparison_rows:
        if not isinstance(row, dict):
            return SchemaValidationResult(False, "invalid comparison row")
        for col in ("cabin", "range", "operating_economics", "field_performance", "verdict"):
            if col not in row:
                return SchemaValidationResult(False, f"comparison row missing {col}")
    return SchemaValidationResult(True)


def _validate_capability(payload: Dict[str, Any]) -> SchemaValidationResult:
    err = _require_keys(payload, ["aircraft", "mission", "verdict", "constraints"])
    if err:
        return SchemaValidationResult(False, err)
    if not str(payload.get("aircraft") or "").strip():
        return SchemaValidationResult(False, "capability aircraft required")
    if not isinstance(payload.get("constraints"), list):
        return SchemaValidationResult(False, "constraints must be list")
    blob = str(payload)
    if _BANNED_CAPABILITY_MARKERS.search(blob):
        return SchemaValidationResult(False, "capability payload contamination")
    return SchemaValidationResult(True)


def _validate_strategic(payload: Dict[str, Any]) -> SchemaValidationResult:
    err = _require_keys(payload, ["conflicts", "operational_domains", "recommendation"])
    if err:
        return SchemaValidationResult(False, err)
    if not isinstance(payload.get("conflicts"), list):
        return SchemaValidationResult(False, "conflicts must be list")
    if not isinstance(payload.get("operational_domains"), list):
        return SchemaValidationResult(False, "operational_domains must be list")
    if payload.get("shortlist"):
        return SchemaValidationResult(False, "strategic must not include shortlist")
    return SchemaValidationResult(True)


def _validate_network(payload: Dict[str, Any]) -> SchemaValidationResult:
    err = _require_keys(
        payload,
        ["primary_hubs", "secondary_hubs", "episodic_routes", "planning_priority"],
    )
    if err:
        return SchemaValidationResult(False, err)
    for key in ("primary_hubs", "secondary_hubs", "episodic_routes", "planning_priority"):
        if not isinstance(payload.get(key), list):
            return SchemaValidationResult(False, f"{key} must be list")
    return SchemaValidationResult(True)


def _validate_recommendation(payload: Dict[str, Any]) -> SchemaValidationResult:
    err = _require_keys(payload, ["shortlist", "mission", "verdict"])
    if err:
        return SchemaValidationResult(False, err)
    if not isinstance(payload.get("shortlist"), list):
        return SchemaValidationResult(False, "shortlist must be list")
    return SchemaValidationResult(True)


def _validate_error(payload: Dict[str, Any]) -> SchemaValidationResult:
    if not str(payload.get("reason") or "").strip():
        return SchemaValidationResult(False, "error reason required")
    return SchemaValidationResult(True)


RENDERER_SCHEMA_REGISTRY: Dict[str, Dict[str, Any]] = {
    "explicit_comparison": {
        "component": COMPONENT_COMPARISON_TABLE_V2,
        "validate": _validate_comparison,
    },
    "named_aircraft_capability": {
        "component": COMPONENT_CAPABILITY_VERDICT_V2,
        "validate": _validate_capability,
    },
    "strategic_fleet_analysis": {
        "component": COMPONENT_STRATEGIC_ANALYSIS_V2,
        "validate": _validate_strategic,
    },
    "network_structure": {
        "component": COMPONENT_NETWORK_TOPOLOGY_V2,
        "validate": _validate_network,
    },
    "recommendation_request": {
        "component": COMPONENT_BROKER_RECOMMENDATION_V2,
        "validate": _validate_recommendation,
    },
    "error": {
        "component": COMPONENT_RENDERER_FAILURE,
        "validate": _validate_error,
    },
}

MODE_TO_COMPONENT = {k: v["component"] for k, v in RENDERER_SCHEMA_REGISTRY.items()}


def validate_envelope(envelope: Dict[str, Any]) -> SchemaValidationResult:
    if not isinstance(envelope, dict):
        return SchemaValidationResult(False, "envelope not object")
    mode = str(envelope.get("mode") or "")
    component = str(envelope.get("component") or "")
    payload = envelope.get("payload")
    if mode not in VALID_MODES:
        return SchemaValidationResult(False, f"invalid mode: {mode}")
    spec = RENDERER_SCHEMA_REGISTRY.get(mode)
    if spec is None:
        return SchemaValidationResult(False, f"no schema for mode: {mode}")
    if component != spec["component"]:
        return SchemaValidationResult(False, f"component mismatch for {mode}")
    if not isinstance(payload, dict):
        return SchemaValidationResult(False, "payload must be object")
    validator: Callable[[Dict[str, Any]], SchemaValidationResult] = spec["validate"]
    return validator(payload)


def assert_no_markdown_in_payload(payload: Dict[str, Any]) -> SchemaValidationResult:
    text = str(payload)
    if _BANNED_PROSE_MARKERS.search(text):
        return SchemaValidationResult(False, "markdown table or prose marker in payload")
    return SchemaValidationResult(True)


__all__ = [
    "RENDERER_SCHEMA_REGISTRY",
    "MODE_TO_COMPONENT",
    "SchemaValidationResult",
    "assert_no_markdown_in_payload",
    "validate_envelope",
]
