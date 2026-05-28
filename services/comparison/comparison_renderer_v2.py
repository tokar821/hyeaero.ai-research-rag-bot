"""
Comparison v2 renderer — JSON schema output only (no prose, no markdown tables).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Union

from services.comparison.comparison_schema_v2 import (
    ComparisonPayloadV2,
    InsufficientComparisonV2,
)
from services.comparison.comparison_validator_v2 import validate_comparison_payload

ComparisonOutput = Union[ComparisonPayloadV2, InsufficientComparisonV2]


def render_comparison_v2(
    payload: ComparisonOutput,
    *,
    mode: str = "explicit_comparison",
) -> str:
    """
    Render schema-compliant JSON only.

    Raises ValueError if mode is not explicit_comparison or validation fails.
    """
    if mode != "explicit_comparison":
        raise ValueError("comparison_renderer_v2 only supports explicit_comparison mode")

    vr = validate_comparison_payload(payload)
    if not vr.ok:
        if payload.get("status") == "INSUFFICIENT_DATA":
            return json.dumps(payload, indent=2, ensure_ascii=False)
        raise ValueError(f"comparison payload validation failed: {vr.reason}")

    return json.dumps(payload, indent=2, ensure_ascii=False)


def render_insufficient_data(reason: str) -> str:
    """Strict insufficient response — JSON only, no prose."""
    body: InsufficientComparisonV2 = {
        "mode": "explicit_comparison",
        "status": "INSUFFICIENT_DATA",
        "reason": (reason or "missing canonical aircraft set").strip(),
    }
    return json.dumps(body, indent=2, ensure_ascii=False)


__all__ = ["render_comparison_v2", "render_insufficient_data"]
