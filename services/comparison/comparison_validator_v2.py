"""
Strict validator for Comparison v2 payloads — no partial tables or hybrid output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from services.comparison.comparison_schema_v2 import (
    AircraftEntryV2,
    ComparisonPayloadV2,
    InsufficientComparisonV2,
)

_BANNED_NAME_RE = re.compile(
    r"\b(?:unverified|unknown|tbd|n/?a|placeholder|partial)\b",
    re.I,
)

_VALID_CATEGORIES = frozenset(
    {"light", "super-midsize", "large-cabin", "ULR", "super_mid", "heavy", "midsize"}
)
_VALID_COST_BANDS = frozenset({"low", "medium", "high", "ultra"})


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str = ""


def _valid_winter_value(value: Any) -> bool:
    return value is True or value is False or value == "conditional"


def validate_aircraft_entry(entry: Dict[str, Any]) -> ValidationResult:
    name = str(entry.get("name") or "").strip()
    if not name or _BANNED_NAME_RE.search(name):
        return ValidationResult(ok=False, reason="invalid aircraft name")
    if len(name) < 4:
        return ValidationResult(ok=False, reason="partial aircraft name")

    category = str(entry.get("category") or "").strip()
    if not category:
        return ValidationResult(ok=False, reason="missing category")

    try:
        score = float(entry.get("mission_fit_score"))
    except (TypeError, ValueError):
        return ValidationResult(ok=False, reason="missing mission_fit_score")
    if not (0.0 <= score <= 1.0):
        return ValidationResult(ok=False, reason="mission_fit_score out of range")

    cost_band = str(entry.get("cost_band") or "").strip().lower()
    if cost_band not in _VALID_COST_BANDS:
        return ValidationResult(ok=False, reason="invalid cost_band")

    if not _valid_winter_value(entry.get("winter_westbound_capability")):
        return ValidationResult(ok=False, reason="invalid winter_westbound_capability")

    range_nm = entry.get("range_nm")
    if range_nm is not None:
        try:
            float(range_nm)
        except (TypeError, ValueError):
            return ValidationResult(ok=False, reason="invalid range_nm")

    seats = entry.get("seats")
    if seats is not None:
        try:
            int(seats)
        except (TypeError, ValueError):
            return ValidationResult(ok=False, reason="invalid seats")

    return ValidationResult(ok=True)


def validate_comparison_payload(
    payload: Union[ComparisonPayloadV2, InsufficientComparisonV2, Dict[str, Any]],
) -> ValidationResult:
    if not isinstance(payload, dict):
        return ValidationResult(ok=False, reason="payload not an object")

    if payload.get("status") == "INSUFFICIENT_DATA":
        if payload.get("mode") != "explicit_comparison":
            return ValidationResult(ok=False, reason="invalid insufficient mode")
        if not str(payload.get("reason") or "").strip():
            return ValidationResult(ok=False, reason="missing insufficient reason")
        return ValidationResult(ok=True)

    if payload.get("mode") != "explicit_comparison":
        return ValidationResult(ok=False, reason="mode must be explicit_comparison")

    aircraft = payload.get("aircraft")
    if not isinstance(aircraft, list) or len(aircraft) < 2:
        return ValidationResult(ok=False, reason="missing canonical aircraft set")

    names_seen: set[str] = set()
    for row in aircraft:
        if not isinstance(row, dict):
            return ValidationResult(ok=False, reason="invalid aircraft row")
        vr = validate_aircraft_entry(row)
        if not vr.ok:
            return vr
        name = str(row.get("name") or "").strip()
        if name in names_seen:
            return ValidationResult(ok=False, reason="duplicate aircraft name")
        names_seen.add(name)

    dq = payload.get("data_quality") or {}
    if dq.get("status") != "OK":
        return ValidationResult(ok=False, reason="data_quality must be OK for full payload")

    matrix = payload.get("comparison_matrix") or {}
    dims = matrix.get("dimensions")
    if not isinstance(dims, list) or len(dims) < 1:
        return ValidationResult(ok=False, reason="missing comparison_matrix dimensions")

    verdict = payload.get("verdict")
    if not isinstance(verdict, dict):
        return ValidationResult(ok=False, reason="missing verdict block")

    return ValidationResult(ok=True)


__all__ = ["ValidationResult", "validate_aircraft_entry", "validate_comparison_payload"]
