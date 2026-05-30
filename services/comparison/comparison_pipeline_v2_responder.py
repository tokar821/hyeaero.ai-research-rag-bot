"""
Comparison v2 user responder — structured spec contrast without mission orchestration.

Wraps ``run_comparison_v2`` and renders compact broker-facing contrast prose.
No ranked shortlists, kernel synthesis, or operational mission advisor flow.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from services.aircraft_truth.constants import UNIFIED_COMPARISON_INSUFFICIENT_MESSAGE
from services.consultant.mission_state import MissionState, build_mission_from_current_turn

_FORBIDDEN_PHRASES = re.compile(
    r"\b(?:good\s+fit|operational\s+synthesis|approved\s+shortlist|mission\s+authority|"
    r"shortlist|best\s+jet|recommend(?:ation)?|fleet\s+segmentation|mission[-\s]?fit|"
    r"operational\s+band|planning\s+band|viable\s+with\s+compromises)\b",
    re.I,
)

_INSUFFICIENT_COMPARISON = UNIFIED_COMPARISON_INSUFFICIENT_MESSAGE


def _guard_answer(text: str) -> str:
    if _FORBIDDEN_PHRASES.search(text or ""):
        return _INSUFFICIENT_COMPARISON
    return (text or "").strip()


def _resolve_compare_models(query: str) -> List[str]:
    from services.comparison.aircraft_registry_lock import lock_comparison_aircraft
    from services.consultant.recommendation_engine import detect_models_from_text

    lock = lock_comparison_aircraft(detect_models_from_text(query or ""))
    return list(lock.canonical)


def _insufficient_message(query: str, models: Optional[List[str]] = None) -> str:
    resolved = [m for m in (models or _resolve_compare_models(query)) if m]
    if len(resolved) == 1:
        return (
            f"Insufficient verified aircraft data to produce a comparison for {resolved[0]}."
        )
    if len(resolved) >= 2:
        pair = f"{resolved[0]} and {resolved[1]}"
        return f"Insufficient verified aircraft data to produce a comparison for {pair}."
    return _INSUFFICIENT_COMPARISON


def _format_structured_contrast(payload: Dict[str, Any]) -> str:
    if payload.get("status") == "INSUFFICIENT_DATA":
        return _INSUFFICIENT_COMPARISON

    aircraft = payload.get("aircraft") or []
    if not isinstance(aircraft, list) or len(aircraft) < 2:
        return _INSUFFICIENT_COMPARISON

    lines: List[str] = ["Verified catalog comparison:"]
    for entry in aircraft:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        category = entry.get("category") or "—"
        range_nm = entry.get("range_nm")
        seats = entry.get("seats")
        cost = entry.get("cost_band") or "—"
        range_s = f"{int(range_nm)} nm" if isinstance(range_nm, (int, float)) else "—"
        seat_s = str(int(seats)) if isinstance(seats, (int, float)) else "—"
        lines.append(
            f"- {name}: {category} class; practical range {range_s}; "
            f"seats {seat_s}; operating cost band {cost}."
        )

    if len(lines) < 3:
        return _INSUFFICIENT_COMPARISON

    a0, a1 = aircraft[0], aircraft[1]
    r0 = a0.get("range_nm")
    r1 = a1.get("range_nm")
    s0 = a0.get("seats")
    s1 = a1.get("seats")
    if isinstance(r0, (int, float)) and isinstance(r1, (int, float)) and r0 != r1:
        leader = a0 if r0 > r1 else a1
        trailer = a1 if leader is a0 else a0
        lines.append(
            f"On verified range, {leader.get('name')} leads {trailer.get('name')} "
            f"({int(max(r0, r1))} nm vs {int(min(r0, r1))} nm catalog practical)."
        )
    if isinstance(s0, (int, float)) and isinstance(s1, (int, float)) and s0 != s1:
        leader = a0 if s0 > s1 else a1
        trailer = a1 if leader is a0 else a0
        lines.append(
            f"On seating capacity, {leader.get('name')} offers more seats than "
            f"{trailer.get('name')} ({int(max(s0, s1))} vs {int(min(s0, s1))})."
        )

    return _guard_answer("\n".join(lines))


def respond_aircraft_comparison(
    query: str,
    *,
    mission: Optional[MissionState] = None,
    compare_models: Optional[List[str]] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Deterministic structured comparison for explicit aircraft contrast queries.

    Uses Comparison v2 only — no ranking pipeline or mission kernel.
    """
    models = [m for m in (compare_models or _resolve_compare_models(query)) if m]
    if len(models) < 2:
        return _insufficient_message(query, models)

    ms = mission if mission is not None else build_mission_from_current_turn(query or "")
    du = dict(data_used or {})

    from services.comparison.comparison_pipeline_v2 import run_comparison_v2

    raw = run_comparison_v2(
        query=query or "",
        mission=ms,
        compare_models=models,
        data_used=du,
        mode="explicit_comparison",
    )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return _insufficient_message(query, models)

    if isinstance(data_used, dict):
        data_used.update({k: v for k, v in du.items() if k not in data_used})

    if not isinstance(payload, dict):
        return _insufficient_message(query, models)

    return _format_structured_contrast(payload)


__all__ = ["respond_aircraft_comparison"]
