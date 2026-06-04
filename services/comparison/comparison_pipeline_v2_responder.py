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
_INSUFFICIENT_DATA = "INSUFFICIENT_DATA: Insufficient verified aircraft data to produce a comparison."


def _guard_answer(text: str) -> str:
    if _FORBIDDEN_PHRASES.search(text or ""):
        return _INSUFFICIENT_COMPARISON
    return (text or "").strip()


def _resolve_compare_models(query: str) -> List[str]:
    from services.aircraft.aircraft_authority_service import resolve_aircraft_alias
    from services.comparison.aircraft_registry_lock import lock_comparison_aircraft
    from services.consultant.recommendation_engine import detect_models_from_text

    raw = detect_models_from_text(query or "")
    resolved = [resolve_aircraft_alias(m) or m for m in raw]
    lock = lock_comparison_aircraft(resolved)
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


def _field_present(val: Any) -> bool:
    if val is None:
        return False
    s = str(val).strip().lower()
    return bool(s) and s not in ("—", "-", "unknown", "n/a", "none")


def _comparison_fields_complete(aircraft: List[Dict[str, Any]]) -> bool:
    if len(aircraft) < 2:
        return False
    for entry in aircraft[:2]:
        if not isinstance(entry, dict):
            return False
        if not _field_present(entry.get("category")):
            return False
        if not _field_present(entry.get("cost_band")):
            return False
    return True


def _format_cabin_cost_deltas(a0: Dict[str, Any], a1: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    n0 = str(a0.get("name") or "").strip()
    n1 = str(a1.get("name") or "").strip()
    c0 = str(a0.get("category") or "").strip()
    c1 = str(a1.get("category") or "").strip()
    if c0 and c1:
        if c0 != c1:
            lines.append(
                f"On cabin class, {n0} is {c0} versus {n1} at {c1} (verified catalog)."
            )
        else:
            lines.append(
                f"On cabin class, both aircraft sit in the {c0} segment (verified catalog)."
            )
    cost0 = str(a0.get("cost_band") or "").strip()
    cost1 = str(a1.get("cost_band") or "").strip()
    if cost0 and cost1:
        if cost0 != cost1:
            lines.append(
                f"On operating cost, {n0} is in the {cost0} band versus "
                f"{n1} in the {cost1} band (catalog operating index)."
            )
        else:
            lines.append(
                f"On operating cost, both aircraft share the {cost0} verified operating cost band."
            )
    return lines


def _format_verdict_section(
    payload: Dict[str, Any],
    a0: Dict[str, Any],
    a1: Dict[str, Any],
) -> str:
    n0 = str(a0.get("name") or "Aircraft A").strip()
    n1 = str(a1.get("name") or "Aircraft B").strip()
    verdict = payload.get("verdict") if isinstance(payload.get("verdict"), dict) else {}
    no_fit = str(verdict.get("no_fit_reason") or "").strip()
    if no_fit:
        return f"VERDICT:\n{_INSUFFICIENT_DATA}"

    best = str(verdict.get("best_overall") or "").strip()
    conditional = str(verdict.get("conditional_winner") or "").strip()
    if best and conditional and best != conditional:
        return (
            f"VERDICT:\n"
            f"Choose {best} if mission-fit and operating profile favor the primary catalog match; "
            f"otherwise choose {conditional} when you need a strong alternate with comparable capability."
        )
    if best:
        alternate = n1 if best == n0 else n0
        return (
            f"VERDICT:\n"
            f"Choose {best} if range, cabin, and operating cost tradeoffs align with your priority; "
            f"otherwise consider {alternate} when a different balance of capability is acceptable."
        )

    r0 = a0.get("range_nm")
    r1 = a1.get("range_nm")
    if isinstance(r0, (int, float)) and isinstance(r1, (int, float)) and r0 != r1:
        leader = a0 if r0 > r1 else a1
        trailer = a1 if leader is a0 else a0
        return (
            f"VERDICT:\n"
            f"Choose {leader.get('name')} if maximum verified range is the deciding factor; "
            f"otherwise choose {trailer.get('name')} when cabin and operating cost bands matter more."
        )
    return (
        f"VERDICT:\n"
        f"Choose {n0} if your mission prioritizes the first aircraft's verified profile; "
        f"otherwise choose {n1} when the alternate tradeoffs better match your constraints."
    )


def _format_structured_contrast(payload: Dict[str, Any]) -> str:
    if payload.get("status") == "INSUFFICIENT_DATA":
        return _INSUFFICIENT_DATA

    aircraft = payload.get("aircraft") or []
    if not isinstance(aircraft, list) or len(aircraft) < 2:
        return _INSUFFICIENT_DATA

    if not _comparison_fields_complete(aircraft):
        return _INSUFFICIENT_DATA

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
        return _INSUFFICIENT_DATA

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

    lines.extend(_format_cabin_cost_deltas(a0, a1))
    lines.append(_format_verdict_section(payload, a0, a1))

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
    du = dict(data_used or {})
    from services.adversarial.adversarial_preprocessor import check_comparison_safety, get_pipeline_query

    q_norm = get_pipeline_query(query or "", du)
    safety = check_comparison_safety(q_norm, du, compare_models=compare_models)
    if safety:
        return safety

    models = [m for m in (compare_models or _resolve_compare_models(q_norm)) if m]
    if len(models) < 2:
        return _insufficient_message(q_norm, models)

    ms = mission if mission is not None else build_mission_from_current_turn(q_norm)

    from services.consistency.consistency_injection_layer import prepare_comparison_consistency

    prepare_comparison_consistency(
        query=q_norm,
        compare_models=models,
        data_used=du,
    )
    locked = (du.get("comparison_v2") or {}).get("models")
    if isinstance(locked, list) and len(locked) >= 2:
        models = list(locked[:2])

    from services.comparison.comparison_pipeline_v2 import run_comparison_v2

    raw = run_comparison_v2(
        query=q_norm,
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

    answer = _format_structured_contrast(payload)
    if len(models) >= 2 and isinstance(data_used, dict):
        try:
            from services.temporal_market.temporal_market_intelligence import (
                format_comparison_temporal_overlay,
            )

            db = data_used.get("db")
            overlay = format_comparison_temporal_overlay(models[0], models[1], db=db)
            if overlay:
                answer = answer + "\n" + "\n".join(overlay)
        except Exception:
            pass
    return _guard_answer(answer)


__all__ = ["respond_aircraft_comparison"]
