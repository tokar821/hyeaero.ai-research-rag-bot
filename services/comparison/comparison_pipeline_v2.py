"""
Comparison v2 pipeline — build, validate, render (explicit_comparison only).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from services.comparison.aircraft_registry_lock import RegistryLockResult, lock_comparison_aircraft
from services.comparison.comparison_renderer_v2 import (
    render_comparison_v2,
    render_insufficient_data,
)
from services.comparison.comparison_schema_v2 import (
    MATRIX_DIMENSIONS,
    AircraftEntryV2,
    ComparisonPayloadV2,
    insufficient_comparison,
)
from services.comparison.comparison_validator_v2 import validate_comparison_payload
from services.consultant.mission_state import MissionState
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES

_CATEGORY_MAP = {
    "light": "light",
    "super_mid": "super-midsize",
    "super-midsize": "super-midsize",
    "midsize": "super-midsize",
    "large": "large-cabin",
    "large_cabin": "large-cabin",
    "heavy": "large-cabin",
    "ultra_long": "ULR",
    "ultra-long": "ULR",
    "ulr": "ULR",
}


def _map_category(spec: Dict[str, Any]) -> str:
    raw = str(spec.get("category") or "").strip().lower().replace(" ", "_")
    return _CATEGORY_MAP.get(raw, "large-cabin")


def _cost_band(spec: Dict[str, Any]) -> str:
    oi = float(spec.get("operating_index") or 0.0)
    if oi <= 0.0:
        return "medium"
    if oi <= 0.55:
        return "low"
    if oi <= 0.72:
        return "medium"
    if oi <= 0.88:
        return "high"
    return "ultra"


def _winter_westbound(spec: Dict[str, Any], mission: MissionState) -> Any:
    cat = str(spec.get("category") or "").lower()
    if "ultra" in cat or "heavy" in cat:
        return True
    if "super_mid" in cat or "large" in cat:
        return "conditional"
    if mission.seasonal_constraints or mission.westbound:
        return "conditional"
    return False


def _mission_fit_score(model: str, spec: Dict[str, Any], mission: MissionState) -> float:
    try:
        from services.consultant.recommendation_engine import score_aircraft_for_mission

        rec = score_aircraft_for_mission(model, spec, mission)
        return max(0.0, min(1.0, float(getattr(rec, "total_score", 0.0) or 0.0)))
    except Exception:
        base = 0.55
        pax = int(mission.passenger_count or 0) or 0
        max_pax = int(spec.get("pax_max_long_range") or spec.get("pax_typical") or 0) or 0
        if pax and max_pax and pax <= max_pax:
            base += 0.1
        if mission.nonstop_requirement and float(spec.get("practical_nm") or 0) > 0:
            base += 0.12
        return max(0.0, min(1.0, base))


def _build_verdict(aircraft: List[AircraftEntryV2]) -> Dict[str, Optional[str]]:
    if not aircraft:
        return {
            "best_overall": None,
            "conditional_winner": None,
            "no_fit_reason": "no aircraft entries",
        }
    ranked = sorted(aircraft, key=lambda a: float(a["mission_fit_score"]), reverse=True)
    best = ranked[0]
    if float(best["mission_fit_score"]) < 0.6:
        return {
            "best_overall": None,
            "conditional_winner": None,
            "no_fit_reason": "no aircraft met minimum mission fit threshold",
        }
    conditional: Optional[str] = None
    if len(ranked) > 1 and float(ranked[1]["mission_fit_score"]) >= 0.6:
        conditional = str(ranked[1]["name"])
    return {
        "best_overall": str(best["name"]),
        "conditional_winner": conditional,
        "no_fit_reason": None,
    }


def _build_aircraft_entries(
    canonical_names: Sequence[str],
    mission: MissionState,
) -> Optional[List[AircraftEntryV2]]:
    from services.data_authority.aircraft_spec_repository import (
        INSUFFICIENT_VERIFIED_COMPARISON,
        require_verified_specs,
    )

    verified, missing = require_verified_specs(canonical_names)
    if missing:
        return None
    entries: List[AircraftEntryV2] = []
    for v in verified:
        name = v.canonical_name
        spec = v.to_profile_dict()
        if not spec:
            return None
        range_nm = spec.get("practical_nm") or spec.get("range_nm")
        seats = spec.get("pax_typical") or spec.get("max_pax")
        entry: AircraftEntryV2 = {
            "name": name,
            "category": _map_category(spec),
            "range_nm": float(range_nm) if range_nm is not None else None,
            "seats": int(seats) if seats is not None else None,
            "mission_fit_score": round(_mission_fit_score(name, spec, mission), 4),
            "cost_band": _cost_band(spec),  # type: ignore[typeddict-item]
            "winter_westbound_capability": _winter_westbound(spec, mission),  # type: ignore[typeddict-item]
        }
        entries.append(entry)
    return entries


def _requires_full_registry_set(
    models: Sequence[str],
    lock: RegistryLockResult,
    query: str = "",
) -> bool:
    """3+ named compare tokens with any rejection → no partial tables (edge compares only)."""
    ql = (query or "").lower()
    if "fleet strategy" in ql or "aircraft strategy" in ql:
        return False
    raw_tokens = [str(m or "").strip() for m in models if str(m or "").strip()]
    if len(raw_tokens) < 3:
        return False
    return len(lock.rejected) > 0


def build_comparison_payload_v2(
    models: Sequence[str],
    mission: MissionState,
    *,
    query: str = "",
) -> ComparisonPayloadV2 | Dict[str, Any]:
    lock = lock_comparison_aircraft(models)
    if _requires_full_registry_set(models, lock, query):
        return insufficient_comparison(
            "incomplete canonical aircraft set; rejected="
            + ",".join(lock.rejected[:6])
        )
    if len(lock.canonical) < 2:
        return insufficient_comparison(
            "missing canonical aircraft set"
            if not lock.rejected
            else f"missing canonical aircraft set; rejected={','.join(lock.rejected[:5])}"
        )

    entries = _build_aircraft_entries(lock.canonical, mission)
    if entries is None or len(entries) < 2:
        from services.data_authority.aircraft_spec_repository import (
            INSUFFICIENT_VERIFIED_COMPARISON,
        )

        return insufficient_comparison(INSUFFICIENT_VERIFIED_COMPARISON)

    verdict = _build_verdict(entries)
    payload: ComparisonPayloadV2 = {
        "mode": "explicit_comparison",
        "aircraft": entries,
        "comparison_matrix": {"dimensions": list(MATRIX_DIMENSIONS)},
        "verdict": verdict,  # type: ignore[typeddict-item]
        "data_quality": {"status": "OK", "reason": "catalog_verified"},
    }
    return payload


def run_comparison_v2(
    *,
    query: str,
    mission: MissionState,
    compare_models: Sequence[str],
    data_used: Optional[Dict[str, Any]] = None,
    mode: str = "explicit_comparison",
) -> str:
    """
    Single entry for explicit_comparison rendering.

    Returns JSON string only (full schema or INSUFFICIENT_DATA).
    """
    if mode != "explicit_comparison":
        return render_insufficient_data("mode is not explicit_comparison")

    payload = build_comparison_payload_v2(compare_models, mission, query=query or "")

    if payload.get("status") == "INSUFFICIENT_DATA":
        out = render_insufficient_data(str(payload.get("reason") or ""))
        if isinstance(data_used, dict):
            data_used["comparison_v2"] = {"status": "INSUFFICIENT_DATA"}
            data_used["comparison_structured_engine"] = {"type": "comparison_v2_insufficient"}
            data_used["broker_narrative_authoritative"] = True
        return out

    vr = validate_comparison_payload(payload)
    if not vr.ok:
        out = render_insufficient_data(vr.reason or "validation failed")
        if isinstance(data_used, dict):
            data_used["comparison_v2"] = {"status": "INSUFFICIENT_DATA", "reason": vr.reason}
            data_used["comparison_structured_engine"] = {"type": "comparison_v2_insufficient"}
            data_used["broker_narrative_authoritative"] = True
        return out

    out = render_comparison_v2(payload, mode=mode)
    if isinstance(data_used, dict):
        data_used["comparison_v2"] = {
            "status": "OK",
            "models": [a["name"] for a in payload.get("aircraft", [])],
        }
        data_used["comparison_structured_engine"] = {
            "type": "comparison_v2_json",
            "models": [a["name"] for a in payload.get("aircraft", [])],
        }
        data_used["broker_narrative_authoritative"] = True
    return out


__all__ = ["build_comparison_payload_v2", "run_comparison_v2"]
