#!/usr/bin/env python3
"""Read-only mission evaluation engine audit — benchmark missions × AIRCRAFT_PROFILES."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("DATA_AUTHORITY_STRICT", "0")

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from services.aircraft_feasibility.engine import evaluate_aircraft_feasibility
from services.aircraft_feasibility.hard_feasibility_engine import (
    VERDICT_CONDITIONAL_FIT,
    VERDICT_GOOD_FIT,
    VERDICT_NOT_A_FIT,
    assess_aircraft_hard_feasibility,
)
from services.aircraft_feasibility.mission_context import mission_context_from_profile
from services.aircraft_feasibility.payload_range import compute_payload_adjusted_range
from services.aircraft_feasibility.range_margin import compute_mission_range_requirement
from services.consultant.mission_state import MissionState
from services.mission.adapters import mission_state_to_profile
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES
from services.mission.models import MissionProfile, PriorityLevel, Route
from services.mission.route_distance_authority import resolve_route_distance


@dataclass
class BenchmarkMission:
    mission_id: str
    route: str
    passengers: int
    nbaa_reserves: bool = True
    westbound: bool = False
    winter: bool = False
    mountain: bool = False
    nonstop: bool = True


BENCHMARKS = [
    BenchmarkMission(
        "A",
        "San Francisco -> Tokyo",
        10,
        nbaa_reserves=True,
        westbound=True,
        winter=True,
        nonstop=True,
    ),
    BenchmarkMission(
        "B",
        "Los Angeles -> London",
        9,
        nbaa_reserves=True,
        westbound=True,
        nonstop=True,
    ),
    BenchmarkMission(
        "C",
        "Aspen -> London",
        8,
        mountain=True,
        nonstop=True,
    ),
    BenchmarkMission(
        "D",
        "Honolulu -> Sydney",
        10,
        nonstop=True,
    ),
    BenchmarkMission(
        "E",
        "Chicago -> Paris",
        14,
        nbaa_reserves=True,
        nonstop=True,
    ),
]


def _build_profile(bm: BenchmarkMission) -> MissionProfile:
    parts = bm.route.replace("→", "->").split("->", 1)
    origin, dest = parts[0].strip(), parts[1].strip()
    return MissionProfile(
        passengers=bm.passengers,
        routes=[Route(origin=origin, destination=dest)],
        nonstop_required=bm.nonstop,
        westbound_sensitive=bm.westbound,
        seasonal_note="winter_headwinds" if bm.winter else None,
        nbaa_reserve_required=bm.nbaa_reserves,
        reserves_requirement="NBAA IFR" if bm.nbaa_reserves else None,
        mountain_airports=bm.mountain,
        mountain_airport_priority=bm.mountain,
        international_ops=True,
    )


def _build_mission_state(bm: BenchmarkMission) -> MissionState:
    return MissionState(
        routes=[bm.route.replace("→", "->")],
        passenger_count=bm.passengers,
        westbound=bm.westbound,
        seasonal_constraints="winter" if bm.winter else "",
        reserves_requirement="NBAA IFR" if bm.nbaa_reserves else "",
        nonstop_requirement=bm.nonstop,
        mountain_airport_requirement=bm.mountain,
    )


def _fit_to_user_verdict(fit: str, feasible: bool) -> str:
    if not feasible or fit == VERDICT_NOT_A_FIT:
        return "NOT REALISTIC"
    if fit == VERDICT_CONDITIONAL_FIT:
        return "CONDITIONAL"
    return "FEASIBLE"


def _primary_reason(verdict, hard) -> str:
    if hard.verdict.rejection_reasons:
        return hard.verdict.rejection_reasons[0]
    if not hard.feasible:
        return "hard_feasibility: margin or corridor gate"
    if hard.fit_verdict == VERDICT_CONDITIONAL_FIT:
        return (
            f"hard_feasibility: dispatch margin {int(hard.verdict.margin_nm)} nm "
            f"(below 150 nm conditional threshold)"
        )
    return (
        f"hard_feasibility: available {int(hard.verdict.available_nm)} nm >= "
        f"required {int(hard.verdict.required_nm)} nm (margin {int(hard.verdict.margin_nm)} nm)"
    )


def audit_one(bm: BenchmarkMission, model: str) -> Dict[str, Any]:
    profile = _build_profile(bm)
    spec = AIRCRAFT_PROFILES[model]
    ctx = mission_context_from_profile(profile)
    route_res = resolve_route_distance(profile.routes[0].label())

    req = compute_mission_range_requirement(ctx)
    adj = compute_payload_adjusted_range(spec, ctx)
    hard = assess_aircraft_hard_feasibility(profile, model)
    verdict = evaluate_aircraft_feasibility(profile, model)

    # Capability layer (broker)
    cap_verdict = None
    cap_reasons: List[str] = []
    try:
        from services.consultant.named_aircraft_capability import (
            evaluate_named_aircraft_capability,
        )

        ms = _build_mission_state(bm)
        cap = evaluate_named_aircraft_capability(
            model, ms, mission_profile=profile, data_used={}
        )
        cap_verdict = cap.get("verdict")
        cap_reasons = list(cap.get("reasons") or [])
    except Exception as exc:
        cap_verdict = f"error:{exc}"

    user_verdict = _fit_to_user_verdict(hard.fit_verdict, hard.feasible)
    if cap_verdict == "MARGINAL" and user_verdict == "FEASIBLE":
        user_verdict_cap = "CONDITIONAL"
    elif cap_verdict == "NOT REALISTIC":
        user_verdict_cap = "NOT REALISTIC"
    else:
        user_verdict_cap = user_verdict

    return {
        "aircraft": model,
        "user_verdict": user_verdict,
        "capability_verdict": cap_verdict,
        "combined_verdict": user_verdict_cap,
        "hard_fit_verdict": hard.fit_verdict,
        "feasible": hard.feasible,
        "primary_calculation": _primary_reason(verdict, hard),
        "rejection_reasons": list(hard.verdict.rejection_reasons),
        "assumptions": {
            "stage_distance_nm": round(ctx.stage_distance_nm, 1),
            "route_source": route_res.source,
            "route_verified": route_res.is_verified,
            "nbaa_reserve_nm": round(req.nbaa_reserve_nm, 1),
            "westbound_penalty_nm": round(req.westbound_required_nm, 1),
            "payload_penalty_required_nm": round(req.payload_required_nm, 1),
            "mountain_penalty_required_nm": round(req.mountain_required_nm, 1),
            "dispatch_margin_nm": round(req.dispatch_margin_nm, 1),
            "total_required_nm": round(req.total_required_nm, 1),
            "practical_baseline_nm": round(adj.practical_baseline_nm, 1),
            "payload_penalty_available_nm": round(adj.payload_penalty_nm, 1),
            "winter_penalty_available_nm": round(adj.winter_penalty_nm, 1),
            "mountain_penalty_available_nm": round(adj.mountain_penalty_nm, 1),
            "available_nm": round(hard.verdict.available_nm, 1),
            "margin_nm": round(hard.verdict.margin_nm, 1),
            "runway_penalty_nm": round(hard.verdict.runway_penalty, 1),
            "flags": {
                "transpacific": ctx.transpacific,
                "transatlantic": ctx.transatlantic,
                "winter_westbound_transpacific": ctx.winter_westbound_transpacific,
                "westbound_sensitive": ctx.westbound_sensitive,
                "mountain_airports": ctx.mountain_airports,
            },
        },
        "capability_reasons": cap_reasons[:4],
    }


def _expected_feasible(bm_id: str, model: str) -> Optional[bool]:
    """Broker-realistic expectation for false-positive detection (None = no opinion)."""
    cat = AIRCRAFT_PROFILES[model].get("category", "")
    ulr = {"Global 7500", "Global 6500", "Gulfstream G650ER", "Gulfstream G650", "Falcon 8X"}
    if bm_id == "A":
        if model in ulr:
            return True
        if cat in ("super-midsize", "light", "turboprop"):
            return False
    if bm_id == "B":
        if model in ulr or model in {"Gulfstream G500", "Falcon 8X", "Global 5500"}:
            return True
        if cat in ("light", "turboprop", "super-midsize"):
            return False
    if bm_id == "C":
        if model in ulr or model in {"Challenger 650", "Falcon 7X", "Falcon 8X"}:
            return True
    if bm_id == "D":
        if model in ulr:
            return True
        if cat in ("super-midsize", "light"):
            return False
    if bm_id == "E":
        if model in {"Global 7500", "Gulfstream G650ER"}:
            return True
        if model in {"Challenger 650", "Citation Latitude", "Learjet 75"}:
            return False  # pax or range
    return None


def main() -> int:
    report: Dict[str, Any] = {
        "engine_constants": {
            "nbaa_reserve_nm": 200,
            "westbound_required_factor": 0.08,
            "winter_westbound_required_factor": 0.12,
            "mission_pax_8_plus_nm": 120,
            "mission_pax_10_plus_nm": 200,
            "mountain_available_penalty_nm": 300,
            "min_dispatch_margin_nm": 150,
            "min_dispatch_margin_ulr_nm": 40,
            "transpacific_min_practical_nm": 5200,
            "transpacific_winter_westbound_min_practical_nm": 5600,
        },
        "missions": [],
        "false_positive_rejections": [],
        "false_negative_acceptances": [],
    }

    models = sorted(AIRCRAFT_PROFILES.keys())

    for bm in BENCHMARKS:
        prof = _build_profile(bm)
        route_label = prof.routes[0].label()
        route_res = resolve_route_distance(route_label)
        mission_block: Dict[str, Any] = {
            "mission_id": bm.mission_id,
            "description": bm.route,
            "passengers": bm.passengers,
            "nbaa_reserves": bm.nbaa_reserves,
            "westbound": bm.westbound,
            "winter": bm.winter,
            "mountain": bm.mountain,
            "route_resolution": {
                "label": route_label,
                "distance_nm": route_res.distance_nm,
                "source": route_res.source,
                "verified": route_res.is_verified,
            },
            "aircraft_results": [],
        }
        for model in models:
            row = audit_one(bm, model)
            mission_block["aircraft_results"].append(row)
            exp = _expected_feasible(bm.mission_id, model)
            got_feasible = row["combined_verdict"] in ("FEASIBLE", "CONDITIONAL")
            if exp is True and row["combined_verdict"] == "NOT REALISTIC":
                report["false_positive_rejections"].append(
                    {
                        "mission": bm.mission_id,
                        "aircraft": model,
                        "engine_verdict": row["combined_verdict"],
                        "reason": row["primary_calculation"],
                        "expected": "FEASIBLE or CONDITIONAL",
                    }
                )
            elif exp is False and row["combined_verdict"] == "FEASIBLE":
                report["false_negative_acceptances"].append(
                    {
                        "mission": bm.mission_id,
                        "aircraft": model,
                        "engine_verdict": row["combined_verdict"],
                        "reason": row["primary_calculation"],
                        "expected": "NOT REALISTIC",
                    }
                )
        mission_block["summary"] = {
            "FEASIBLE": sum(
                1 for r in mission_block["aircraft_results"] if r["combined_verdict"] == "FEASIBLE"
            ),
            "CONDITIONAL": sum(
                1
                for r in mission_block["aircraft_results"]
                if r["combined_verdict"] == "CONDITIONAL"
            ),
            "NOT REALISTIC": sum(
                1
                for r in mission_block["aircraft_results"]
                if r["combined_verdict"] == "NOT REALISTIC"
            ),
        }
        report["missions"].append(mission_block)

    out = _ROOT / "evals" / "mission_engine_audit_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({m["mission_id"]: m["summary"] for m in report["missions"]}, indent=2))
    print(f"False positive rejections: {len(report['false_positive_rejections'])}")
    print(f"False negative acceptances: {len(report['false_negative_acceptances'])}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
