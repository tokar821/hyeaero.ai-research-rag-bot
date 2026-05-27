"""
Adversarial mission eval suite — operational credibility over benchmark pass rate.

Randomized and hand-crafted cases probe elimination, geodesic policy, airport constraints,
and formatter invariants.
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

os.environ.setdefault("CONSULTANT_ORCHESTRATION", "1")

_RANDOM_SEED = 42

_EUROPE_US = (
    ("Geneva", "London"),
    ("Paris", "New York"),
    ("Zurich", "Miami"),
    ("Frankfurt", "Boston"),
)
_MOUNTAIN = (
    ("Dallas", "Aspen"),
    ("Aspen", "Telluride"),
    ("Denver", "Aspen"),
)
_CARIBBEAN = (
    ("Miami", "Nassau"),
    ("Miami", "St Maarten"),
)


@dataclass
class AdversarialCase:
    case_id: str
    query: str
    check: Callable[[Dict[str, Any]], Optional[str]]


@dataclass
class AdversarialResult:
    passed: int = 0
    failed: int = 0
    failures: List[str] = field(default_factory=list)


def _run(query: str) -> Dict[str, Any]:
    from services.elimination.elimination_invariant import (
        assert_elimination_invariant,
        collect_eliminated_models,
    )
    from services.orchestration.pipeline_orchestrator import run_deterministic_stages

    data_used: Dict[str, Any] = {}
    pipeline, _ = run_deterministic_stages(query, data_used=data_used)
    recs = [r for r in (pipeline.recommendations or []) if not r.avoid]
    eliminated = collect_eliminated_models(
        data_used=data_used,
        elimination_log=pipeline.elimination_log,
        feasibility_map=pipeline.feasibility_map,
        explicit_eliminated=pipeline.eliminated_models,
    )
    presented = [r.model for r in recs]
    try:
        assert_elimination_invariant(presented, eliminated)
        invariant_ok = True
    except AssertionError as exc:
        invariant_ok = False
        invariant_err = str(exc)
    else:
        invariant_err = ""

    fleet_plan = data_used.get("fleet_composition_plan") or {}
    packet = data_used.get("hye_reasoning_packet") or {}
    return {
        "pipeline": pipeline,
        "data_used": data_used,
        "rec_models": presented,
        "eliminated": eliminated,
        "invariant_ok": invariant_ok,
        "invariant_err": invariant_err,
        "fleet_plan": fleet_plan,
        "reasoning_packet": packet,
    }


def _check_invariant(ctx: Dict[str, Any]) -> Optional[str]:
    if not ctx["invariant_ok"]:
        return ctx["invariant_err"] or "elimination invariant violated"
    return None


def _check_no_super_mid_transatlantic(ctx: Dict[str, Any]) -> Optional[str]:
    inv = _check_invariant(ctx)
    if inv:
        return inv
    sm = ("challenger 350", "g280", "praetor", "longitude", "latitude")
    blob = " ".join(m.lower() for m in ctx["rec_models"])
    if any(s in blob for s in sm):
        return f"super-mid in shortlist: {ctx['rec_models']}"
    return None


def _check_mountain_no_heavy(ctx: Dict[str, Any]) -> Optional[str]:
    inv = _check_invariant(ctx)
    if inv:
        return inv
    heavy = ("g650", "global 7500", "challenger 350", "g280")
    blob = " ".join(m.lower() for m in ctx["rec_models"])
    if ctx["rec_models"] and any(h in blob for h in heavy):
        return f"heavy on mountain: {ctx['rec_models']}"
    return None


def _check_geodesic_no_primary(ctx: Dict[str, Any]) -> Optional[str]:
    inv = _check_invariant(ctx)
    if inv:
        return inv
    auth = ctx["data_used"].get("route_distance_authority") or []
    if not any(r.get("source") == "geodesic" for r in auth):
        return None
    from services.broker.broker_verdicts import BrokerVerdict

    primary = BrokerVerdict.PRIMARY_RECOMMENDATION.value
    for r in ctx["pipeline"].recommendations or []:
        if (r.fit_verdict or "") == primary:
            return f"PRIMARY on geodesic-only route: {r.model}"
    return None


def _check_unresolved_blocks_rank(ctx: Dict[str, Any]) -> Optional[str]:
    if ctx["data_used"].get("route_blocks_ranking") and ctx["rec_models"]:
        return f"ranked on unresolved route: {ctx['rec_models']}"
    return _check_invariant(ctx)


def _build_random_cases(rng: random.Random, n: int = 12) -> List[AdversarialCase]:
    cases: List[AdversarialCase] = []
    for i in range(n):
        kind = rng.choice(("europe_us", "mountain", "caribbean", "impossible"))
        pax = rng.randint(4, 12)
        if kind == "europe_us":
            o, d = rng.choice(_EUROPE_US)
            q = f"{pax} passengers {o} to {d} nonstop winter recommend"
            cases.append(
                AdversarialCase(
                    f"rand_eu_{i}",
                    q,
                    _check_no_super_mid_transatlantic,
                )
            )
        elif kind == "mountain":
            o, d = rng.choice(_MOUNTAIN)
            q = f"{pax} pax {o} to {d} hot and high recommend"
            cases.append(
                AdversarialCase(
                    f"rand_mtn_{i}",
                    q,
                    _check_mountain_no_heavy,
                )
            )
        elif kind == "caribbean":
            o, d = rng.choice(_CARIBBEAN)
            q = f"{pax} passengers {o} to {d} short runway recommend"
            cases.append(
                AdversarialCase(
                    f"rand_car_{i}",
                    q,
                    lambda ctx: (
                        _check_invariant(ctx)
                        or (
                            "ULR on caribbean short"
                            if any("global 7500" in m.lower() for m in ctx["rec_models"][:1])
                            else None
                        )
                    ),
                )
            )
        else:
            o, d = rng.choice(_EUROPE_US)
            q = f"{pax} passengers {o} to {d} nonstop light jet recommend"
            cases.append(
                AdversarialCase(
                    f"rand_impossible_{i}",
                    q,
                    lambda ctx: (
                        _check_invariant(ctx)
                        or (
                            "light jet recommended on impossible leg"
                            if any(
                                k in " ".join(ctx["rec_models"]).lower()
                                for k in ("cj2", "cj3", "phenom 300")
                            )
                            else None
                        )
                    ),
                )
            )
    return cases


def _check_fleet_multi_domain(ctx: Dict[str, Any]) -> Optional[str]:
    inv = _check_invariant(ctx)
    if inv:
        return inv
    fp = ctx.get("fleet_plan") or {}
    if not fp.get("multi_aircraft_required") and not fp.get("multi_domain_required"):
        return "expected multi-domain fleet decomposition"
    if not fp.get("single_aircraft_structurally_invalid"):
        return "expected single_aircraft_structurally_invalid"
    if fp.get("trigger") != "elimination_failure":
        return f"expected elimination_failure trigger, got {fp.get('trigger')}"
    try:
        from services.fleet.fleet_invariant import assert_fleet_invariants
        from services.fleet.fleet_composition import FleetCompositionPlan, FleetRoleAssignment, MissionSegmentRole

        assignments = []
        for a in fp.get("assignments") or []:
            if not isinstance(a, dict):
                continue
            assignments.append(
                FleetRoleAssignment(
                    role=MissionSegmentRole(a.get("role", "ulr_international")),
                    segment_label=str(a.get("segment_label") or ""),
                    primary_model=str(a.get("primary_model") or ""),
                    domain_feasible=bool(a.get("domain_feasible", True)),
                )
            )
        plan = FleetCompositionPlan(
            multi_aircraft_required=True,
            assignments=assignments,
            domain_traces=list(fp.get("domain_traces") or []),
        )
        assert_fleet_invariants(plan)
    except Exception as exc:
        return f"fleet invariant: {exc}"
    return None


def _check_one_aircraft_only_impossible(ctx: Dict[str, Any]) -> Optional[str]:
    inv = _check_invariant(ctx)
    if inv:
        return inv
    fp = ctx.get("fleet_plan") or {}
    if not fp.get("multi_aircraft_required"):
        return "one-aircraft-only on impossible set should still decompose"
    if fp.get("trigger") != "impossible_single_aircraft_constraint":
        return f"expected impossible_single_aircraft_constraint, got {fp.get('trigger')}"
    return None


def _check_packet_fleet_audit_trace(ctx: Dict[str, Any]) -> Optional[str]:
    """P3 — immutable packet must carry per-domain audit segments."""
    fp = ctx.get("fleet_plan") or {}
    if not fp.get("multi_aircraft_required") and not fp.get("multi_domain_required"):
        return None
    packet = ctx.get("reasoning_packet") or {}
    if not packet.get("immutable"):
        return "missing hye_reasoning_packet"
    audit = packet.get("fleet_audit") or {}
    if not audit.get("segments"):
        return "packet missing fleet_audit.segments"
    try:
        from services.telemetry.reasoning_packet_enforcement import validate_packet_fleet_audit

        issues = validate_packet_fleet_audit(packet)
        if issues:
            return f"packet fleet audit: {issues[0]}"
    except Exception as exc:
        return f"packet audit validation failed: {exc}"
    for seg in audit.get("segments") or []:
        if not isinstance(seg, dict):
            continue
        if not seg.get("constraint_triggers"):
            return f"segment {seg.get('domain')} missing constraint_triggers"
    return None


def _check_fleet_seasonal_split(ctx: Dict[str, Any]) -> Optional[str]:
    inv = _check_invariant(ctx)
    if inv:
        return inv
    fp = ctx.get("fleet_plan") or {}
    if not fp.get("multi_aircraft_required"):
        return "expected seasonal multi-domain fleet"
    audit = (ctx.get("reasoning_packet") or {}).get("fleet_audit") or {}
    if len(audit.get("segments") or []) < 2:
        return "expected >=2 domain segments in fleet_audit"
    return _check_packet_fleet_audit_trace(ctx)


def _check_fleet_fractional_charter(ctx: Dict[str, Any]) -> Optional[str]:
    inv = _check_invariant(ctx)
    if inv:
        return inv
    fp = ctx.get("fleet_plan") or {}
    if not fp.get("multi_aircraft_required"):
        return "expected fractional/charter hybrid decomposition"
    packet = ctx.get("reasoning_packet") or {}
    if not packet.get("fleet_audit", {}).get("segments"):
        return "missing fleet audit for hybrid mission"
    return _check_packet_fleet_audit_trace(ctx)


def _check_fleet_no_eliminated_in_roles(ctx: Dict[str, Any]) -> Optional[str]:
    inv = _check_invariant(ctx)
    if inv:
        return inv
    fp = ctx.get("fleet_plan") or {}
    if not fp:
        return None
    traces = {str(t.get("domain") or ""): t for t in fp.get("domain_traces") or [] if isinstance(t, dict)}
    for a in fp.get("assignments") or []:
        if not isinstance(a, dict):
            continue
        primary = (a.get("primary_model") or "").strip()
        if not primary:
            continue
        role = str(a.get("role") or "")
        tr = traces.get(role) or {}
        domain_elim = {str(m).lower() for m in tr.get("eliminated_models") or []}
        if primary.lower() in domain_elim:
            return f"domain-eliminated aircraft in fleet role: {primary}"
    return None


HAND_CRAFTED: List[AdversarialCase] = [
    AdversarialCase(
        "fleet_teb_lon_aspen",
        "8 passengers TEB → London nonstop winter westbound and KASE → KTEX hot and high — recommend",
        lambda ctx: _check_fleet_multi_domain(ctx) or _check_packet_fleet_audit_trace(ctx),
    ),
    AdversarialCase(
        "fleet_one_aircraft_only",
        "one aircraft only: 6 pax TEB London nonstop and KASE Telluride hot and high",
        _check_one_aircraft_only_impossible,
    ),
    AdversarialCase(
        "fleet_miami_caribbean_mountain",
        "10 passengers Miami Caribbean hops and Aspen winter — recommend",
        _check_fleet_no_eliminated_in_roles,
    ),
    AdversarialCase(
        "fleet_seasonal_split",
        "summer: TEB to London nonstop 8 pax; winter: KASE to KTEX hot and high — one operational program",
        _check_fleet_seasonal_split,
    ),
    AdversarialCase(
        "fleet_fractional_charter_hybrid",
        "fractional card for transatlantic; ad hoc charter for Caribbean hops and Aspen — recommend fleet",
        _check_fleet_fractional_charter,
    ),
    AdversarialCase(
        "payload_contradiction",
        "12 passengers Aspen to Telluride nonstop with maximum baggage recommend",
        _check_invariant,
    ),
    AdversarialCase(
        "geodesic_geneva_nyc",
        "6 passengers Geneva to New York nonstop recommend",
        _check_geodesic_no_primary,
    ),
    AdversarialCase(
        "fictional_route",
        "8 passengers Fictional City Alpha to Beta nonstop recommend",
        _check_unresolved_blocks_rank,
    ),
]


def run_adversarial_suite(
    *,
    random_cases: int = 12,
    verbose: bool = True,
) -> AdversarialResult:
    rng = random.Random(_RANDOM_SEED)
    cases = list(HAND_CRAFTED) + _build_random_cases(rng, random_cases)
    out = AdversarialResult()
    for case in cases:
        try:
            ctx = _run(case.query)
            err = case.check(ctx)
        except Exception as exc:
            err = f"exception: {exc}"
        if err:
            out.failed += 1
            msg = f"[FAIL] {case.case_id}: {err}"
            out.failures.append(msg)
            if verbose:
                print(msg)
        else:
            out.passed += 1
            if verbose:
                print(f"[PASS] {case.case_id}")
    if verbose:
        print(f"\nAdversarial suite: {out.passed} passed, {out.failed} failed")
    return out


if __name__ == "__main__":
    import sys

    sys.exit(1 if run_adversarial_suite().failed else 0)
