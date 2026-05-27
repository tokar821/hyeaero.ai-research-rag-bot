"""
Fleet-level invariants — no fleet member may violate elimination rules independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from services.elimination.elimination_invariant import _norm
from services.fleet.fleet_composition import FleetCompositionPlan, FleetRoleAssignment

logger = logging.getLogger(__name__)


@dataclass
class FleetInvariantReport:
    ok: bool
    violations: List[str] = field(default_factory=list)
    stripped_models: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "violations": list(self.violations),
            "stripped_models": list(self.stripped_models),
        }


def _segment_eliminated_for_assignment(
    assignment: FleetRoleAssignment,
    global_eliminated: Set[str],
) -> Optional[str]:
    if not assignment.primary_model:
        return None
    model_key = _norm(assignment.primary_model)
    if model_key in global_eliminated:
        return f"{assignment.primary_model} globally eliminated but assigned to {assignment.segment_label}"
    for alt in assignment.alternates:
        if _norm(alt) in global_eliminated:
            return f"alternate {alt} globally eliminated in role {assignment.role.value}"
    return None


def enforce_fleet_elimination_invariant(
    plan: FleetCompositionPlan,
    *,
    data_used: Optional[Dict[str, Any]] = None,
    elimination_log: Optional[Sequence[Dict[str, Any]]] = None,
    feasibility_map: Optional[Dict[str, Any]] = None,
    explicit_eliminated: Optional[Sequence[str]] = None,
) -> FleetCompositionPlan:
    """
    Strip fleet assignments that violate global or per-domain elimination lineage.
    """
    if not plan.multi_domain_required:
        return plan

    # Hard mission-wide blocks only — not feasibility_map range failures on other legs.
    trace_feasible_by_domain: Dict[str, Set[str]] = {}
    trace_eliminated_by_domain: Dict[str, Set[str]] = {}
    for trace in plan.domain_traces:
        if isinstance(trace, dict):
            dom = str(trace.get("domain") or "")
            trace_feasible_by_domain[dom] = {_norm(m) for m in trace.get("feasible_models") or []}
            trace_eliminated_by_domain[dom] = {_norm(m) for m in trace.get("eliminated_models") or []}

    violations: List[str] = []
    cleaned: List[FleetRoleAssignment] = []
    for a in plan.assignments:
        dom_key = a.role.value
        dom_feasible = trace_feasible_by_domain.get(dom_key, set())
        domain_elim = trace_eliminated_by_domain.get(dom_key, set())
        issue = None
        if a.primary_model and domain_elim and _norm(a.primary_model) in domain_elim:
            issue = f"{a.primary_model} eliminated for {dom_key} domain"
        elif a.primary_model and dom_feasible and _norm(a.primary_model) not in dom_feasible:
            issue = f"{a.primary_model} not in domain feasible set for {dom_key}"
        if issue:
            violations.append(issue)
            logger.error("FLEET_INVARIANT_VIOLATION: %s", issue)
            cleaned.append(
                FleetRoleAssignment(
                    role=a.role,
                    segment_label=a.segment_label,
                    primary_model="",
                    fit_verdict=a.fit_verdict,
                    rationale="Assignment removed — model failed elimination for this domain.",
                    alternates=[
                        alt
                        for alt in a.alternates
                        if _norm(alt) in dom_feasible and _norm(alt) not in domain_elim
                    ],
                    eliminated_from_role=list(a.eliminated_from_role),
                    domain_feasible=False,
                )
            )
        else:
            cleaned.append(a)

    plan.assignments = cleaned
    plan.invariant_report = FleetInvariantReport(
        ok=len(violations) == 0,
        violations=violations,
        stripped_models=[a.segment_label for a in cleaned if not a.primary_model and violations],
    )
    return plan


def assert_fleet_invariants(
    plan: FleetCompositionPlan,
    *,
    global_eliminated: Optional[Set[str]] = None,
) -> None:
    """Raise on fleet invariant breach — for tests and adversarial evals."""
    hard_eliminated = global_eliminated or set()

    for a in plan.assignments:
        if not a.primary_model:
            continue
        issue = _segment_eliminated_for_assignment(a, hard_eliminated)
        if issue:
            raise AssertionError(issue)

    trace_feasible_by_domain: Dict[str, Set[str]] = {}
    trace_eliminated_by_domain: Dict[str, Set[str]] = {}
    for trace in plan.domain_traces:
        if isinstance(trace, dict):
            dom = str(trace.get("domain") or "")
            trace_feasible_by_domain[dom] = {_norm(m) for m in trace.get("feasible_models") or []}
            trace_eliminated_by_domain[dom] = {_norm(m) for m in trace.get("eliminated_models") or []}

    for a in plan.assignments:
        if not a.primary_model:
            continue
        dom_key = a.role.value
        feasible = trace_feasible_by_domain.get(dom_key)
        if feasible is not None and _norm(a.primary_model) not in feasible:
            raise AssertionError(
                f"{a.primary_model} assigned to {dom_key} but not in domain feasible set"
            )
        domain_elim = trace_eliminated_by_domain.get(dom_key, set())
        if _norm(a.primary_model) in domain_elim:
            raise AssertionError(
                f"{a.primary_model} eliminated for {dom_key} but assigned to fleet role"
            )
