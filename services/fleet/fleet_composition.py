"""
P2 — Multi-domain operational fleet composition.

Some missions are multi-domain operational problems, not aircraft selection problems.
Segmentation triggers on elimination failure (no universal survivor), not preference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.fleet.fleet_domain_analysis import (
    OperationalDomain,
    SegmentationTrigger,
    analyze_multi_domain_operational_problem,
    identify_operational_domains,
)
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES
from services.mission.models import MissionProfile

# Legacy segment roles — map 1:1 to operational domains
class MissionSegmentRole(str, Enum):
    ULR_INTERNATIONAL = OperationalDomain.ULR_CLASS.value
    MOUNTAIN_FIELD = OperationalDomain.SHORT_FIELD_HIGH_PERFORMANCE.value
    REGIONAL_UTILITY = "regional_utility"
    COAST_TO_COAST = "coast_to_coast"
    CARIBBEAN_SHUTTLE = OperationalDomain.REGIONAL_STOL_FLEX.value


_ROLE_TO_DOMAIN = {
    MissionSegmentRole.ULR_INTERNATIONAL: OperationalDomain.ULR_CLASS,
    MissionSegmentRole.MOUNTAIN_FIELD: OperationalDomain.SHORT_FIELD_HIGH_PERFORMANCE,
    MissionSegmentRole.CARIBBEAN_SHUTTLE: OperationalDomain.REGIONAL_STOL_FLEX,
}

_DOMAIN_TO_ROLE = {v: k for k, v in _ROLE_TO_DOMAIN.items()}


@dataclass
class MissionSegment:
    role: MissionSegmentRole
    label: str
    stage_nm: float = 0.0
    required_nm: float = 0.0
    route_labels: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "label": self.label,
            "stage_nm": round(self.stage_nm, 1),
            "required_nm": round(self.required_nm, 1),
            "route_labels": list(self.route_labels),
            "notes": list(self.notes),
        }


@dataclass
class FleetRoleAssignment:
    role: MissionSegmentRole
    segment_label: str
    primary_model: str
    fit_verdict: str = ""
    rationale: str = ""
    alternates: List[str] = field(default_factory=list)
    eliminated_from_role: List[str] = field(default_factory=list)
    domain_feasible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "segment_label": self.segment_label,
            "primary_model": self.primary_model,
            "fit_verdict": self.fit_verdict,
            "rationale": self.rationale,
            "alternates": list(self.alternates),
            "eliminated_from_role": list(self.eliminated_from_role),
            "domain_feasible": self.domain_feasible,
        }


@dataclass
class FleetCompositionPlan:
    multi_aircraft_required: bool
    segments: List[MissionSegment] = field(default_factory=list)
    assignments: List[FleetRoleAssignment] = field(default_factory=list)
    ownership_note: str = ""
    doctrine: str = ""
    trigger: str = SegmentationTrigger.NONE.value
    single_aircraft_structurally_invalid: bool = False
    universal_survivors: List[str] = field(default_factory=list)
    domain_traces: List[Dict[str, Any]] = field(default_factory=list)
    invariant_report: Optional[Any] = None

    @property
    def multi_domain_required(self) -> bool:
        return self.multi_aircraft_required

    def presented_models(self) -> List[str]:
        return [a.primary_model for a in self.assignments if a.primary_model]

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "multi_aircraft_required": self.multi_aircraft_required,
            "multi_domain_required": self.multi_aircraft_required,
            "segments": [s.to_dict() for s in self.segments],
            "assignments": [a.to_dict() for a in self.assignments],
            "ownership_note": self.ownership_note,
            "doctrine": self.doctrine,
            "presented_models": self.presented_models(),
            "trigger": self.trigger,
            "single_aircraft_structurally_invalid": self.single_aircraft_structurally_invalid,
            "universal_survivors": list(self.universal_survivors),
            "domain_traces": list(self.domain_traces),
        }
        if self.invariant_report is not None:
            out["fleet_invariant"] = (
                self.invariant_report.to_dict()
                if hasattr(self.invariant_report, "to_dict")
                else self.invariant_report
            )
        return out


def detect_multi_aircraft_mission(
    profile: MissionProfile,
    mission: MissionState,
    *,
    query: str = "",
    pool: Optional[Sequence[str]] = None,
    data_used: Optional[Dict[str, Any]] = None,
    global_eliminated: Optional[Set[str]] = None,
) -> bool:
    """
    True only when multi-domain decomposition is required (elimination failure), not preference.
    """
    candidates = list(pool or AIRCRAFT_PROFILES.keys())
    analysis = analyze_multi_domain_operational_problem(
        profile,
        mission,
        candidates,
        query=query,
        data_used=data_used,
        global_eliminated=global_eliminated,
    )
    return analysis.multi_domain_required


def _domain_specs_to_segments(analysis_domains) -> List[MissionSegment]:
    segments: List[MissionSegment] = []
    for dom in analysis_domains:
        role = _DOMAIN_TO_ROLE.get(dom.domain, MissionSegmentRole.COAST_TO_COAST)
        segments.append(
            MissionSegment(
                role=role,
                label=dom.label,
                stage_nm=dom.stage_nm,
                required_nm=dom.required_nm,
                route_labels=list(dom.route_labels),
                notes=[f"Constraint triggers: {', '.join(dom.constraint_triggers)}"],
            )
        )
    return segments


def _pick_for_domain(
    segment: MissionSegment,
    domain_feasible: Sequence[str],
    recommendations_by_model: Dict[str, AircraftRecommendation],
    domain_trace: Dict[str, Any],
) -> FleetRoleAssignment:
    eliminated_in_domain = list(domain_trace.get("eliminated_models") or [])
    lineage = domain_trace.get("elimination_lineage") or []
    candidates = [m for m in domain_feasible if m in AIRCRAFT_PROFILES]
    if not candidates:
        return FleetRoleAssignment(
            role=segment.role,
            segment_label=segment.label,
            primary_model="",
            domain_feasible=False,
            rationale="No aircraft survives this domain's corridor, airport, and payload gates.",
            eliminated_from_role=eliminated_in_domain[:6],
        )

    primary = candidates[0]
    if len(candidates) > 1:
        scored = []
        for m in candidates:
            spec = AIRCRAFT_PROFILES.get(m) or {}
            sc = float(spec.get("dispatch_score") or 0.7)
            if segment.role == MissionSegmentRole.MOUNTAIN_FIELD:
                sc += float(spec.get("short_field_score") or 0) * 0.5
            elif segment.role == MissionSegmentRole.ULR_INTERNATIONAL:
                sc += float(spec.get("practical_nm") or 0) / 10000.0
            scored.append((sc, m))
        scored.sort(reverse=True)
        primary = scored[0][1]

    rec = recommendations_by_model.get(primary)
    verdict = (rec.fit_verdict if rec else "") or "VIABLE WITH COMPROMISES"
    alts = [m for m in candidates if m != primary][:2]

    corridor = domain_trace.get("corridor_classification") or ""
    rationale = (
        f"Independently feasible within {segment.role.value} domain"
        + (f" (corridor: {corridor})" if corridor else "")
        + " — not a global ranking preference."
    )

    return FleetRoleAssignment(
        role=segment.role,
        segment_label=segment.label,
        primary_model=primary,
        fit_verdict=verdict,
        rationale=rationale,
        alternates=alts,
        eliminated_from_role=eliminated_in_domain[:4],
        domain_feasible=True,
    )


def _ownership_crossover_note(profile: MissionProfile) -> str:
    posture = profile.ownership_posture or profile.ownership_interest
    if not posture:
        return ""
    val = posture.value if hasattr(posture, "value") else str(posture)
    freq = (profile.mission_frequency or "").lower()
    if val == "fractional":
        return (
            "Fractional may cover ULR legs; mountain and Caribbean domains often need "
            "ad hoc charter or a dedicated STOL-capable card."
        )
    if val == "charter":
        return (
            "Charter-by-leg fits multi-domain missions; expect different operators per domain."
        )
    if val == "full_ownership" and ("high" in freq or "200" in freq):
        return (
            "High-utilization owners often pair ULR flagship with a field-performance aircraft."
        )
    return ""


def build_fleet_composition_plan(
    profile: MissionProfile,
    mission: MissionState,
    recommendations: Sequence[AircraftRecommendation],
    *,
    query: str = "",
    feasible_models: Optional[Sequence[str]] = None,
    data_used: Optional[Dict[str, Any]] = None,
    elimination_log: Optional[Sequence[Dict[str, Any]]] = None,
    feasibility_map: Optional[Dict[str, Any]] = None,
    explicit_eliminated: Optional[Sequence[str]] = None,
) -> FleetCompositionPlan:
    """
    Build per-domain assignments when multi-domain operational decomposition is required.
    """
    pool = list(feasible_models or []) or [
        r.model for r in recommendations if not r.avoid
    ] or list(AIRCRAFT_PROFILES.keys())

    try:
        from services.aircraft_truth import filter_truth_verified_models

        pool = filter_truth_verified_models(pool)
    except Exception:
        pass

    analysis = analyze_multi_domain_operational_problem(
        profile,
        mission,
        pool,
        query=query,
        data_used=data_used,
        global_eliminated=None,
    )

    if not analysis.multi_domain_required:
        return FleetCompositionPlan(
            multi_aircraft_required=False,
            doctrine=analysis.doctrine,
            universal_survivors=list(analysis.universal_survivors),
            domain_traces=[t.to_dict() for t in analysis.domain_traces],
            trigger=analysis.trigger.value,
        )

    segments = _domain_specs_to_segments(analysis.domains)
    rec_by_model = {r.model: r for r in recommendations if not r.avoid}
    trace_by_domain = {t.domain: t.to_dict() for t in analysis.domain_traces}

    assignments: List[FleetRoleAssignment] = []
    used: Set[str] = set()
    for seg in segments:
        op_domain = _ROLE_TO_DOMAIN.get(seg.role)
        if op_domain is None:
            continue
        tr = trace_by_domain.get(op_domain, {})
        feasible = [m for m in tr.get("feasible_models") or [] if m not in used]
        if not feasible:
            feasible = list(tr.get("feasible_models") or [])
        assignment = _pick_for_domain(seg, feasible, rec_by_model, tr)
        if assignment.primary_model:
            used.add(assignment.primary_model)
        assignments.append(assignment)

    plan = FleetCompositionPlan(
        multi_aircraft_required=True,
        segments=segments,
        assignments=assignments,
        ownership_note=_ownership_crossover_note(profile),
        doctrine=analysis.doctrine,
        trigger=analysis.trigger.value,
        single_aircraft_structurally_invalid=analysis.single_aircraft_structurally_invalid,
        universal_survivors=list(analysis.universal_survivors),
        domain_traces=[t.to_dict() for t in analysis.domain_traces],
    )

    from services.fleet.fleet_invariant import enforce_fleet_elimination_invariant

    plan = enforce_fleet_elimination_invariant(
        plan,
        data_used=data_used,
        elimination_log=elimination_log,
        feasibility_map=feasibility_map,
        explicit_eliminated=explicit_eliminated,
    )
    return plan


def format_fleet_composition_block(plan: FleetCompositionPlan) -> str:
    """Deterministic multi-domain advisory prose."""
    if not plan.multi_aircraft_required or not plan.assignments:
        if plan.single_aircraft_structurally_invalid:
            return (
                "Operational verdict: one aircraft cannot structurally cover every domain "
                "in this mission set — this is not a 'less optimal' single-jet pick, it is invalid. "
                + (plan.doctrine or "")
            ).strip()
        return ""

    lines = [
        "Multi-domain operational composition (each domain independently feasible):",
        "",
    ]
    if plan.single_aircraft_structurally_invalid:
        lines.append(
            "Verdict: a single platform spanning all domains is structurally invalid — "
            "decomposition was triggered by elimination failure across domains, not preference."
        )
        lines.append("")

    for i, a in enumerate(plan.assignments, 1):
        if not a.primary_model:
            lines.append(
                f"{i}. {a.segment_label} — no survivor after corridor/airport/payload gates."
            )
            continue
        alt = f" Alternates: {', '.join(a.alternates)}." if a.alternates else ""
        lines.append(
            f"{i}. {a.segment_label} — {a.primary_model} ({a.fit_verdict or 'domain fit'}). "
            f"{a.rationale}{alt}"
        )

    if plan.ownership_note:
        lines.append("")
        lines.append(f"Ownership / utilization: {plan.ownership_note}")
    lines.append("")
    lines.append(plan.doctrine)
    return "\n".join(lines).strip()


def merge_fleet_into_recommendations(
    recommendations: List[AircraftRecommendation],
    plan: FleetCompositionPlan,
) -> List[AircraftRecommendation]:
    """Fleet primaries first — only domain-feasible assignments."""
    if not plan.multi_aircraft_required:
        return recommendations

    by_model = {r.model: r for r in recommendations}
    ordered: List[AircraftRecommendation] = []
    for a in plan.assignments:
        if not a.primary_model or not a.domain_feasible:
            continue
        rec = by_model.get(a.primary_model)
        if rec:
            ordered.append(rec)
    for r in recommendations:
        if r.model not in {x.model for x in ordered}:
            ordered.append(r)
    for i, r in enumerate(ordered, start=1):
        r.rank = i
    return ordered[:6]


def recommendations_from_fleet_plan(
    data_used: Optional[Dict[str, Any]],
    *,
    mission: Optional[MissionState] = None,
) -> List[AircraftRecommendation]:
    """Build recommendation stubs from fleet doctrine when ranking must defer to portfolio."""
    if not isinstance(data_used, dict):
        return []
    raw = data_used.get("fleet_composition_plan")
    if not isinstance(raw, dict) or not raw.get("multi_aircraft_required"):
        return []
    out: List[AircraftRecommendation] = []
    for i, a in enumerate(raw.get("assignments") or [], start=1):
        if not isinstance(a, dict):
            continue
        model = str(a.get("primary_model") or "").strip()
        if not model:
            continue
        spec = AIRCRAFT_PROFILES.get(model) or {}
        out.append(
            AircraftRecommendation(
                model=model,
                category=str(spec.get("category") or ""),
                total_score=0.72,
                confidence=0.65,
                rank=i,
                avoid=False,
                fit=a.get("fit_verdict") or "VIABLE WITH COMPROMISES",
                fit_verdict=a.get("fit_verdict") or "VIABLE WITH COMPROMISES",
                explanation=None,
            )
        )
    return out[:6]
