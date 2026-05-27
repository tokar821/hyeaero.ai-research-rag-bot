"""
Mission authority kernel — immutable law for mission structure, fleet doctrine, and presented aircraft.

Packet + segment graph + structural proof = kernel.
Renderer and LLM may only project the kernel; they may not reinterpret semantics.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.consultant.route_feasibility import RouteFeasibilityAssessment
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES
from services.mission.mission_graph import (
    MissionGraph,
    MissionSegmentProfile,
    SegmentKind,
    build_mission_graph,
    load_mission_graph,
    save_mission_graph,
)
from services.mission.mission_ranking_projection import RankingProjectionTrace
from services.mission.mission_understanding_engine import (
    MissionUnderstandingPacket,
    format_ownership_economics_overlay,
    needs_ownership_overlay,
)
from services.mission.models import MissionProfile
from services.mission.structural_decomposition import (
    StructuralDecompositionProof,
    attach_structural_decomposition_proof,
    needs_structural_decomposition,
)

MISSION_AUTHORITY_KERNEL_KEY = "mission_authority_kernel"
KERNEL_BLOCK_MARKER = "OPERATIONAL SYNTHESIS (AUTHORITATIVE)"

_SINGLE_AIRCRAFT_COLLAPSE_RE = re.compile(
    r"\b(?:one aircraft|single aircraft|only one jet|same jet for (?:everything|all))\b",
    re.I,
)
_KERNEL_SECTION_RE = re.compile(
    rf"{re.escape(KERNEL_BLOCK_MARKER)}.*?(?=\n\n(?:Per-segment|Aircraft Options|Aircraft Class|Verdict:)|\Z)",
    re.DOTALL | re.I,
)


@dataclass
class SegmentAircraftRole:
    segment_id: str
    segment_label: str
    operational_band: str
    primary_model: str = ""
    fit_verdict: str = ""
    rationale: str = ""
    route_labels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "segment_label": self.segment_label,
            "operational_band": self.operational_band,
            "primary_model": self.primary_model,
            "fit_verdict": self.fit_verdict,
            "rationale": self.rationale,
            "route_labels": list(self.route_labels),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SegmentAircraftRole":
        return cls(
            segment_id=str(raw.get("segment_id") or ""),
            segment_label=str(raw.get("segment_label") or ""),
            operational_band=str(raw.get("operational_band") or ""),
            primary_model=str(raw.get("primary_model") or ""),
            fit_verdict=str(raw.get("fit_verdict") or ""),
            rationale=str(raw.get("rationale") or ""),
            route_labels=[str(r) for r in (raw.get("route_labels") or []) if r],
        )


_CATEGORY_BY_SEGMENT_KIND: Dict[SegmentKind, frozenset] = {
    SegmentKind.ULR_CONTINUATION: frozenset({"ultra-long", "large"}),
    SegmentKind.PACIFIC_ULR: frozenset({"ultra-long", "large"}),
    SegmentKind.TRANSATLANTIC_EXECUTIVE: frozenset({"large", "super-midsize", "ultra-long"}),
    SegmentKind.MOUNTAIN_FIELD: frozenset({"light", "super-midsize", "midsize", "turboprop"}),
    SegmentKind.INDUSTRIAL_FIELD: frozenset({"light", "super-midsize", "midsize"}),
    SegmentKind.CARIBBEAN_REGIONAL: frozenset({"light", "super-midsize", "midsize"}),
    SegmentKind.DOMESTIC_EXECUTIVE: frozenset({"super-midsize", "midsize", "large"}),
}


@dataclass
class MissionAuthorityKernel:
    """The only authoritative mission narrative + aircraft presentation law."""

    mission_fit_route: str = ""
    pax_display: str = "Not stated"
    operational_read: str = ""
    segments: List[MissionSegmentProfile] = field(default_factory=list)
    segment_roles: List[SegmentAircraftRole] = field(default_factory=list)
    fleet_doctrine: str = ""
    ownership_overlay: str = ""
    structural_decomposition: bool = False
    structural_reason: str = ""
    single_aircraft_forbidden: bool = False
    route_certainty_degraded: bool = False
    peak_segment_id: str = ""
    authorized_models: List[str] = field(default_factory=list)
    segment_bound_presentation: bool = False
    content_hash: str = ""

    def authorized_model_set(self) -> Set[str]:
        return {m.lower() for m in self.authorized_models if m}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_fit_route": self.mission_fit_route,
            "pax_display": self.pax_display,
            "operational_read": self.operational_read,
            "segments": [s.to_dict() for s in self.segments],
            "segment_roles": [r.to_dict() for r in self.segment_roles],
            "fleet_doctrine": self.fleet_doctrine,
            "ownership_overlay": self.ownership_overlay,
            "structural_decomposition": self.structural_decomposition,
            "structural_reason": self.structural_reason,
            "single_aircraft_forbidden": self.single_aircraft_forbidden,
            "route_certainty_degraded": self.route_certainty_degraded,
            "peak_segment_id": self.peak_segment_id,
            "authorized_models": list(self.authorized_models),
            "segment_bound_presentation": self.segment_bound_presentation,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> Optional["MissionAuthorityKernel"]:
        if not isinstance(raw, dict):
            return None
        segs = [
            MissionSegmentProfile.from_dict(s)
            for s in (raw.get("segments") or [])
            if isinstance(s, dict)
        ]
        roles = [
            SegmentAircraftRole.from_dict(r)
            for r in (raw.get("segment_roles") or [])
            if isinstance(r, dict)
        ]
        return cls(
            mission_fit_route=str(raw.get("mission_fit_route") or ""),
            pax_display=str(raw.get("pax_display") or "Not stated"),
            operational_read=str(raw.get("operational_read") or ""),
            segments=segs,
            segment_roles=roles,
            fleet_doctrine=str(raw.get("fleet_doctrine") or ""),
            ownership_overlay=str(raw.get("ownership_overlay") or ""),
            structural_decomposition=bool(raw.get("structural_decomposition")),
            structural_reason=str(raw.get("structural_reason") or ""),
            single_aircraft_forbidden=bool(raw.get("single_aircraft_forbidden")),
            route_certainty_degraded=bool(raw.get("route_certainty_degraded")),
            peak_segment_id=str(raw.get("peak_segment_id") or ""),
            authorized_models=[str(m) for m in (raw.get("authorized_models") or []) if m],
            content_hash=str(raw.get("content_hash") or ""),
        )


@dataclass
class KernelEnforcementReport:
    ok: bool = True
    reject_merge: bool = False
    violations: List[str] = field(default_factory=list)
    used_canonical_projection: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "reject_merge": self.reject_merge,
            "violations": list(self.violations),
            "used_canonical_projection": self.used_canonical_projection,
        }


def _dedupe_sentences(text: str) -> str:
    parts = [p.strip() for p in re.split(r"\.\s+", (text or "").strip()) if p.strip()]
    seen: set[str] = set()
    out: List[str] = []
    for p in parts:
        key = p.lower()[:120]
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    if not out:
        return ""
    return ". ".join(out) + ("." if out else "")


def _model_category(model: str) -> str:
    return str((AIRCRAFT_PROFILES.get(model) or {}).get("category") or "").lower()


def _segment_kinds_active(graph: MissionGraph) -> Set[SegmentKind]:
    return {s.kind for s in graph.segments}


def _category_allowed_for_kernel(model: str, graph: MissionGraph, packet: MissionUnderstandingPacket) -> bool:
    cat = _model_category(model)
    if not cat:
        return False
    kinds = _segment_kinds_active(graph)
    ceiling = (packet.inferred_constraints.get("planning_band_ceiling") or "").lower().replace("-", "_")
    if ceiling == "super_midsize" and cat == "ultra-long":
        if SegmentKind.ULR_CONTINUATION not in kinds and SegmentKind.PACIFIC_ULR not in kinds:
            return False
    if not kinds:
        if ceiling == "super_midsize":
            return cat in ("super-midsize", "midsize", "large")
        return True
    for kind in kinds:
        allowed = _CATEGORY_BY_SEGMENT_KIND.get(kind)
        if allowed and cat in allowed:
            if ceiling == "super_midsize" and cat == "ultra-long":
                continue
            return True
    return False


def _segment_roles_from_fleet(data_used: Optional[Dict[str, Any]], graph: MissionGraph) -> List[SegmentAircraftRole]:
    du = data_used if isinstance(data_used, dict) else {}
    raw = du.get("fleet_composition_plan")
    if not isinstance(raw, dict):
        return []
    roles: List[SegmentAircraftRole] = []
    seg_by_label = {s.label.lower(): s for s in graph.segments}
    for a in raw.get("assignments") or []:
        if not isinstance(a, dict):
            continue
        label = str(a.get("segment_label") or "")
        seg = seg_by_label.get(label.lower())
        roles.append(
            SegmentAircraftRole(
                segment_id=seg.segment_id if seg else label.lower().replace(" ", "_"),
                segment_label=label or (seg.label if seg else "Domain"),
                operational_band=(seg.operational_band if seg else ""),
                primary_model=str(a.get("primary_model") or ""),
                fit_verdict=str(a.get("fit_verdict") or ""),
                rationale=str(a.get("rationale") or ""),
                route_labels=list(seg.route_labels) if seg else [],
            )
        )
    return roles


def _bind_authorized_models(
    recommendations: Sequence[AircraftRecommendation],
    graph: MissionGraph,
    packet: MissionUnderstandingPacket,
    segment_roles: List[SegmentAircraftRole],
    *,
    structural: bool,
) -> List[str]:
    authorized: List[str] = []
    seen: set[str] = set()

    def _add(model: str) -> None:
        m = (model or "").strip()
        if not m:
            return
        key = m.lower()
        if key in seen:
            return
        if not _category_allowed_for_kernel(m, graph, packet):
            return
        seen.add(key)
        authorized.append(m)

    for role in segment_roles:
        if role.primary_model:
            _add(role.primary_model)

    if structural and segment_roles:
        return authorized

    for rec in recommendations:
        if getattr(rec, "avoid", False):
            continue
        _add(rec.model)

    return authorized


def _fleet_doctrine_text(
    data_used: Optional[Dict[str, Any]],
    proof: StructuralDecompositionProof,
    *,
    structure_resolution: Optional[Any] = None,
) -> str:
    if not proof.required:
        return ""
    du = data_used if isinstance(data_used, dict) else {}
    raw = du.get("fleet_composition_plan")
    if isinstance(raw, dict) and raw.get("doctrine"):
        doctrine = str(raw.get("doctrine") or "").strip()
        if structure_resolution is not None:
            from services.mission.mission_structure_resolution import sanitize_doctrine_text

            doctrine = sanitize_doctrine_text(doctrine, structure_resolution)
        try:
            from services.fleet.fleet_composition import (
                FleetCompositionPlan,
                FleetRoleAssignment,
                MissionSegment,
                MissionSegmentRole,
                format_fleet_composition_block,
            )

            plan = FleetCompositionPlan(
                multi_aircraft_required=bool(
                    raw.get("multi_aircraft_required") or raw.get("multi_domain_required")
                ),
                doctrine=doctrine,
                ownership_note=str(raw.get("ownership_note") or ""),
                single_aircraft_structurally_invalid=bool(
                    raw.get("single_aircraft_structurally_invalid")
                ),
            )
            for s in raw.get("segments") or []:
                if isinstance(s, dict):
                    try:
                        seg_role = MissionSegmentRole(str(s.get("role") or "coast_to_coast"))
                    except ValueError:
                        seg_role = MissionSegmentRole.COAST_TO_COAST
                    plan.segments.append(
                        MissionSegment(
                            role=seg_role,
                            label=str(s.get("label") or ""),
                            stage_nm=float(s.get("stage_nm") or 0),
                            required_nm=float(s.get("required_nm") or 0),
                            route_labels=list(s.get("route_labels") or []),
                        )
                    )
            for a in raw.get("assignments") or []:
                if isinstance(a, dict):
                    try:
                        role = MissionSegmentRole(str(a.get("role") or "coast_to_coast"))
                    except ValueError:
                        role = MissionSegmentRole.COAST_TO_COAST
                    plan.assignments.append(
                        FleetRoleAssignment(
                            role=role,
                            segment_label=str(a.get("segment_label") or ""),
                            primary_model=str(a.get("primary_model") or ""),
                            fit_verdict=str(a.get("fit_verdict") or ""),
                            rationale=str(a.get("rationale") or ""),
                            alternates=list(a.get("alternates") or []),
                        )
                    )
            block = format_fleet_composition_block(plan)
            if block:
                return block
        except Exception:
            pass
        return doctrine
    if proof.reason:
        return (
            "Fleet Structure:\n\n"
            "* Structural decomposition required — one aircraft cannot span these operational domains.\n"
            f"* Proof: {proof.reason}"
        )
    return ""


def _route_display(
    mission: MissionState,
    graph: MissionGraph,
    trace: Optional[RankingProjectionTrace],
    *,
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    from services.mission.mission_ranking_projection import order_routes_by_stage_nm
    from services.mission.planning_hierarchy_resolver import (
        PLANNING_HIERARCHY_KEY,
        PlanningHierarchy,
        format_peak_route_display,
    )

    if isinstance(data_used, dict):
        raw_h = data_used.get(PLANNING_HIERARCHY_KEY)
        if isinstance(raw_h, dict) and raw_h.get("peak_leg"):
            hierarchy = PlanningHierarchy(
                peak_leg=str(raw_h.get("peak_leg") or ""),
                continuation_legs=list(raw_h.get("continuation_legs") or []),
                supporting_legs=list(raw_h.get("supporting_legs") or []),
                utilization_legs=list(raw_h.get("utilization_legs") or []),
                peak_segment_id=str(raw_h.get("peak_segment_id") or ""),
            )
            return format_peak_route_display(hierarchy, "")

    peak = next((s for s in graph.segments if s.is_peak_planning), None)
    if peak and peak.route_labels:
        primary = peak.route_labels[0]
        others = [r for s in graph.segments for r in s.route_labels if r != primary]
        if others:
            return f"{primary} (peak planning); other legs: {'; '.join(others[:3])}"
        return primary
    routes = list(mission.routes or [])
    if trace and trace.route_display_order:
        routes = list(trace.route_display_order)
    elif routes:
        routes = order_routes_by_stage_nm(routes)
    if not routes:
        return "Not fully resolved — segment structure below still applies"
    if len(routes) == 1:
        return routes[0]
    return "; ".join(routes[:4])


def _kernel_content_hash(kernel: MissionAuthorityKernel) -> str:
    blob = "|".join(
        [
            kernel.operational_read[:500],
            kernel.mission_fit_route,
            str(kernel.structural_decomposition),
            ",".join(kernel.authorized_models),
            ",".join(r.primary_model for r in kernel.segment_roles),
        ]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def build_mission_authority_kernel(
    mission: MissionState,
    packet: MissionUnderstandingPacket,
    profile: Optional[MissionProfile] = None,
    *,
    recommendations: Optional[Sequence[AircraftRecommendation]] = None,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
    route_certainty_degraded: bool = False,
    projection_trace: Optional[RankingProjectionTrace] = None,
    feasible_models: Optional[Sequence[str]] = None,
) -> MissionAuthorityKernel:
    """Build immutable mission law once per turn — all downstream layers consume this."""
    prof = profile or _empty_profile(mission)
    proof = needs_structural_decomposition(
        packet,
        profile=prof,
        mission=mission,
        query=query,
        data_used=data_used,
        feasible_models=feasible_models,
    )
    attach_structural_decomposition_proof(data_used, proof)

    graph = load_mission_graph(data_used)
    if graph is None:
        graph = build_mission_graph(
            packet,
            prof,
            mission,
            structural_incompatibility=proof.required,
            query=query,
        )
        if isinstance(data_used, dict):
            save_mission_graph(data_used, graph)

    from services.mission.phase2_structural_synthesis import (
        apply_phase2_structural_synthesis,
        freeze_authoritative_operational_kernel,
    )

    graph, structure_resolution, _segment_auths, suppression = apply_phase2_structural_synthesis(
        graph,
        packet,
        prof,
        mission,
        query=query,
        data_used=data_used,
        feasible_models=feasible_models,
    )
    if structure_resolution.decomposition_required:
        proof = StructuralDecompositionProof(
            required=True,
            reason=structure_resolution.decomposition_reason or proof.reason,
            proof_kind=structure_resolution.proof_source or proof.proof_kind,
        )

    from services.mission.mission_presentation_policy import requires_segment_bound_presentation

    segment_roles = _segment_roles_from_fleet(data_used, graph)
    segment_bound = requires_segment_bound_presentation(
        graph, prof, packet, mission=mission
    ) or bool(packet.inferred_constraints.get("defer_global_shortlist"))
    recs = list(recommendations or [])
    structural_bind = proof.required or segment_bound or suppression.suppress_aircraft_specificity
    authorized = _bind_authorized_models(
        recs, graph, packet, segment_roles, structural=structural_bind
    )
    if suppression.suppress_aircraft_specificity:
        authorized = []

    fleet_raw = (data_used or {}).get("fleet_composition_plan") if isinstance(data_used, dict) else {}
    single_invalid = bool(
        isinstance(fleet_raw, dict) and fleet_raw.get("single_aircraft_structurally_invalid")
    )

    pax_disp = "Not stated"
    if mission.passenger_count is not None:
        pax_disp = str(mission.passenger_count)
        dist_raw = packet.explicit_constraints.get("passenger_distribution")
        if isinstance(dist_raw, dict) and dist_raw.get("is_variable"):
            lo, hi = dist_raw.get("min_pax"), dist_raw.get("max_pax")
            if lo is not None and hi is not None:
                pax_disp = f"{lo}–{hi} (planning load {mission.passenger_count})"

    kernel = MissionAuthorityKernel(
        mission_fit_route=_route_display(
            mission, graph, projection_trace, data_used=data_used
        ),
        pax_display=pax_disp,
        operational_read=_dedupe_sentences((packet.operational_synthesis or "").strip()),
        segments=list(graph.segments),
        segment_roles=segment_roles,
        fleet_doctrine=_fleet_doctrine_text(
            data_used, proof, structure_resolution=structure_resolution
        ).strip(),
        ownership_overlay=(
            format_ownership_economics_overlay(query, mission).strip()
            if needs_ownership_overlay(query, packet)
            else ""
        ),
        structural_decomposition=structure_resolution.decomposition_required,
        structural_reason=structure_resolution.decomposition_reason or proof.reason,
        single_aircraft_forbidden=structure_resolution.decomposition_required
        and (single_invalid or bool(segment_roles) or len(graph.segments) >= 2),
        route_certainty_degraded=route_certainty_degraded,
        peak_segment_id=graph.peak_segment_id,
        authorized_models=authorized,
        segment_bound_presentation=segment_bound and not proof.required,
    )
    kernel.content_hash = _kernel_content_hash(kernel)

    if isinstance(data_used, dict):
        freeze_authoritative_operational_kernel(data_used, kernel.to_dict())
        data_used[MISSION_AUTHORITY_KERNEL_KEY] = kernel.to_dict()
        data_used["mission_authority_bound"] = 1
        data_used["narrative_authority_built"] = 1

    return kernel


def _empty_profile(mission: MissionState) -> MissionProfile:
    from services.mission.models import Route

    p = MissionProfile()
    if mission.passenger_count:
        p.passengers = mission.passenger_count
    for lbl in mission.routes or []:
        r = Route.from_label(lbl)
        if r:
            p.routes.append(r)
    return p


def render_kernel_synthesis(kernel: MissionAuthorityKernel) -> str:
    """Projection only — no new reasoning."""
    lines: List[str] = [
        KERNEL_BLOCK_MARKER,
        "",
        "Mission Fit:",
        "",
        f"* Route: {kernel.mission_fit_route}",
        f"* Pax: {kernel.pax_display}",
    ]
    if kernel.route_certainty_degraded:
        lines.append(
            "* Route certainty: degraded — kernel segment structure remains binding."
        )
    if kernel.operational_read:
        lines.append(f"* Operational read: {kernel.operational_read}")

    if kernel.segments:
        lines.extend(["", "Operational segments:", ""])
        for seg in kernel.segments:
            auth_raw = (seg.constraints or {}).get("segment_authority")
            band = (seg.operational_band or "").strip()
            routes = "; ".join(seg.route_labels[:3]) if seg.route_labels else ""
            if not routes:
                continue
            peak = " [peak planning leg]" if seg.is_peak_planning else ""
            why = ""
            implication = ""
            conflict = ""
            if isinstance(auth_raw, dict):
                why = str(auth_raw.get("why_exists") or "").strip()
                implication = str(auth_raw.get("implication") or "").strip()
                conflict = str(auth_raw.get("conflict_note") or "").strip()
            lines.append(f"* Operational Segment: {seg.label}{peak}")
            # Compatibility: ensure legacy phrase appears for ME continuation segments.
            if seg.kind == SegmentKind.ULR_CONTINUATION:
                if "Middle East ULR continuation" not in why:
                    lines.append("  Authority: Middle East ULR continuation")
            if why:
                lines.append(f"  Authority: {why}")
            elif band:
                lines.append(f"  Authority: {band}")
            lines.append(f"  Routes: {routes}")
            constraint_keys = [
                k
                for k, v in (seg.constraints or {}).items()
                if k != "segment_authority" and v
            ]
            if constraint_keys:
                lines.append(f"  Operational constraint: {', '.join(constraint_keys)}")
            if conflict:
                lines.append(f"  Conflict: {conflict}")
            if implication:
                lines.append(f"  Implication: {implication}")

    if kernel.structural_decomposition and kernel.fleet_doctrine:
        lines.extend(["", kernel.fleet_doctrine])
    elif kernel.structural_decomposition and kernel.structural_reason:
        lines.extend(
            [
                "",
                "Fleet Structure:",
                "",
                f"* {kernel.structural_reason}",
            ]
        )

    if kernel.ownership_overlay:
        lines.extend(["", kernel.ownership_overlay])

    return "\n".join(lines).strip()


def render_kernel_aircraft_section(
    kernel: MissionAuthorityKernel,
    recommendations: Sequence[AircraftRecommendation],
) -> str:
    """Per-segment roles when structural; otherwise authorized pipeline models only."""
    _segment_scoped = (
        kernel.segment_bound_presentation
        or (
            kernel.structural_decomposition
            and len(kernel.segments) >= 2
            and not kernel.segment_roles
        )
    )
    if _segment_scoped and kernel.segments and not (
        kernel.structural_decomposition and kernel.segment_roles
    ):
        # Legacy contract: keep "Aircraft Options" header even when segment-scoped.
        lines = ["Aircraft Options:", "", "Per-segment operational posture:", ""]
        for seg in kernel.segments[:6]:
            band = (seg.operational_band or "").strip()
            routes = (
                "; ".join(seg.route_labels[:3])
                if seg.route_labels
                else "routes bound to this segment"
            )
            header = f"* {seg.label}"
            if band:
                header += f": {band}"
            lines.append(header)
            lines.append(f"  Routes: {routes}")
        lines.extend(
            [
                "",
                "* Global aircraft shortlist suppressed — planning stays segment-scoped "
                "until each corridor's runway and stage requirements are isolated.",
            ]
        )
        return "\n".join(lines).strip()

    from services.mission.recommendation_suppression import STRUCTURE_GUIDANCE

    if kernel.structural_decomposition and not kernel.segment_roles:
        return (
            f"{STRUCTURE_GUIDANCE}\n\n"
            "Per-segment class bands apply — global aircraft shortlist suppressed until "
            "each corridor's structural conflicts are resolved."
        )

    if kernel.structural_decomposition and kernel.segment_roles:
        lines = ["Per-segment aircraft roles:", ""]
        for role in kernel.segment_roles:
            if role.primary_model:
                lines.append(
                    f"* {role.segment_label}: {role.primary_model}"
                    + (f" — {role.fit_verdict}" if role.fit_verdict else "")
                )
                if role.rationale:
                    lines.append(f"  Rationale: {role.rationale}")
            elif role.operational_band:
                lines.append(f"* {role.segment_label}: class band — {role.operational_band}")
        if kernel.single_aircraft_forbidden:
            lines.extend(
                [
                    "",
                    "* Single-aircraft shortlist is invalid for this mission — portfolio roles above are binding.",
                ]
            )
        return "\n".join(lines).strip()

    allowed = kernel.authorized_model_set()
    viable = []
    seen: set[str] = set()
    for rec in recommendations:
        if getattr(rec, "avoid", False):
            continue
        key = (rec.model or "").lower()
        if not key or key in seen:
            continue
        if allowed and key not in allowed:
            continue
        seen.add(key)
        viable.append(rec)

    if not viable and kernel.segments:
        lines = ["Aircraft Class Band:", ""]
        for seg in kernel.segments[:4]:
            if seg.operational_band:
                lines.append(f"* [{seg.label}] {seg.operational_band}")
        return "\n".join(lines).strip()

    if not viable:
        return ""

    from services.consultant.response_architecture import format_recommendation_options_and_verdict

    return format_recommendation_options_and_verdict(viable[:3])


def render_kernel_verdict(kernel: MissionAuthorityKernel, recommendations: Sequence[AircraftRecommendation]) -> str:
    if kernel.structural_decomposition and kernel.segment_roles:
        models = [r.primary_model for r in kernel.segment_roles if r.primary_model]
        detail = ", ".join(models) if models else "per-segment class bands — see roles above"
        return f"* VIABLE WITH COMPROMISES: multi-aircraft portfolio — {detail}"
    allowed = kernel.authorized_models[:3]
    if allowed:
        return f"* VIABLE WITH COMPROMISES: {', '.join(allowed)}"
    return "* VIABLE WITH COMPROMISES: segment class guidance pending"


def project_kernel_advisory(
    kernel: MissionAuthorityKernel,
    recommendations: Sequence[AircraftRecommendation],
    *,
    opener: str = "",
) -> str:
    """Canonical full advisory — single render path."""
    from services.broker.broker_language import sanitize_broker_language

    parts = [render_kernel_synthesis(kernel)]
    if opener:
        parts.extend(["", opener])
    aircraft = render_kernel_aircraft_section(kernel, recommendations)
    if aircraft:
        parts.extend(["", aircraft])
    parts.extend(["", "Verdict:", "", render_kernel_verdict(kernel, recommendations)])
    text = "\n".join(parts).strip()
    return sanitize_broker_language(dedupe_kernel_body(text))


def dedupe_kernel_body(body: str) -> str:
    text = (body or "").strip()
    if not text:
        return text
    markers = list(re.finditer(re.escape(KERNEL_BLOCK_MARKER), text))
    if len(markers) > 1:
        cut = markers[1].start()
        tail = _KERNEL_SECTION_RE.sub("", text[cut:])
        text = text[:cut] + re.sub(r"\n{3,}", "\n\n", tail)
    return text.strip()


def filter_recommendations_by_kernel(
    recommendations: Sequence[AircraftRecommendation],
    kernel: MissionAuthorityKernel,
) -> List[AircraftRecommendation]:
    """Drop models not authorized by segment binding — no flagship fallback injection."""
    if kernel.structural_decomposition and kernel.single_aircraft_forbidden:
        allowed = kernel.authorized_model_set()
        if not allowed:
            return []
        out = []
        for rec in recommendations:
            if (rec.model or "").lower() in allowed:
                out.append(rec)
        return out
    allowed = kernel.authorized_model_set()
    if not allowed:
        return []
    return [
        r
        for r in recommendations
        if not getattr(r, "avoid", False) and (r.model or "").lower() in allowed
    ]


def _detect_kernel_violations(
    answer: str,
    kernel: MissionAuthorityKernel,
) -> List[str]:
    violations: List[str] = []
    text = answer or ""
    if KERNEL_BLOCK_MARKER not in text:
        violations.append("missing_kernel_marker")
    if text.count(KERNEL_BLOCK_MARKER) > 1:
        violations.append("duplicate_kernel_marker")
    if kernel.operational_read and len(kernel.operational_read) > 24:
        if kernel.operational_read[:24].lower() not in text.lower():
            violations.append("operational_read_diverged")
    try:
        from services.consultant.recommendation_engine import detect_models_from_text

        mentioned = detect_models_from_text(text)
        allowed = kernel.authorized_model_set()
        if kernel.structural_decomposition and kernel.segment_roles:
            allowed = allowed | {r.primary_model.lower() for r in kernel.segment_roles if r.primary_model}
        for m in mentioned:
            if allowed and m.lower() not in allowed:
                violations.append(f"unauthorized_aircraft:{m}")
    except Exception:
        pass
    if kernel.single_aircraft_forbidden and _SINGLE_AIRCRAFT_COLLAPSE_RE.search(text):
        violations.append("single_aircraft_collapse")
    if kernel.structural_decomposition and "Operational segments:" not in text:
        violations.append("missing_segment_structure")
    return violations


def enforce_kernel_authority(
    answer: str,
    kernel: MissionAuthorityKernel,
    recommendations: Sequence[AircraftRecommendation],
    *,
    opener: str = "",
) -> Tuple[str, KernelEnforcementReport]:
    """
    Reject LLM merge when prose diverges from kernel law; return canonical projection.
    """
    report = KernelEnforcementReport(ok=True)
    violations = _detect_kernel_violations(answer, kernel)
    if violations:
        report.ok = False
        report.reject_merge = True
        report.violations = violations
        report.used_canonical_projection = True
        canonical = project_kernel_advisory(
            kernel,
            filter_recommendations_by_kernel(recommendations, kernel),
            opener=opener,
        )
        return canonical, report

    deduped = dedupe_kernel_body(answer)
    if deduped != answer:
        report.violations.append("deduped_duplicate_sections")
    return deduped, report


def load_mission_authority_kernel(
    data_used: Optional[Dict[str, Any]],
) -> Optional[MissionAuthorityKernel]:
    if not isinstance(data_used, dict):
        return None
    return MissionAuthorityKernel.from_dict(data_used.get(MISSION_AUTHORITY_KERNEL_KEY))


def attach_kernel_enforcement_report(
    data_used: Optional[Dict[str, Any]],
    report: KernelEnforcementReport,
) -> None:
    if isinstance(data_used, dict):
        data_used["kernel_authority_enforcement"] = report.to_dict()


__all__ = [
    "KERNEL_BLOCK_MARKER",
    "MISSION_AUTHORITY_KERNEL_KEY",
    "KernelEnforcementReport",
    "MissionAuthorityKernel",
    "SegmentAircraftRole",
    "attach_kernel_enforcement_report",
    "build_mission_authority_kernel",
    "dedupe_kernel_body",
    "enforce_kernel_authority",
    "filter_recommendations_by_kernel",
    "load_mission_authority_kernel",
    "project_kernel_advisory",
    "render_kernel_synthesis",
]
