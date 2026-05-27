"""
Multi-domain operational analysis — not an aircraft preference problem.

Domains (examples):
  - TEB–LON → ULR class
  - Aspen/KASE → short-field / high-performance class
  - Caribbean hops → regional STOL / flexible ops class

A single aircraft spanning all active domains is structurally invalid when no model
passes every domain's hard gates — not merely "less optimal."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from services.consultant.mission_state import MissionState
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES
from services.mission.models import MissionProfile

_MOUNTAIN_ICAO_RE = re.compile(
    r"\b(?:KASE|KTEX|KEGE|KJAC|aspen|telluride|jackson\s+hole|eagle\s+county)\b",
    re.I,
)
_CARIBBEAN_RE = re.compile(
    r"\b(?:caribbean|nassau|st\s+maarten|st\s+thomas|mynn|tncm|tist|bahamas)\b",
    re.I,
)
_INTL_RE = re.compile(
    r"\b(?:london|paris|geneva|europe|france|uk|tokyo|hong\s+kong|transatlantic)\b",
    re.I,
)
_ONE_AIRCRAFT_ONLY_RE = re.compile(
    r"\b(?:one\s+aircraft\s+only|single\s+aircraft|only\s+one\s+jet|just\s+one\s+plane)\b",
    re.I,
)
_INDUSTRIAL_AIRPORT_RE = re.compile(
    r"\b(?:industrial\s+airports?|smaller\s+industrial|factory\s+site|plant\s+site)\b",
    re.I,
)

_DOMAIN_CATEGORY_ULR = frozenset({"ultra-long", "large"})
_DOMAIN_CATEGORY_SHORT_FIELD = frozenset({"light", "super-midsize", "midsize", "turboprop"})
_ULR_PRACTICAL_MIN_NM = 5200.0
_SHORT_FIELD_MIN = 0.85


class OperationalDomain(str, Enum):
    """Operational problem classes — not marketing categories."""

    ULR_CLASS = "ulr_class"
    SHORT_FIELD_HIGH_PERFORMANCE = "short_field_high_performance"
    REGIONAL_STOL_FLEX = "regional_stol_flex"


class SegmentationTrigger(str, Enum):
    """Why the pipeline decomposed the mission."""

    NONE = "none"
    ELIMINATION_FAILURE = "elimination_failure"
    USER_EXPLICIT_MULTI = "user_explicit_multi"
    IMPOSSIBLE_SINGLE_AIRCRAFT_CONSTRAINT = "impossible_single_aircraft_constraint"


@dataclass
class OperationalDomainSpec:
    domain: OperationalDomain
    label: str
    route_labels: List[str] = field(default_factory=list)
    stage_nm: float = 0.0
    required_nm: float = 0.0
    constraint_triggers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain.value,
            "label": self.label,
            "route_labels": list(self.route_labels),
            "stage_nm": round(self.stage_nm, 1),
            "required_nm": round(self.required_nm, 1),
            "constraint_triggers": list(self.constraint_triggers),
        }


@dataclass
class DomainFeasibilityTrace:
    """Per-domain elimination lineage for P3 audit."""

    domain: OperationalDomain
    feasible_models: List[str] = field(default_factory=list)
    eliminated_models: List[str] = field(default_factory=list)
    elimination_lineage: List[Dict[str, str]] = field(default_factory=list)
    corridor_classification: Optional[str] = None
    corridor_decision: Optional[str] = None
    payload_assumptions: Dict[str, Any] = field(default_factory=dict)
    constraint_triggers: List[str] = field(default_factory=list)
    airport_profiles: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain.value,
            "feasible_models": list(self.feasible_models),
            "eliminated_models": list(self.eliminated_models),
            "elimination_lineage": list(self.elimination_lineage),
            "corridor_classification": self.corridor_classification,
            "corridor_decision": self.corridor_decision,
            "payload_assumptions": dict(self.payload_assumptions),
            "constraint_triggers": list(self.constraint_triggers),
            "airport_profiles": list(self.airport_profiles),
        }


@dataclass
class MultiDomainAnalysis:
    domains: List[OperationalDomainSpec] = field(default_factory=list)
    multi_domain_required: bool = False
    single_aircraft_structurally_invalid: bool = False
    universal_survivors: List[str] = field(default_factory=list)
    trigger: SegmentationTrigger = SegmentationTrigger.NONE
    domain_traces: List[DomainFeasibilityTrace] = field(default_factory=list)
    doctrine: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domains": [d.to_dict() for d in self.domains],
            "multi_domain_required": self.multi_domain_required,
            "single_aircraft_structurally_invalid": self.single_aircraft_structurally_invalid,
            "universal_survivors": list(self.universal_survivors),
            "trigger": self.trigger.value,
            "domain_traces": [t.to_dict() for t in self.domain_traces],
            "doctrine": self.doctrine,
        }


def _norm(model: str) -> str:
    return re.sub(r"\s+", " ", (model or "").strip().lower())


def identify_operational_domains(
    profile: MissionProfile,
    mission: MissionState,
    *,
    query: str = "",
) -> List[OperationalDomainSpec]:
    """Detect distinct operational domains present in the mission — not preferences."""
    routes = list(mission.routes or []) or profile.route_labels()
    ql = (query or "").lower()
    query_blob = f"{' '.join(routes)} {query or ''}"
    domains: List[OperationalDomainSpec] = []

    peak_nm = 0.0
    route_res = []
    try:
        from services.mission.route_distance_authority import (
            resolve_mission_route_authority,
            peak_verified_stage_nm,
        )

        route_res = resolve_mission_route_authority(routes)
        peak_nm = peak_verified_stage_nm(route_res)
        if peak_nm <= 0 and route_res:
            peak_nm = max(r.distance_nm for r in route_res)
    except Exception:
        pass

    mountain_routes = [r for r in routes if _MOUNTAIN_ICAO_RE.search(r)]
    query_mountain = bool(_MOUNTAIN_ICAO_RE.search(query_blob))
    query_industrial = bool(_INDUSTRIAL_AIRPORT_RE.search(query_blob))
    if (
        mountain_routes
        or query_mountain
        or query_industrial
        or profile.mountain_airports
        or profile.mountain_airport_priority
        or mission.mountain_airport_requirement
        or (mission.runway_constraints and "short_field" in str(mission.runway_constraints))
    ):
        label = (
            "Domestic field-access (industrial & short-runway airports)"
            if query_industrial
            else "Short-field / high-performance (mountain & hot-high)"
        )
        domestic_routes = [r for r in routes if not _INTL_RE.search(r)]
        domains.append(
            OperationalDomainSpec(
                domain=OperationalDomain.SHORT_FIELD_HIGH_PERFORMANCE,
                label=label,
                route_labels=(
                    mountain_routes
                    or domestic_routes
                    or [r for r in routes if _MOUNTAIN_ICAO_RE.search(r)]
                    or (routes[:1] if routes else [])
                ),
                stage_nm=800.0,
                required_nm=1200.0,
                constraint_triggers=(
                    ["industrial_airport_access", "runway_length", "climb_gradient"]
                    if query_industrial
                    else ["airport_elevation", "hot_high", "climb_gradient", "runway_length"]
                ),
            )
        )

    intl_routes = [r for r in routes if _INTL_RE.search(r)]
    query_intl = bool(_INTL_RE.search(query_blob)) or "nonstop" in ql and peak_nm >= 2500
    if (
        intl_routes
        or query_intl
        or peak_nm >= 2800
        or profile.international_ops
        or profile.nonstop_required
    ):
        label_route = (
            intl_routes[0]
            if intl_routes
            else (routes[0] if routes else ("TEB → London" if "london" in ql else "ULR stage"))
        )
        domains.append(
            OperationalDomainSpec(
                domain=OperationalDomain.ULR_CLASS,
                label=f"ULR class — {label_route}",
                route_labels=intl_routes or routes[:1],
                stage_nm=peak_nm or 3100.0,
                required_nm=(peak_nm or 3100.0) + 400.0,
                constraint_triggers=["corridor_transatlantic", "catalog_nonstop", "nbaa_reserve"],
            )
        )

    caribbean_routes = [r for r in routes if _CARIBBEAN_RE.search(r)]
    if caribbean_routes or ("caribbean" in ql and "miami" in ql):
        domains.append(
            OperationalDomainSpec(
                domain=OperationalDomain.REGIONAL_STOL_FLEX,
                label="Regional STOL / flexible ops (Caribbean & short strips)",
                route_labels=caribbean_routes or routes[:2],
                stage_nm=600.0,
                required_nm=900.0,
                constraint_triggers=["short_runway", "high_cycle_economics", "tropical_performance"],
            )
        )

    if not domains and routes:
        domains.append(
            OperationalDomainSpec(
                domain=OperationalDomain.ULR_CLASS,
                label="Primary mission domain",
                route_labels=routes[:2],
                stage_nm=peak_nm or 2000.0,
                required_nm=(peak_nm or 2000.0) + 250.0,
                constraint_triggers=["stage_length"],
            )
        )

    # Deduplicate by domain type
    seen: Set[OperationalDomain] = set()
    unique: List[OperationalDomainSpec] = []
    for d in domains:
        if d.domain not in seen:
            seen.add(d.domain)
            unique.append(d)
    return unique[:3]


def _hard_gate_domain(
    model: str,
    spec: Dict[str, Any],
    domain: OperationalDomainSpec,
) -> Tuple[bool, Optional[str]]:
    """Domain hard gate — failure means structurally invalid for this domain."""
    cat = str(spec.get("category") or "").lower()
    practical = float(spec.get("practical_nm") or 0)
    short_field = float(spec.get("short_field_score") or 0.5)
    runway_ft = float(spec.get("runway_ft") or 5000)

    if domain.domain == OperationalDomain.SHORT_FIELD_HIGH_PERFORMANCE:
        if cat == "ultra-long":
            return False, "ultra_long_structurally_invalid_at_mountain_airport"
        if short_field < _SHORT_FIELD_MIN:
            return False, "insufficient_short_field_for_hot_high"
        if runway_ft > 4500 and short_field < 0.9:
            return False, "runway_footprint_exceeds_mountain_strip_envelope"
        return True, None

    if domain.domain == OperationalDomain.ULR_CLASS:
        if practical < domain.required_nm:
            return False, "practical_nm_below_ulr_domain_requirement"
        if cat in ("light", "midsize") and domain.stage_nm >= 3000:
            return False, "category_structurally_invalid_for_ulr_nonstop"
        if practical < _ULR_PRACTICAL_MIN_NM and domain.stage_nm >= 3000:
            if cat not in _DOMAIN_CATEGORY_ULR:
                return False, "not_ulr_class_for_verified_stage"
        return True, None

    if domain.domain == OperationalDomain.REGIONAL_STOL_FLEX:
        if cat == "ultra-long":
            return False, "ulr_structurally_invalid_for_caribbean_shuttle"
        if short_field < 0.65 and runway_ft > 4000:
            return False, "poor_stol_flex_for_regional_hops"
        return True, None

    return True, None


def _domain_elimination_pass(
    pool: Sequence[str],
    domain: OperationalDomainSpec,
    *,
    profile: MissionProfile,
    mission: MissionState,
    data_used: Optional[Dict[str, Any]] = None,
    global_eliminated: Optional[Set[str]] = None,
) -> DomainFeasibilityTrace:
    """Run authoritative constraints per domain; return trace + feasible set."""
    trace = DomainFeasibilityTrace(
        domain=domain.domain,
        constraint_triggers=list(domain.constraint_triggers),
    )
    # Per-domain corridor/airport/payload gates are authoritative — do not pre-filter
    # by mission-wide feasibility-map failures (e.g. PC-24 eliminated on ULR but valid for mountain).
    candidates = list(pool)
    specs = {m: AIRCRAFT_PROFILES.get(m) or {} for m in candidates}

    # Airport constraints (mountain / caribbean domains)
    if domain.domain in (
        OperationalDomain.SHORT_FIELD_HIGH_PERFORMANCE,
        OperationalDomain.REGIONAL_STOL_FLEX,
    ):
        routes = domain.route_labels or list(mission.routes or [])
        if routes:
            try:
                from services.airport.airport_operational_constraints import (
                    apply_airport_constraint_elimination,
                )

                airport_result = apply_airport_constraint_elimination(
                    list(candidates),
                    route_labels=routes,
                    model_specs=specs,
                )
                trace.airport_profiles = [a.to_dict() for a in airport_result.airports]
                for m in airport_result.eliminated:
                    trace.elimination_lineage.append(
                        {
                            "stage": "airport_constraint",
                            "model": m,
                            "reason": airport_result.reasons.get(m, "airport"),
                        }
                    )
                candidates = list(airport_result.survivors)
            except Exception:
                pass

    # Corridor (ULR domain)
    if domain.domain == OperationalDomain.ULR_CLASS:
        try:
            from services.elimination.corridor_elimination import apply_corridor_hard_elimination
            from services.mission.route_distance_authority import resolve_mission_route_authority

            routes = domain.route_labels or list(mission.routes or [])
            resolutions = resolve_mission_route_authority(routes)
            model_categories = {
                m.lower(): str((AIRCRAFT_PROFILES.get(m) or {}).get("category") or "")
                for m in candidates
            }
            corridor_result = apply_corridor_hard_elimination(
                list(candidates),
                profile,
                model_categories=model_categories,
                route_resolutions=resolutions,
            )
            trace.corridor_classification = corridor_result.corridor_id or None
            trace.corridor_decision = (
                f"verified_stage_nm={corridor_result.verified_stage_nm}; "
                f"confidence_min={corridor_result.route_confidence_min}"
            )
            for m in corridor_result.eliminated:
                trace.elimination_lineage.append(
                    {
                        "stage": "corridor",
                        "model": m,
                        "reason": corridor_result.reasons.get(m, "corridor"),
                    }
                )
            candidates = list(corridor_result.survivors)
        except Exception:
            pass

    # Payload assumptions (mission-level snapshot per domain)
    if isinstance(data_used, dict):
        op_ctx = data_used.get("mission_operational_context") or {}
        if isinstance(op_ctx, dict) and op_ctx.get("payload"):
            trace.payload_assumptions = dict(op_ctx.get("payload") or {})
        elif data_used.get("hye_reasoning_packet"):
            pkt = data_used["hye_reasoning_packet"]
            if isinstance(pkt, dict):
                trace.payload_assumptions = dict(pkt.get("payload_assumptions") or {})

    survivors: List[str] = []
    for model in candidates:
        spec = specs.get(model) or {}
        ok, reason = _hard_gate_domain(model, spec, domain)
        if ok:
            survivors.append(model)
        else:
            trace.eliminated_models.append(model)
            trace.elimination_lineage.append(
                {"stage": f"domain_{domain.domain.value}", "model": model, "reason": reason or "gate"}
            )

    trace.feasible_models = survivors
    return trace


def analyze_multi_domain_operational_problem(
    profile: MissionProfile,
    mission: MissionState,
    pool: Sequence[str],
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
    global_eliminated: Optional[Set[str]] = None,
) -> MultiDomainAnalysis:
    """
    Determine whether this is a multi-domain operational problem requiring decomposition.

    Segmentation triggers on elimination failure (no universal survivor), not preference.
    """
    domains = identify_operational_domains(profile, mission, query=query)
    if len(domains) < 2:
        doctrine = "Single operational domain — standard in-band selection applies."
        if isinstance(data_used, dict):
            from services.mission.mission_structure_resolution import (
                load_structure_resolution,
            )

            resolution = load_structure_resolution(data_used)
            if resolution and resolution.decomposition_required:
                doctrine = (
                    "Multi-domain operational problem — mission decomposition precedes "
                    "model selection. "
                    f"{resolution.decomposition_reason}"
                )
            raw_struct = data_used.get("structural_decomposition")
            if isinstance(raw_struct, dict) and raw_struct.get("required"):
                doctrine = (
                    "Multi-domain operational problem — one aircraft cannot span these "
                    "operational domains."
                )
        return MultiDomainAnalysis(
            domains=domains,
            doctrine=doctrine,
        )

    traces: List[DomainFeasibilityTrace] = []
    per_domain_survivors: List[Set[str]] = []
    for dom in domains:
        tr = _domain_elimination_pass(
            pool,
            dom,
            profile=profile,
            mission=mission,
            data_used=data_used,
            global_eliminated=global_eliminated,
        )
        traces.append(tr)
        per_domain_survivors.append({_norm(m) for m in tr.feasible_models})

    universal: Set[str] = per_domain_survivors[0].copy() if per_domain_survivors else set()
    for s in per_domain_survivors[1:]:
        universal &= s

    universal_models = [m for m in pool if _norm(m) in universal]
    ql = (query or "").lower()
    user_explicit = bool(
        re.search(r"\b(?:two\s+(?:aircraft|jets)|split\s+fleet|g\s*650.*pc[-\s]?24)\b", ql, re.I)
    )
    one_aircraft_only = bool(_ONE_AIRCRAFT_ONLY_RE.search(ql))

    if not universal_models:
        trigger = SegmentationTrigger.ELIMINATION_FAILURE
        if one_aircraft_only:
            trigger = SegmentationTrigger.IMPOSSIBLE_SINGLE_AIRCRAFT_CONSTRAINT
        return MultiDomainAnalysis(
            domains=domains,
            multi_domain_required=True,
            single_aircraft_structurally_invalid=True,
            universal_survivors=[],
            trigger=trigger,
            domain_traces=traces,
            doctrine=(
                "Multi-domain operational problem: no single aircraft passes all domain hard gates. "
                "A single platform spanning every domain is structurally invalid — not a ranking preference."
            ),
        )

    if user_explicit and len(domains) >= 2:
        return MultiDomainAnalysis(
            domains=domains,
            multi_domain_required=True,
            single_aircraft_structurally_invalid=False,
            universal_survivors=universal_models,
            trigger=SegmentationTrigger.USER_EXPLICIT_MULTI,
            domain_traces=traces,
            doctrine=(
                "User requested multi-aircraft framing; domains are documented per operational class."
            ),
        )

    pkt_incompatible = False
    if isinstance(data_used, dict):
        pkt = data_used.get("mission_understanding_packet") or {}
        if isinstance(pkt, dict):
            inf = pkt.get("inferred_constraints") or {}
            pkt_incompatible = bool(inf.get("incompatible_mission_bands"))

    if pkt_incompatible and len(domains) >= 2:
        return MultiDomainAnalysis(
            domains=domains,
            multi_domain_required=True,
            single_aircraft_structurally_invalid=not bool(universal_models),
            universal_survivors=universal_models,
            trigger=(
                SegmentationTrigger.ELIMINATION_FAILURE
                if not universal_models
                else SegmentationTrigger.USER_EXPLICIT_MULTI
            ),
            domain_traces=traces,
            doctrine=(
                "Incompatible operational bands — ULR oceanic corridors and domestic field-access "
                "require a portfolio approach; one platform is a compromise, not a clean fit."
            ),
        )

    return MultiDomainAnalysis(
        domains=domains,
        multi_domain_required=False,
        single_aircraft_structurally_invalid=False,
        universal_survivors=universal_models,
        trigger=SegmentationTrigger.NONE,
        domain_traces=traces,
        doctrine=(
            f"Multiple domains identified but {universal_models[0]} (and peers) pass all domain gates — "
            "single-aircraft selection remains valid."
        ),
    )
