"""
HACK v1 — Hard Aviation Constraint Kernel.

Authoritative physics gate that runs BEFORE ranking, tier recovery, and recommendation
generation. Rejected aircraft are permanently excluded for the query turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from services.consultant.mission_state import MissionState
from services.mission.adapters import mission_state_to_profile
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES
from services.mission.feasibility_engine import (
    _is_westbound_profile,
    _is_winter_westbound_transpacific,
    compute_practical_range,
    peak_required_route_nm,
    required_route_nm_with_margin,
)
from services.mission.models import MissionProfile, Route

HACK_V1_METADATA_KEY = "hack_v1"
HACK_V1_EMPTY_MESSAGE = "NO PHYSICALLY VIABLE AIRCRAFT IN CURRENT CONSTRAINT SPACE"

_LIGHT_CATEGORIES = frozenset({"light", "turboprop"})
_MIDSIZE_CATEGORIES = frozenset({"midsize"})
_SUPER_MID_CATEGORIES = frozenset({"super-midsize"})
_LARGE_PLUS = frozenset({"large", "ultra-long", "ultra_long"})

# Canonical light jets that must never appear on long-stage missions
_LIGHT_JET_MODELS = frozenset(
    {"Citation CJ2", "Citation CJ4", "Learjet 75", "Phenom 300", "Pilatus PC-12"}
)

_TRANSATLANTIC_MIN_NM = 2800.0
_LARGE_STAGE_MIN_NM = 3500.0
_ULR_STAGE_MIN_NM = 5000.0
_GRAVEL_MIN_SHORT_FIELD = 0.78
_ARCTIC_MIN_SHORT_FIELD = 0.72

_EUROPE_RE = re.compile(
    r"\b(?:london|paris|zurich|geneva|frankfurt|europe|uk|united\s+kingdom)\b",
    re.I,
)
_US_WEST_RE = re.compile(
    r"\b(?:san\s+francisco|los\s+angeles|seattle|sfo|lax)\b",
    re.I,
)
_GRAVEL_ARCTIC_RE = re.compile(
    r"\b(?:gravel|arctic|nunavut|yellowknife|remote\s+strip|unpaved|bush)\b",
    re.I,
)
_MOUNTAIN_RE = re.compile(
    r"\b(?:aspen|jackson\s+hole|telluride|mountain|hot/high|hot\s+and\s+high)\b",
    re.I,
)


@dataclass(frozen=True)
class HackV1Rejection:
    model: str
    rule_id: str
    reason: str

    def to_dict(self) -> Dict[str, str]:
        return {"model": self.model, "rule_id": self.rule_id, "reason": self.reason}


@dataclass
class HackV1Result:
    """Kernel output contract — feasible list + rejection log only."""

    feasible_aircraft_list: List[str] = field(default_factory=list)
    rejection_log: List[HackV1Rejection] = field(default_factory=list)
    constraint_empty: bool = False
    mission_context: Dict[str, Any] = field(default_factory=dict)

    @property
    def permanent_exclusions(self) -> Set[str]:
        return {r.model for r in self.rejection_log}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feasible_aircraft_list": list(self.feasible_aircraft_list),
            "rejection_log": [r.to_dict() for r in self.rejection_log],
            "constraint_empty": self.constraint_empty,
            "mission_context": dict(self.mission_context),
            "permanent_exclusions": sorted(self.permanent_exclusions),
        }


def load_hack_v1_result(data_used: Optional[Dict[str, Any]]) -> Optional[HackV1Result]:
    if not isinstance(data_used, dict):
        return None
    raw = data_used.get(HACK_V1_METADATA_KEY)
    if not isinstance(raw, dict):
        return None
    rejections = [
        HackV1Rejection(
            model=str(r.get("model") or ""),
            rule_id=str(r.get("rule_id") or ""),
            reason=str(r.get("reason") or ""),
        )
        for r in (raw.get("rejection_log") or [])
        if isinstance(r, dict)
    ]
    return HackV1Result(
        feasible_aircraft_list=list(raw.get("feasible_aircraft_list") or []),
        rejection_log=rejections,
        constraint_empty=bool(raw.get("constraint_empty")),
        mission_context=dict(raw.get("mission_context") or {}),
    )


def hack_v1_permanent_exclusions(data_used: Optional[Dict[str, Any]]) -> frozenset[str]:
    loaded = load_hack_v1_result(data_used)
    if loaded is not None:
        return frozenset(loaded.permanent_exclusions)
    if isinstance(data_used, dict):
        raw = data_used.get("hack_v1_permanent_exclusions")
        if isinstance(raw, (list, tuple, set)):
            return frozenset(str(m) for m in raw if m)
    return frozenset()


def hack_v1_constraint_empty(data_used: Optional[Dict[str, Any]]) -> bool:
    if isinstance(data_used, dict) and data_used.get("hack_v1_constraint_empty"):
        return True
    loaded = load_hack_v1_result(data_used)
    if loaded is None:
        return False
    if loaded.constraint_empty:
        return True
    # Zero survivors after intersecting upstream feasibility with HACK v1.
    return not (loaded.feasible_aircraft_list or [])


def attach_hack_v1_metadata(data_used: Dict[str, Any], result: HackV1Result) -> None:
    data_used[HACK_V1_METADATA_KEY] = result.to_dict()
    data_used["hack_v1_permanent_exclusions"] = sorted(result.permanent_exclusions)
    data_used["hack_v1_constraint_empty"] = result.constraint_empty
    data_used["hack_v1_feasible_aircraft"] = list(result.feasible_aircraft_list)


def _route_blob(profile: MissionProfile, query: str = "") -> str:
    parts = [r.label() for r in (profile.routes or [])]
    parts.append(query or "")
    return " ".join(parts).lower()


def _route_distance_nm(route: Route) -> float:
    from services.mission.feasibility_engine import estimate_route_distance_nm

    return estimate_route_distance_nm(route.label())


def _peak_required_nm_for_aircraft(
    profile: MissionProfile,
    spec: Dict[str, Any],
) -> float:
    """
    Worst-case required NM only on legs this airframe's band can physically attempt.

    Occasional ULR legs do not poison field-aircraft evaluation (dominant utilization).
    """
    routes = profile.routes or []
    if not routes:
        return peak_required_route_nm(profile)

    practical = float(spec.get("practical_nm") or 0)
    category = str(spec.get("category") or "").lower()
    peaks: List[float] = []

    for route in routes:
        dist = _route_distance_nm(route)
        if dist <= 0:
            continue
        if practical > 0 and dist > practical * 1.12:
            if category in _LIGHT_CATEGORIES | _MIDSIZE_CATEGORIES | _SUPER_MID_CATEGORIES:
                continue
        peaks.append(required_route_nm_with_margin(profile, route=route))

    if peaks:
        return max(peaks)
    # No leg lies in this airframe's envelope — do not fall back to global ULR peak.
    return 0.0


def _mission_type_flags(
    profile: MissionProfile,
    *,
    query: str = "",
) -> Dict[str, Any]:
    blob = _route_blob(profile, query)
    required_nm = peak_required_route_nm(profile)
    pax = profile.passengers or 6

    transatlantic = bool(
        required_nm >= _TRANSATLANTIC_MIN_NM
        or (_EUROPE_RE.search(blob) and _US_WEST_RE.search(blob))
        or _EUROPE_RE.search(blob)
    )
    transpacific = bool(
        required_nm >= 4200
        and re.search(r"tokyo|seoul|singapore|hong\s+kong|beijing", blob)
        and re.search(r"san\s+francisco|los\s+angeles|seattle|new\s+york", blob)
    )
    westbound = _is_westbound_profile(profile)
    winter_westbound_transatlantic = (
        transatlantic and westbound and _is_winter_westbound_transpacific(profile)
    ) or (
        transatlantic
        and westbound
        and re.search(r"winter", (profile.seasonal_note or "").lower() + blob)
    )
    gravel_arctic = bool(_GRAVEL_ARCTIC_RE.search(blob))
    mountain = bool(
        profile.mountain_airport_priority
        or profile.mountain_airports
        or _MOUNTAIN_RE.search(blob)
    )

    return {
        "required_route_nm": round(required_nm, 1),
        "passenger_load": pax,
        "transatlantic": transatlantic,
        "transpacific": transpacific,
        "westbound": westbound,
        "winter_westbound_transatlantic": winter_westbound_transatlantic,
        "gravel_arctic": gravel_arctic,
        "mountain": mountain,
    }


def _minimum_certified_categories(required_nm: float, ctx: Dict[str, Any]) -> Set[str]:
    """Aircraft category must be in this set — no cross-band promotion."""
    if required_nm <= 0:
        return {"ultra-long", "ultra_long", "large", "super-midsize", "midsize", "light", "turboprop"}
    if required_nm >= _ULR_STAGE_MIN_NM or ctx.get("transpacific"):
        return {"ultra-long", "ultra_long"}
    if required_nm >= _LARGE_STAGE_MIN_NM:
        return {"ultra-long", "ultra_long", "large"}
    if required_nm >= _TRANSATLANTIC_MIN_NM or ctx.get("transatlantic"):
        return {"ultra-long", "ultra_long", "large", "super-midsize"}
    return {"ultra-long", "ultra_long", "large", "super-midsize", "midsize", "light", "turboprop"}


def _evaluate_model(
    model: str,
    profile: MissionProfile,
    ctx: Dict[str, Any],
) -> Optional[HackV1Rejection]:
    spec = AIRCRAFT_PROFILES.get(model)
    if not spec:
        return HackV1Rejection(model, "unknown_model", f"Unknown aircraft profile: {model}")

    category = str(spec.get("category") or "").lower()
    pax = int(ctx.get("passenger_load") or 6)
    required_nm = _peak_required_nm_for_aircraft(profile, spec)
    ctx_eval = {**ctx, "required_route_nm": required_nm}

    # --- Class band hard blocks (before range math) ---
    if ctx.get("winter_westbound_transatlantic"):
        if category in _LIGHT_CATEGORIES | _MIDSIZE_CATEGORIES | _SUPER_MID_CATEGORIES:
            return HackV1Rejection(
                model,
                "westbound_winter_transatlantic_class",
                "Winter westbound transatlantic — light and midsize jets cannot hold reserves and headwind margin.",
            )

    if ctx_eval.get("transatlantic") and pax > 6 and category in _LIGHT_CATEGORIES:
        return HackV1Rejection(
            model,
            "transatlantic_light_jet_pax",
            f"Transatlantic mission with {pax} passengers — light-jet class is not certified for this band.",
        )

    if model in _LIGHT_JET_MODELS and (
        ctx_eval.get("transatlantic")
        or ctx_eval.get("transpacific")
        or float(ctx.get("required_route_nm") or 0) >= _TRANSATLANTIC_MIN_NM
    ):
        return HackV1Rejection(
            model,
            "light_jet_long_stage",
            f"Light jet cannot satisfy {int(required_nm)} nm stage with NBAA reserves (e.g. transatlantic / ULR corridor).",
        )

    allowed_cats = _minimum_certified_categories(required_nm, ctx_eval)
    if category not in allowed_cats:
        return HackV1Rejection(
            model,
            "class_band_violation",
            f"Mission requires {', '.join(sorted(allowed_cats))} band; {model} is {category}.",
        )

    # --- Runway / field type ---
    short_field = float(spec.get("short_field_score") or 0)
    runway_ft = float(spec.get("runway_ft") or 5000)

    if ctx.get("gravel_arctic"):
        if short_field < _GRAVEL_MIN_SHORT_FIELD and runway_ft > 4000:
            return HackV1Rejection(
                model,
                "gravel_runway_incompatible",
                "Gravel / remote strip mission requires STOL-capable aircraft — runway class incompatible.",
            )
        if category in {"ultra-long", "ultra_long", "large"} and short_field < _ARCTIC_MIN_SHORT_FIELD:
            return HackV1Rejection(
                model,
                "arctic_heavy_jet",
                "Arctic gravel strips — heavy jets lack field dispatch compatibility.",
            )

    if ctx.get("mountain") and category in {"ultra-long", "ultra_long"}:
        return HackV1Rejection(
            model,
            "mountain_ulr_conflict",
            "Mountain / hot-high access — ULR class cannot satisfy runway and performance envelope.",
        )

    # --- Hard range physics (NBAA / payload / wind) ---
    if required_nm > 0:
        baggage = "high" if (profile.baggage_priority or "").lower() in ("high", "medium") else "normal"
        available = compute_practical_range(
            spec,
            passengers=pax,
            baggage_weight=baggage,
            westbound=bool(ctx_eval.get("westbound")),
            nbaa_reserves=True,
            winter_westbound=bool(ctx_eval.get("winter_westbound_transatlantic")),
            mountain=bool(ctx_eval.get("mountain")),
        )
        if available < required_nm:
            return HackV1Rejection(
                model,
                "range_physics_hard_reject",
                (
                    f"Required ~{int(required_nm)} nm (NBAA reserves, payload, wind) exceeds "
                    f"practical available ~{int(available)} nm — hard reject, not conditional fit."
                ),
            )

    return None


def run_hack_v1_constraint_kernel(
    mission_profile: MissionProfile,
    candidate_models: Sequence[str],
    *,
    query: str = "",
    mission_state: Optional[MissionState] = None,
) -> HackV1Result:
    """
    Evaluate all candidates against hard aviation physics.

    Returns feasible_aircraft_list and rejection_log only (no ranking or verdict).
    """
    if mission_state is not None and not mission_profile.routes:
        mission_profile = mission_state_to_profile(mission_state)

    ctx = _mission_type_flags(mission_profile, query=query)
    models = list(dict.fromkeys(m for m in candidate_models if m))

    feasible: List[str] = []
    rejections: List[HackV1Rejection] = []

    for model in models:
        rejection = _evaluate_model(model, mission_profile, ctx)
        if rejection is not None:
            rejections.append(rejection)
        else:
            feasible.append(model)

    return HackV1Result(
        feasible_aircraft_list=feasible,
        rejection_log=rejections,
        constraint_empty=len(feasible) == 0 and len(models) > 0,
        mission_context=ctx,
    )


def apply_hack_v1_gate(
    mission_profile: MissionProfile,
    feasible_models: Sequence[str],
    *,
    all_candidates: Sequence[str],
    query: str = "",
    mission_state: Optional[MissionState] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], HackV1Result]:
    """
    Intersect upstream feasible list with HACK v1 survivors.

    Evaluates union of feasible + eliminated candidates so rejections are logged for
  all models that entered the pipeline.
    """
    universe = list(dict.fromkeys(list(feasible_models) + list(all_candidates)))
    result = run_hack_v1_constraint_kernel(
        mission_profile,
        universe,
        query=query,
        mission_state=mission_state,
    )
    passed = set(result.feasible_aircraft_list)
    if feasible_models:
        filtered = [m for m in feasible_models if m in passed]
    else:
        # Capability graph may return zero survivors while the kernel still finds
        # feasible aircraft in the evaluated candidate universe (e.g. field/STOL).
        filtered = list(result.feasible_aircraft_list)

    result.feasible_aircraft_list = filtered
    # Authoritative: no aircraft may proceed to ranking when the post-gate set is empty.
    result.constraint_empty = not bool(filtered)

    if isinstance(data_used, dict):
        attach_hack_v1_metadata(data_used, result)

    return filtered, result


def filter_models_by_hack_v1(
    models: Sequence[str],
    data_used: Optional[Dict[str, Any]],
) -> List[str]:
    """Remove permanently excluded models — used by tier recovery and rank entry."""
    exclusions = hack_v1_permanent_exclusions(data_used)
    if not exclusions:
        return list(models)
    return [m for m in models if m not in exclusions]


__all__ = [
    "HACK_V1_EMPTY_MESSAGE",
    "HACK_V1_METADATA_KEY",
    "HackV1Rejection",
    "HackV1Result",
    "apply_hack_v1_gate",
    "attach_hack_v1_metadata",
    "filter_models_by_hack_v1",
    "hack_v1_constraint_empty",
    "hack_v1_permanent_exclusions",
    "load_hack_v1_result",
    "run_hack_v1_constraint_kernel",
]
