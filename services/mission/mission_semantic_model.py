"""
Mission Understanding Phase 2 — semantic intent stabilization.

Runs AFTER route extraction, BEFORE pre-ranking / geographic finalization / ranking.
Does NOT generate routes or recommend aircraft — only stabilizes what the mission means.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from services.consultant.mission_state import MissionState
from services.mission.models import MissionProfile

MISSION_SEMANTIC_MODEL_KEY = "mission_semantic_model"

# --- Domain identifiers (stable vocabulary for downstream gates) ---
DOMAIN_EXECUTIVE = "executive_transport"
DOMAIN_INDUSTRIAL = "industrial_field"
DOMAIN_ARCTIC = "arctic_operations"
DOMAIN_MINING = "mining_logistics"
DOMAIN_MOUNTAIN = "mountain_leisure"
DOMAIN_CARIBBEAN = "caribbean_regional"
DOMAIN_ULR = "ulr_continuation"
DOMAIN_DOMESTIC = "domestic_utilization"

_DOMAIN_BASE_WEIGHTS: Dict[str, float] = {
    DOMAIN_EXECUTIVE: 0.82,
    DOMAIN_INDUSTRIAL: 0.92,
    DOMAIN_ARCTIC: 0.98,
    DOMAIN_MINING: 0.88,
    DOMAIN_MOUNTAIN: 0.38,
    DOMAIN_CARIBBEAN: 0.55,
    DOMAIN_ULR: 0.78,
    DOMAIN_DOMESTIC: 0.62,
}

_HARD_DOMAINS = frozenset({DOMAIN_ARCTIC, DOMAIN_INDUSTRIAL, DOMAIN_MINING})

# --- Signal patterns ---
_ARCTIC_RE = re.compile(
    r"\b(?:arctic|nunavut|yellowknife|northern\s+canada|northern\s+alberta|"
    r"gravel\s+strips?|ice\s+strip|winter\s+dispatch|polar\s+headwind)\b",
    re.I,
)
_INDUSTRIAL_RE = re.compile(
    r"\b(?:oil\s+fields?|drilling\s+sites?|remote\s+drilling|offshore\s+rigs?|"
    r"permian|desert\s+energy|industrial\s+access|field\s+access|industrial|unpaved)\b",
    re.I,
)
_MINING_RE = re.compile(
    r"\b(?:mining|pilbara|extraction\s+strips?|australian\s+extraction|"
    r"west\s+african\s+mining|resource\s+strips?)\b",
    re.I,
)
_EXECUTIVE_RE = re.compile(
    r"\b(?:executives?|leadership|chairman|ceo|founder|principal|nonstop|"
    r"transatlantic|london|paris|geneva|frankfurt|zurich)\b",
    re.I,
)
_MOUNTAIN_LEISURE_RE = re.compile(
    r"\b(?:ski\s+regions?|ski\s+access|aspen|vail|telluride|jackson\s+hole|banff|"
    r"winter\s+resort)\b",
    re.I,
)
_CARIBBEAN_RE = re.compile(
    r"\b(?:caribbean|bahamas|turks\s*&\s*caicos|tropical\s+islands?|island\s+hops?)\b",
    re.I,
)
_ULR_RE = re.compile(
    r"\b(?:dubai|abu\s+dhabi|riyadh|doha|singapore|tokyo|hong\s+kong|seoul|ulr|"
    r"ultra[- ]?long|nonstop\s+(?:dubai|singapore|tokyo|riyadh))\b",
    re.I,
)
_DOMESTIC_UTIL_RE = re.compile(
    r"\b(?:domestic|shuttle|short\s+hops?|east\s+coast\s+corridor|"
    r"\d{1,3}\s*%\s+of\s+(?:our\s+)?flying|boston|washington|chicago)\b",
    re.I,
)
_SINGLE_AIRCRAFT_RE = re.compile(
    r"\b(?:one\s+aircraft|single\s+aircraft|only\s+one\s+jet|ideally\s+one)\b",
    re.I,
)
_WINTER_FAILURE_RE = re.compile(
    r"\b(?:winter\s+dispatch|dispatch\s+failures?|range\s+failures?|"
    r"headwind|polar)\b",
    re.I,
)
_SKI_CRITICAL_RE = re.compile(
    r"\b(?:ski\s+is\s+critical|must\s+reach\s+aspen|nonstop\s+aspen|"
    r"ski\s+mandate|winter\s+ops\s+priority)\b",
    re.I,
)
_FOUNDER_ULR_RE = re.compile(
    r"\b(?:founder|chairman|ceo|principal)\b.*\b(?:nonstop|requires?|insists?)\b.*"
    r"\b(?:singapore|dubai|tokyo|riyadh|abu\s+dhabi)\b"
    r"|\b(?:nonstop|requires?)\b.*\b(?:founder|chairman)\b",
    re.I,
)
_CEO_ULR_MANDATE_RE = re.compile(
    r"\b(?:ceo|founder|chairman|principal)\b.*\b(?:demands?|mandate|requires?|insists?)\b.*"
    r"\b(?:dubai|singapore|tokyo|riyadh|abu\s+dhabi)\b.*\b(?:nonstop|capability)\b"
    r"|\b(?:dubai|singapore|tokyo|riyadh|abu\s+dhabi)\b.*\b(?:nonstop|capability)\b.*\bfrom\s+new\s+york\b",
    re.I,
)

_CONTINUATION_HUB_AMBIGUITY_RE = re.compile(
    r"\bnew\s+york\b.*\bsingapore\b.*\bvia\s+dubai\b|\bsingapore\b.*\bvia\s+dubai\b",
    re.I,
)


@dataclass
class MissionSemanticModel:
    mission_domains: List[str] = field(default_factory=list)
    domain_weights: List[float] = field(default_factory=list)
    operational_priority_order: List[str] = field(default_factory=list)
    constraint_flags: List[str] = field(default_factory=list)
    continuation_constraints: List[str] = field(default_factory=list)
    invalid_interpretations: List[str] = field(default_factory=list)
    reasoning_clarity_notes: List[str] = field(default_factory=list)
    hard_domains: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_domains": list(self.mission_domains),
            "domain_weights": [round(w, 3) for w in self.domain_weights],
            "operational_priority_order": list(self.operational_priority_order),
            "constraint_flags": list(self.constraint_flags),
            "continuation_constraints": list(self.continuation_constraints),
            "invalid_interpretations": list(self.invalid_interpretations),
            "reasoning_clarity_notes": list(self.reasoning_clarity_notes),
            "hard_domains": list(self.hard_domains),
        }


def _detect_domains(
    text: str,
    profile: MissionProfile,
    mission: MissionState,
) -> List[Tuple[str, float, bool, str]]:
    """
    Return (domain, weight, hard_domain, constraint_multiplier_label) tuples.
    """
    tl = text or ""
    blob = f"{tl} {' '.join(profile.route_labels())}".lower()
    detected: List[Tuple[str, float, bool, str]] = []

    def _add(domain: str, *, boost: float = 0.0, hard: bool = False, mult: str = "standard") -> None:
        base = _DOMAIN_BASE_WEIGHTS.get(domain, 0.5)
        weight = min(1.0, base + boost)
        if any(d[0] == domain for d in detected):
            return
        detected.append((domain, weight, hard, mult))

    if _ARCTIC_RE.search(tl) or _ARCTIC_RE.search(blob):
        boost = 0.05 if _WINTER_FAILURE_RE.search(tl) else 0.0
        _add(DOMAIN_ARCTIC, boost=boost, hard=True, mult="extreme")

    if _INDUSTRIAL_RE.search(tl) or _INDUSTRIAL_RE.search(blob):
        _add(DOMAIN_INDUSTRIAL, hard=True, mult="high")

    if _MINING_RE.search(tl):
        _add(DOMAIN_MINING, hard=True, mult="high")

    if _EXECUTIVE_RE.search(tl) or profile.nonstop_required:
        _add(DOMAIN_EXECUTIVE, boost=0.05 if profile.nonstop_required else 0.0)

    if _ULR_RE.search(tl) or profile.international_ops:
        _add(DOMAIN_ULR)

    if _DOMESTIC_UTIL_RE.search(tl):
        _add(DOMAIN_DOMESTIC, boost=0.08)

    if _MOUNTAIN_LEISURE_RE.search(tl):
        weight_boost = 0.35 if _SKI_CRITICAL_RE.search(tl) else 0.0
        _add(DOMAIN_MOUNTAIN, boost=weight_boost, hard=_SKI_CRITICAL_RE.search(tl) is not None)

    if _CARIBBEAN_RE.search(tl):
        _add(DOMAIN_CARIBBEAN)

    if not detected and profile.routes:
        _add(DOMAIN_EXECUTIVE)

    return detected


def _build_constraint_flags(detected: Sequence[Tuple[str, float, bool, str]], text: str) -> List[str]:
    flags: List[str] = []
    domains = {d[0] for d in detected}
    mults = {d[0]: d[3] for d in detected}

    if DOMAIN_ARCTIC in domains:
        flags.extend(
            [
                "arctic_hard_domain",
                "arctic_extreme_constraint_multiplier",
                "runway_performance_binding",
            ]
        )
        if _WINTER_FAILURE_RE.search(text or ""):
            flags.append("winter_dispatch_binding")

    if DOMAIN_INDUSTRIAL in domains:
        flags.extend(
            [
                "industrial_hard_domain",
                "industrial_high_constraint_weight",
                "field_access_over_cabin",
            ]
        )

    if DOMAIN_MINING in domains:
        flags.extend(["mining_hard_domain", "industrial_high_constraint_weight"])

    if DOMAIN_ARCTIC in domains and mults.get(DOMAIN_ARCTIC) == "extreme":
        flags.append("constraint_multiplier_extreme")

    if DOMAIN_INDUSTRIAL in domains or DOMAIN_MINING in domains:
        flags.append("constraint_multiplier_high")

    if DOMAIN_EXECUTIVE in domains and (DOMAIN_ARCTIC in domains or DOMAIN_INDUSTRIAL in domains):
        flags.append("executive_and_field_dual_posture")

    if len([d for d in detected if d[2]]) >= 2:
        flags.append("multi_hard_domain_mission")

    return list(dict.fromkeys(flags))


def _build_continuation_constraints(text: str, detected: Sequence[Tuple[str, float, bool, str]]) -> List[str]:
    constraints: List[str] = []
    domains = {d[0] for d in detected}

    if _FOUNDER_ULR_RE.search(text or "") or _CEO_ULR_MANDATE_RE.search(text or ""):
        constraints.append("ulr_continuation_requires_mandate_hub_origin")

    if DOMAIN_ULR in domains and DOMAIN_DOMESTIC in domains:
        constraints.append("domestic_utilization_dominates_except_founder_ulr")

    if DOMAIN_INDUSTRIAL in domains or DOMAIN_ARCTIC in domains:
        constraints.append("field_ops_not_subordinate_to_executive_cabin_priority")

    if DOMAIN_CARIBBEAN in domains and DOMAIN_ULR in domains:
        constraints.append("caribbean_and_me_continuation_require_segment_split")

    if DOMAIN_ULR in domains and _CONTINUATION_HUB_AMBIGUITY_RE.search(text or ""):
        constraints.append("continuation_hubs_semantic_only_not_primary_origin")

    return constraints


def _build_invalid_interpretations(
    text: str,
    detected: Sequence[Tuple[str, float, bool, str]],
) -> List[str]:
    invalid: List[str] = []
    domains = {d[0] for d in detected}
    hard = [d[0] for d in detected if d[2]]

    if DOMAIN_ARCTIC in domains and DOMAIN_EXECUTIVE in domains:
        invalid.append("single_ulr_covers_arctic_gravel_and_transatlantic_executive")

    if len(hard) >= 2:
        invalid.append("single_aircraft_universal_hard_domain_coverage")

    if _SINGLE_AIRCRAFT_RE.search(text or "") and len(hard) >= 1:
        invalid.append("single_aircraft_preference_over_hard_domain_conflict")

    if DOMAIN_ARCTIC in domains:
        invalid.append("brochure_range_on_gravel_or_ice_strips")

    if DOMAIN_INDUSTRIAL in domains and DOMAIN_ULR in domains:
        invalid.append("executive_ulr_subordinates_industrial_field_access")

    if DOMAIN_MOUNTAIN in domains and DOMAIN_ULR in domains and DOMAIN_ARCTIC not in domains:
        weight_map = {d[0]: d[1] for d in detected}
        if weight_map.get(DOMAIN_MOUNTAIN, 0) < weight_map.get(DOMAIN_ULR, 0):
            invalid.append("mountain_leisure_equated_with_ulr_mandate")

    return list(dict.fromkeys(invalid))


def _build_reasoning_notes(
    detected: Sequence[Tuple[str, float, bool, str]],
    invalid: Sequence[str],
) -> List[str]:
    notes: List[str] = []
    hard = [d[0] for d in detected if d[2]]

    if DOMAIN_ARCTIC in hard:
        notes.append(
            "Arctic / gravel operations bind runway and winter dispatch before cabin or brochure range."
        )
    if DOMAIN_INDUSTRIAL in hard or DOMAIN_MINING in hard:
        notes.append(
            "Industrial or mining field access is a hard operational constraint — not a soft preference."
        )
    if len(hard) >= 2:
        notes.append(
            "Multiple hard domains detected — mission semantics require segmented structure, not one universal jet."
        )
    if "single_aircraft_preference_over_hard_domain_conflict" in invalid:
        notes.append(
            "Stated single-aircraft preference conflicts with hard domain constraints; decomposition is semantically required."
        )
    if not notes and detected:
        notes.append(
            f"Operational priority order: {', '.join(d[0].replace('_', ' ') for d in sorted(detected, key=lambda x: -x[1])[:3])}."
        )
    return notes


def build_mission_semantic_model(
    query: str,
    profile: MissionProfile,
    mission: MissionState,
) -> MissionSemanticModel:
    """Derive explicit semantic mission model from extracted inputs — no route generation."""
    detected = _detect_domains(query, profile, mission)
    detected_sorted = sorted(detected, key=lambda x: (-x[1], x[0]))

    model = MissionSemanticModel(
        mission_domains=[d[0] for d in detected_sorted],
        domain_weights=[d[1] for d in detected_sorted],
        operational_priority_order=[d[0] for d in detected_sorted],
        hard_domains=[d[0] for d in detected_sorted if d[2]],
    )

    model.constraint_flags = _build_constraint_flags(detected, query)
    model.continuation_constraints = _build_continuation_constraints(query, detected)
    model.invalid_interpretations = _build_invalid_interpretations(query, detected)
    model.reasoning_clarity_notes = _build_reasoning_notes(detected, model.invalid_interpretations)

    return model


def apply_semantic_model_to_packet(
    packet: Any,
    model: MissionSemanticModel,
    *,
    profile: Optional[MissionProfile] = None,
) -> None:
    """Merge semantic model into mission understanding packet — semantics only."""
    inf = packet.inferred_constraints
    inf["mission_semantic_domains"] = list(model.mission_domains)
    inf["mission_domain_weights"] = dict(zip(model.mission_domains, model.domain_weights))
    inf["operational_priority_order"] = list(model.operational_priority_order)
    inf["semantic_invalid_interpretations"] = list(model.invalid_interpretations)
    inf["hard_domains"] = list(model.hard_domains)

    # Continuation constraints become explicit semantic flags
    for c in model.continuation_constraints:
        inf[c] = True

    for flag in model.constraint_flags:
        inf[flag] = True

    if "arctic_hard_domain" in model.constraint_flags:
        packet.runway_complexity = "mountain"
        inf["industrial_airport_access"] = True
        inf["mountain_ops"] = True

    if "industrial_hard_domain" in model.constraint_flags or "mining_hard_domain" in model.constraint_flags:
        inf["industrial_airport_access"] = True
        if profile is not None and profile.runway_priority.value == "none":
            from services.mission.models import PriorityLevel

            profile.runway_priority = PriorityLevel.HIGH

    if "executive_and_field_dual_posture" in model.constraint_flags:
        inf["dual_use_or_multi_leg"] = True
        inf["incompatible_mission_bands"] = True

    if "multi_hard_domain_mission" in model.constraint_flags:
        inf["defer_global_shortlist"] = True
        inf.setdefault(
            "defer_global_shortlist_reason",
            "Multiple hard operational domains — semantic model blocks universal single-aircraft interpretation.",
        )

    for note in model.reasoning_clarity_notes:
        if note not in packet.understanding_notes:
            packet.understanding_notes.append(note)

    env_map = {
        DOMAIN_ARCTIC: "arctic_field",
        DOMAIN_INDUSTRIAL: "industrial_field",
        DOMAIN_MINING: "mining_logistics",
        DOMAIN_MOUNTAIN: "mountain_leisure",
        DOMAIN_CARIBBEAN: "caribbean_regional",
        DOMAIN_ULR: "ulr_continuation",
        DOMAIN_DOMESTIC: "domestic_utilization",
        DOMAIN_EXECUTIVE: "executive_transport",
    }
    for domain in model.operational_priority_order[:4]:
        tag = env_map.get(domain)
        if tag and tag not in packet.operational_environment:
            packet.operational_environment.append(tag)


def stabilize_mission_semantics(
    query: str,
    profile: MissionProfile,
    mission: MissionState,
    packet: Optional[Any] = None,
    *,
    data_used: Optional[Dict[str, Any]] = None,
) -> MissionSemanticModel:
    """Build and attach semantic model — entry point for pipeline integration."""
    model = build_mission_semantic_model(query, profile, mission)
    if packet is not None:
        apply_semantic_model_to_packet(packet, model, profile=profile)
    if isinstance(data_used, dict):
        data_used[MISSION_SEMANTIC_MODEL_KEY] = model.to_dict()
    return model


__all__ = [
    "MISSION_SEMANTIC_MODEL_KEY",
    "MissionSemanticModel",
    "DOMAIN_ARCTIC",
    "DOMAIN_EXECUTIVE",
    "DOMAIN_INDUSTRIAL",
    "DOMAIN_MINING",
    "apply_semantic_model_to_packet",
    "build_mission_semantic_model",
    "stabilize_mission_semantics",
]
