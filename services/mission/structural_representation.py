"""
Structural proofs derived from pre-ranking representation — engine law before fleet/rank.

Not answer templating: sets packet flags + persisted proof for decomposition gates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from services.consultant.mission_state import MissionState
from services.mission.mission_understanding_engine import MissionUnderstandingPacket
from services.mission.models import MissionProfile, PriorityLevel
from services.mission.route_extractor import resolve_place
from services.mission.structural_decomposition import (
    STRUCTURAL_DECOMPOSITION_KEY,
    StructuralDecompositionProof,
    attach_structural_decomposition_proof,
)

STRUCTURAL_REPRESENTATION_KEY = "structural_representation"

_MOUNTAIN_CANON = frozenset({"aspen", "telluride", "jackson hole", "jackson"})
_ULR_CANON = frozenset(
    {"tokyo", "singapore", "dubai", "abu dhabi", "riyadh", "hong kong", "sydney"}
)
_TRANSATLANTIC_CANON = frozenset({"london", "zurich", "frankfurt", "paris", "europe"})


@dataclass
class StructuralRepresentationProof:
    required: bool = False
    proof_kind: str = ""
    reason: str = ""
    triggers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required": self.required,
            "proof_kind": self.proof_kind,
            "reason": self.reason,
            "triggers": list(self.triggers),
        }


def _endpoint_countries(profile: MissionProfile) -> Set[str]:
    countries: Set[str] = set()
    for r in profile.routes:
        for raw in (r.origin, r.destination):
            place, conf = resolve_place(raw)
            if place and conf >= 0.72 and place.country:
                countries.add(place.country)
    return countries


def evaluate_structural_representation(
    text: str,
    profile: MissionProfile,
    packet: Optional[MissionUnderstandingPacket],
    *,
    governance: Optional[Dict[str, Any]] = None,
    industrial: Optional[Dict[str, Any]] = None,
) -> StructuralRepresentationProof:
    """Derive structural proof from encoded mission state."""
    proof = StructuralRepresentationProof()
    tl = (text or "").lower()
    gov = governance or {}
    ind = industrial or {}

    # Founder / executive continuation vs company domestic scope
    founder_me = bool(
        re.search(
            r"\b(?:founder|chairman)\b.*\b(?:nonstop|flies?)\b.*\b"
            r"(?:abu\s+dhabi|dubai|singapore|riyadh|tokyo)\b",
            tl,
            re.I,
        )
        or re.search(
            r"\bnonstop\s+to\s+(?:abu\s+dhabi|dubai|singapore)\b.*\b(?:founder|chairman)\b",
            tl,
            re.I,
        )
    )
    company_na = bool(
        re.search(
            r"\b(?:rest of the company|never leaves north america|company never)\b",
            tl,
            re.I,
        )
        or (
            len(profile.routes) >= 3
            and any(
                "chicago" in lbl.lower() or "san francisco" in lbl.lower()
                for lbl in profile.route_labels()
            )
            and any(
                c in " ".join(profile.route_labels()).lower()
                for c in ("abu dhabi", "dubai", "singapore")
            )
        )
    )
    if founder_me and company_na:
        proof.required = True
        proof.proof_kind = "founder_company_asymmetry"
        proof.reason = (
            "Founder-exclusive ULR continuation vs company-wide domestic scope — "
            "single platform cannot span both governance envelopes."
        )
        proof.triggers.append("founder_company_asymmetry")

    # Mountain + ULR / transatlantic in same mission graph
    labels = " ".join(profile.route_labels()).lower()
    has_mountain = any(m in labels or m in tl for m in _MOUNTAIN_CANON)
    has_ulr = any(u in labels or u in tl for u in _ULR_CANON)
    has_transatlantic = any(t in labels or t in tl for t in _TRANSATLANTIC_CANON)
    if has_mountain and (has_ulr or has_transatlantic):
        proof.required = True
        proof.proof_kind = proof.proof_kind or "mountain_ulr_incompatibility"
        proof.reason = proof.reason or (
            "Mountain field-access and oceanic ULR/transatlantic stages cannot share one airframe."
        )
        proof.triggers.append("mountain_ulr_incompatibility")

    # Industrial field + transatlantic (reinforce)
    if ind.get("active") and (has_transatlantic or "london" in labels):
        proof.required = True
        proof.proof_kind = proof.proof_kind or "industrial_transatlantic_conflict"
        proof.reason = proof.reason or (
            "Industrial/remote field access and transatlantic executive nonstop are incompatible bands."
        )
        proof.triggers.append("industrial_transatlantic_conflict")

    # Multi-continent + cargo + variable pax → portfolio structural load
    countries = _endpoint_countries(profile)
    pax_var = bool(
        packet
        and packet.inferred_constraints.get("passenger_load_variable")
    )
    cargo = bool(
        (packet and packet.inferred_constraints.get("runway_over_cabin"))
        or (packet and packet.explicit_constraints.get("passenger_distribution", {}).get("cargo_required"))
        or ind.get("active")
    )
    if packet:
        dist = packet.explicit_constraints.get("passenger_distribution") or {}
        cargo = cargo or bool(dist.get("cargo_required"))
    if len(countries) >= 3 and (pax_var or cargo):
        proof.required = True
        proof.proof_kind = proof.proof_kind or "multi_continent_portfolio"
        proof.reason = proof.reason or (
            "Multi-continent mission with variable payload/cargo — fleet portfolio planning required."
        )
        proof.triggers.append("multi_continent_portfolio")

    # Governance utilization conflict from mission_governance layer
    if gov.get("utilization_mission_conflict"):
        proof.required = True
        proof.proof_kind = proof.proof_kind or "governance_utilization_conflict"
        proof.triggers.append("governance_utilization_conflict")

    # Leadership insists one aircraft + any structural trigger above
    if proof.required and (
        gov.get("single_aircraft_preference")
        or (packet and packet.inferred_constraints.get("single_aircraft_request"))
    ):
        proof.triggers.append("single_aircraft_request_denied")

    return proof


def apply_structural_representation(
    text: str,
    profile: MissionProfile,
    mission: MissionState,
    packet: Optional[MissionUnderstandingPacket],
    data_used: Optional[Dict[str, Any]] = None,
    *,
    governance: Optional[Dict[str, Any]] = None,
    industrial: Optional[Dict[str, Any]] = None,
) -> StructuralRepresentationProof:
    """Apply structural proof to packet + data_used for downstream decomposition."""
    rep = evaluate_structural_representation(
        text, profile, packet, governance=governance, industrial=industrial
    )
    du = data_used if isinstance(data_used, dict) else {}

    if packet is not None and rep.required:
        packet.inferred_constraints["incompatible_mission_bands"] = True
        packet.inferred_constraints["structural_representation_proof"] = rep.proof_kind
        if rep.proof_kind == "founder_company_asymmetry":
            packet.inferred_constraints["founder_company_asymmetry"] = True
        if rep.proof_kind == "multi_continent_portfolio":
            packet.inferred_constraints["multi_continent_portfolio"] = True
            packet.inferred_constraints["defer_global_shortlist"] = True
            packet.recommend_aircraft = False
            dist = packet.explicit_constraints.get("passenger_distribution") or {}
            if dist.get("cargo_required"):
                packet.inferred_constraints["cargo_over_cabin"] = True
                if profile.baggage_priority == PriorityLevel.NONE:
                    profile.baggage_priority = PriorityLevel.HIGH
            bands = list(packet.fallback_operational_band or [])
            for b in (
                "Multi-leg ultra-long-range executive band",
                "Transatlantic super-mid / heavy-cabin executive band",
                "Latin America / transatlantic executive band",
            ):
                if b not in bands:
                    bands.append(b)
            packet.fallback_operational_band = bands
        else:
            bands = list(packet.fallback_operational_band or [])
            additions = (
                "Mountain field-flexible short-strip band",
                "Multi-leg ultra-long-range executive band",
                "Domestic field-access executive band",
                "Transatlantic super-mid / heavy-cabin executive band",
            )
            for b in additions:
                if b not in bands:
                    bands.append(b)
            packet.fallback_operational_band = bands

    decomp = StructuralDecompositionProof(
        required=rep.required,
        reason=rep.reason,
        proof_kind=rep.proof_kind or "representation_proof",
    )
    attach_structural_decomposition_proof(du, decomp)
    du[STRUCTURAL_REPRESENTATION_KEY] = rep.to_dict()

    if rep.required:
        du["fleet_strategy_required"] = True
        mission_flags = du.get("mission_operational_graph")
        if isinstance(mission_flags, dict):
            mission_flags["fleet_strategy_required"] = True
            mission_flags["incompatible_bands"] = True

    return rep
