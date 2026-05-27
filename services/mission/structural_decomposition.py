"""
Structural fleet decomposition gate — hard operational impossibility, not mild multi-leg overlap.

Domestic + occasional Europe is dual-use, not structural decomposition.
Aspen + Dubai nonstop, or ULR + short-field, is structural.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from services.consultant.mission_state import MissionState
from services.mission.mission_understanding_engine import (
    MissionUnderstandingPacket,
    bands_are_incompatible,
)
from services.mission.models import MissionProfile

STRUCTURAL_DECOMPOSITION_KEY = "structural_decomposition"


@dataclass
class StructuralDecompositionProof:
    required: bool
    reason: str = ""
    proof_kind: str = ""  # incompatible_bands | elimination_failure | domain_impossibility

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required": self.required,
            "reason": self.reason,
            "proof_kind": self.proof_kind,
        }


def needs_structural_decomposition(
    packet: Optional[MissionUnderstandingPacket],
    *,
    profile: Optional[MissionProfile] = None,
    mission: Optional[MissionState] = None,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
    feasible_models: Optional[Sequence[str]] = None,
) -> StructuralDecompositionProof:
    """
    True only when one aircraft cannot credibly span domains (hard proof).

    Does NOT trigger on dual_use_or_multi_leg, mild range spread, or continuation alone.
    """
    if packet is None:
        return StructuralDecompositionProof(required=False)

    if isinstance(data_used, dict):
        raw = data_used.get("structural_representation")
        if isinstance(raw, dict) and raw.get("required"):
            return StructuralDecompositionProof(
                required=True,
                reason=str(raw.get("reason") or "Pre-ranking structural representation proof."),
                proof_kind=str(raw.get("proof_kind") or "representation_proof"),
            )

    bands = list(packet.fallback_operational_band or [])
    if packet.inferred_constraints.get("incompatible_mission_bands"):
        return StructuralDecompositionProof(
            required=True,
            reason="Incompatible operational bands — ULR/oceanic and short-field domains conflict.",
            proof_kind="incompatible_bands",
        )
    if bands_are_incompatible(bands):
        return StructuralDecompositionProof(
            required=True,
            reason="Band proof: ultra-long-range and short-field requirements cannot share one platform.",
            proof_kind="incompatible_bands",
        )

    if profile is not None and mission is not None:
        try:
            from services.fleet.fleet_domain_analysis import (
                analyze_multi_domain_operational_problem,
            )
            from services.mission.aircraft_profiles import AIRCRAFT_PROFILES

            pool = list(feasible_models or AIRCRAFT_PROFILES.keys())
            analysis = analyze_multi_domain_operational_problem(
                profile,
                mission,
                pool,
                query=query,
                data_used=data_used,
            )
            if len(analysis.domains) < 2:
                return StructuralDecompositionProof(required=False)
            if (
                analysis.multi_domain_required
                and analysis.single_aircraft_structurally_invalid
                and not analysis.universal_survivors
            ):
                return StructuralDecompositionProof(
                    required=True,
                    reason=analysis.doctrine or "No universal survivor across operational domains.",
                    proof_kind="elimination_failure",
                )
        except Exception:
            pass

    return StructuralDecompositionProof(required=False)


def attach_structural_decomposition_proof(
    data_used: Optional[Dict[str, Any]],
    proof: StructuralDecompositionProof,
) -> None:
    if isinstance(data_used, dict):
        data_used[STRUCTURAL_DECOMPOSITION_KEY] = proof.to_dict()


__all__ = [
    "STRUCTURAL_DECOMPOSITION_KEY",
    "StructuralDecompositionProof",
    "attach_structural_decomposition_proof",
    "needs_structural_decomposition",
]
