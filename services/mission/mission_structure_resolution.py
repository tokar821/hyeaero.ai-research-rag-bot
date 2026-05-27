"""
Single structural verdict authority — decomposition and rendering stay synchronized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from services.mission.mission_graph import MissionGraph
from services.mission.segment_kinds import SegmentKind
from services.mission.mission_understanding_engine import MissionUnderstandingPacket
from services.mission.structural_decomposition import (
    StructuralDecompositionProof,
    needs_structural_decomposition,
)

MISSION_STRUCTURE_RESOLUTION_KEY = "mission_structure_resolution"

_SINGLE_DOMAIN_PHRASE = "single operational domain"
_IN_BAND_PHRASE = "standard in-band selection"


@dataclass
class MissionStructureResolution:
    structurally_single_aircraft_valid: bool
    decomposition_required: bool
    decomposition_reason: str
    incompatible_domains: List[str]
    proof_source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "structurally_single_aircraft_valid": self.structurally_single_aircraft_valid,
            "decomposition_required": self.decomposition_required,
            "decomposition_reason": self.decomposition_reason,
            "incompatible_domains": list(self.incompatible_domains),
            "proof_source": self.proof_source,
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> Optional["MissionStructureResolution"]:
        if not isinstance(raw, dict):
            return None
        return cls(
            structurally_single_aircraft_valid=bool(
                raw.get("structurally_single_aircraft_valid")
            ),
            decomposition_required=bool(raw.get("decomposition_required")),
            decomposition_reason=str(raw.get("decomposition_reason") or ""),
            incompatible_domains=[
                str(d) for d in (raw.get("incompatible_domains") or []) if d
            ],
            proof_source=str(raw.get("proof_source") or ""),
        )


def _incompatible_domain_labels(graph: MissionGraph) -> List[str]:
    kinds = {s.kind for s in graph.segments}
    labels: List[str] = []
    if SegmentKind.PACIFIC_ULR in kinds or SegmentKind.ULR_CONTINUATION in kinds:
        labels.append("ulr_continuation")
    if SegmentKind.MOUNTAIN_FIELD in kinds:
        labels.append("mountain_field")
    if SegmentKind.INDUSTRIAL_FIELD in kinds:
        labels.append("industrial_field")
    if SegmentKind.TRANSATLANTIC_EXECUTIVE in kinds and SegmentKind.DOMESTIC_EXECUTIVE in kinds:
        if "domestic_executive" not in labels:
            labels.append("domestic_executive")
    return labels


def resolve_mission_structure(
    packet: Optional[MissionUnderstandingPacket],
    graph: MissionGraph,
    proof: StructuralDecompositionProof,
    *,
    profile=None,
    mission=None,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> MissionStructureResolution:
    """One verdict object for kernel, fleet doctrine, and recommendation suppression."""
    if proof.required:
        return MissionStructureResolution(
            structurally_single_aircraft_valid=False,
            decomposition_required=True,
            decomposition_reason=proof.reason or "Structural decomposition proof.",
            incompatible_domains=_incompatible_domain_labels(graph),
            proof_source=proof.proof_kind or "structural_decomposition",
        )

    if graph.structural_incompatibility and len(graph.segments) >= 2:
        kinds = {s.kind for s in graph.segments}
        if len(kinds) >= 2:
            return MissionStructureResolution(
                structurally_single_aircraft_valid=False,
                decomposition_required=True,
                decomposition_reason="Multiple incompatible operational segments in mission graph.",
                incompatible_domains=_incompatible_domain_labels(graph),
                proof_source="mission_graph",
            )

    if packet and packet.inferred_constraints.get("incompatible_mission_bands"):
        return MissionStructureResolution(
            structurally_single_aircraft_valid=False,
            decomposition_required=True,
            decomposition_reason="Incompatible operational bands on mission packet.",
            incompatible_domains=_incompatible_domain_labels(graph),
            proof_source="incompatible_bands",
        )

    if packet and packet.inferred_constraints.get("defer_global_shortlist"):
        return MissionStructureResolution(
            structurally_single_aircraft_valid=False,
            decomposition_required=True,
            decomposition_reason=str(
                packet.inferred_constraints.get("defer_global_shortlist_reason")
                or "Governance or portfolio structure defers single-aircraft selection."
            ),
            incompatible_domains=_incompatible_domain_labels(graph),
            proof_source="governance",
        )

    return MissionStructureResolution(
        structurally_single_aircraft_valid=True,
        decomposition_required=False,
        decomposition_reason="",
        incompatible_domains=[],
        proof_source="none",
    )


def build_mission_structure_resolution(
    packet: Optional[MissionUnderstandingPacket],
    graph: MissionGraph,
    *,
    profile=None,
    mission=None,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
    feasible_models=None,
) -> MissionStructureResolution:
    proof = needs_structural_decomposition(
        packet,
        profile=profile,
        mission=mission,
        query=query,
        data_used=data_used,
        feasible_models=feasible_models,
    )
    return resolve_mission_structure(
        packet,
        graph,
        proof,
        profile=profile,
        mission=mission,
        query=query,
        data_used=data_used,
    )


def forbid_single_domain_doctrine(resolution: MissionStructureResolution) -> bool:
    return resolution.decomposition_required


def sanitize_doctrine_text(
    doctrine: str,
    resolution: MissionStructureResolution,
) -> str:
    """Strip contradictory single-domain phrases when decomposition is required."""
    if not forbid_single_domain_doctrine(resolution):
        return (doctrine or "").strip()
    text = doctrine or ""
    if not text:
        return (
            "Multi-domain operational problem — mission decomposition precedes model selection. "
            f"{resolution.decomposition_reason}"
        ).strip()
    lower = text.lower()
    if _SINGLE_DOMAIN_PHRASE in lower or _IN_BAND_PHRASE in lower:
        return (
            "Multi-domain operational problem — one aircraft cannot span these operational domains. "
            f"{resolution.decomposition_reason or 'See segment structure above.'}"
        ).strip()
    return text.strip()


def save_structure_resolution(
    data_used: Optional[Dict[str, Any]],
    resolution: MissionStructureResolution,
) -> None:
    if isinstance(data_used, dict):
        data_used[MISSION_STRUCTURE_RESOLUTION_KEY] = resolution.to_dict()


def load_structure_resolution(
    data_used: Optional[Dict[str, Any]],
) -> Optional[MissionStructureResolution]:
    if not isinstance(data_used, dict):
        return None
    return MissionStructureResolution.from_dict(
        data_used.get(MISSION_STRUCTURE_RESOLUTION_KEY)
    )


__all__ = [
    "MISSION_STRUCTURE_RESOLUTION_KEY",
    "MissionStructureResolution",
    "build_mission_structure_resolution",
    "forbid_single_domain_doctrine",
    "load_structure_resolution",
    "resolve_mission_structure",
    "sanitize_doctrine_text",
    "save_structure_resolution",
]
