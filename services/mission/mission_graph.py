"""
Segmented mission representation — operational bands and constraints are local to segments.

Global MissionState remains for legacy feasibility; narrative and fleet doctrine consume MissionGraph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from services.consultant.mission_state import MissionState
from services.mission.mission_understanding_engine import MissionUnderstandingPacket
from services.mission.models import MissionProfile
from services.mission.segment_kinds import SegmentKind

MISSION_GRAPH_KEY = "mission_segment_graph"

_BAND_ULR_RE = re.compile(r"ultra-long|transatlantic.*ulr|multi-leg ultra", re.I)
_BAND_CONTINUATION_RE = re.compile(r"continuation|middle east", re.I)
_BAND_SHORT_FIELD_RE = re.compile(
    r"mountain|short-runway|short-strip|field-access|domestic field|caribbean.*regional",
    re.I,
)
_BAND_TRANSATLANTIC_RE = re.compile(r"transatlantic|heavy-cabin executive", re.I)
_BAND_CARIBBEAN_RE = re.compile(r"caribbean", re.I)
_ME_RE = re.compile(r"\b(?:dubai|riyadh|doha|abu\s+dhabi|jeddah|middle\s+east)\b", re.I)
_MOUNTAIN_ROUTE_RE = re.compile(
    r"\b(?:aspen|kase|ktex|kege|jackson|telluride|sun\s+valley)\b",
    re.I,
)
_PACIFIC_ANCHOR_RE = re.compile(
    r"\b(?:tokyo|osaka|seoul|hong\s+kong|singapore|sydney|beijing|shanghai|"
    r"manila|auckland|los\s+angeles|san\s+francisco|vancouver)\b",
    re.I,
)
_INDUSTRIAL_ROUTE_RE = re.compile(
    r"\b(?:remote\s+drilling|arctic\s+oil|west\s+africa|oil\s+platform|mining|gravel)\b",
    re.I,
)


@dataclass
class MissionSegmentProfile:
    segment_id: str
    kind: SegmentKind
    label: str
    operational_band: str
    route_labels: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    is_peak_planning: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "kind": self.kind.value,
            "label": self.label,
            "operational_band": self.operational_band,
            "route_labels": list(self.route_labels),
            "constraints": dict(self.constraints),
            "is_peak_planning": self.is_peak_planning,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "MissionSegmentProfile":
        try:
            kind = SegmentKind(str(raw.get("kind") or SegmentKind.DOMESTIC_EXECUTIVE.value))
        except ValueError:
            kind = SegmentKind.DOMESTIC_EXECUTIVE
        return cls(
            segment_id=str(raw.get("segment_id") or "segment_a"),
            kind=kind,
            label=str(raw.get("label") or ""),
            operational_band=str(raw.get("operational_band") or ""),
            route_labels=[str(r) for r in (raw.get("route_labels") or []) if r],
            constraints=dict(raw.get("constraints") or {}),
            is_peak_planning=bool(raw.get("is_peak_planning")),
        )


@dataclass
class MissionGraph:
    segments: List[MissionSegmentProfile] = field(default_factory=list)
    peak_segment_id: str = ""
    structural_incompatibility: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "peak_segment_id": self.peak_segment_id,
            "structural_incompatibility": self.structural_incompatibility,
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "MissionGraph":
        if not isinstance(raw, dict):
            return cls()
        segs = [
            MissionSegmentProfile.from_dict(s)
            for s in (raw.get("segments") or [])
            if isinstance(s, dict)
        ]
        return cls(
            segments=segs,
            peak_segment_id=str(raw.get("peak_segment_id") or ""),
            structural_incompatibility=bool(raw.get("structural_incompatibility")),
        )


def _band_kind_from_label(band: str) -> SegmentKind:
    b = band or ""
    if _BAND_CONTINUATION_RE.search(b) and _BAND_ULR_RE.search(b):
        return SegmentKind.ULR_CONTINUATION
    if _BAND_ULR_RE.search(b) and _BAND_CONTINUATION_RE.search(b):
        return SegmentKind.ULR_CONTINUATION
    if _BAND_CONTINUATION_RE.search(b):
        return SegmentKind.ULR_CONTINUATION
    if _BAND_ULR_RE.search(b) or "multi-leg ultra" in b.lower():
        return SegmentKind.PACIFIC_ULR
    if _BAND_SHORT_FIELD_RE.search(b) and "mountain" in b.lower():
        return SegmentKind.MOUNTAIN_FIELD
    if _BAND_SHORT_FIELD_RE.search(b) and "field-access" in b.lower():
        return SegmentKind.INDUSTRIAL_FIELD
    if _BAND_CARIBBEAN_RE.search(b):
        return SegmentKind.CARIBBEAN_REGIONAL
    if _BAND_TRANSATLANTIC_RE.search(b):
        return SegmentKind.TRANSATLANTIC_EXECUTIVE
    return SegmentKind.DOMESTIC_EXECUTIVE


def _route_segment_kind(route_label: str, stage_nm: float) -> SegmentKind:
    lbl = route_label or ""
    if _ME_RE.search(lbl):
        return SegmentKind.ULR_CONTINUATION
    if _MOUNTAIN_ROUTE_RE.search(lbl):
        return SegmentKind.MOUNTAIN_FIELD
    if _INDUSTRIAL_ROUTE_RE.search(lbl):
        return SegmentKind.INDUSTRIAL_FIELD
    if stage_nm >= 4800 and _PACIFIC_ANCHOR_RE.search(lbl):
        return SegmentKind.PACIFIC_ULR
    if stage_nm >= 2800:
        return SegmentKind.TRANSATLANTIC_EXECUTIVE
    if _BAND_CARIBBEAN_RE.search(lbl):
        return SegmentKind.CARIBBEAN_REGIONAL
    return SegmentKind.DOMESTIC_EXECUTIVE


def _estimate_route_nm(label: str) -> float:
    try:
        from services.consultant.route_feasibility import estimate_route_distance_nm

        return float(estimate_route_distance_nm(label) or 0)
    except Exception:
        return 0.0


def _attach_bands_to_existing_segments(
    segments_by_kind: Dict[SegmentKind, MissionSegmentProfile],
    bands: Sequence[str],
    *,
    query: str = "",
) -> None:
    """Bands may enrich route-born segments only — never create ghost segments."""
    from services.mission.operational_band_validator import validate_operational_band

    for band in bands:
        kind = _band_kind_from_label(band)
        seg = segments_by_kind.get(kind)
        if seg is None:
            continue
        validation = validate_operational_band(
            band,
            route_labels=seg.route_labels,
            query=query,
        )
        if not validation.renderable:
            continue
        if band not in (seg.operational_band or ""):
            seg.operational_band = (
                band
                if not seg.operational_band
                else f"{seg.operational_band}; {band}"
            )


def build_mission_graph(
    packet: MissionUnderstandingPacket,
    profile: MissionProfile,
    mission: MissionState,
    *,
    structural_incompatibility: bool = False,
    query: str = "",
) -> MissionGraph:
    """
    Route-first segment graph — bands and constraints attach only to proven segments.
    """
    bands = list(dict.fromkeys(packet.fallback_operational_band or []))
    routes = list(profile.route_labels() or mission.routes or [])
    route_nm = [(lbl, _estimate_route_nm(lbl)) for lbl in routes]

    segments_by_kind: Dict[SegmentKind, MissionSegmentProfile] = {}

    for lbl, nm in route_nm:
        kind = _route_segment_kind(lbl, nm)
        if kind not in segments_by_kind:
            segments_by_kind[kind] = MissionSegmentProfile(
                segment_id=kind.value,
                kind=kind,
                label=kind.value.replace("_", " ").title(),
                operational_band="",
                route_labels=[],
            )
        seg = segments_by_kind[kind]
        if lbl not in seg.route_labels:
            seg.route_labels.append(lbl)

    _attach_bands_to_existing_segments(segments_by_kind, bands, query=query)

    cargo_required = False
    dist_raw = packet.explicit_constraints.get("passenger_distribution")
    if isinstance(dist_raw, dict):
        cargo_required = bool(dist_raw.get("cargo_required"))

    from services.mission.segment_authority import infer_domain_constraints_for_segment

    for seg in segments_by_kind.values():
        scoped = infer_domain_constraints_for_segment(seg, cargo_required=cargo_required)
        seg.constraints = scoped.to_dict()

    segments = [s for s in segments_by_kind.values() if s.route_labels]

    peak_id = ""
    peak_nm = 0.0
    for seg in segments:
        seg_nm = max((_estimate_route_nm(r) for r in seg.route_labels), default=0.0)
        if seg.kind == SegmentKind.ULR_CONTINUATION and seg.route_labels:
            if seg_nm >= peak_nm:
                peak_nm = seg_nm
                peak_id = seg.segment_id
        elif seg_nm > peak_nm:
            peak_nm = seg_nm
            peak_id = seg.segment_id
    for seg in segments:
        seg.is_peak_planning = seg.segment_id == peak_id and bool(peak_id)

    return MissionGraph(
        segments=segments,
        peak_segment_id=peak_id,
        structural_incompatibility=structural_incompatibility,
    )


def save_mission_graph(data_used: Dict[str, Any], graph: MissionGraph) -> None:
    data_used[MISSION_GRAPH_KEY] = graph.to_dict()


def load_mission_graph(data_used: Optional[Dict[str, Any]]) -> Optional[MissionGraph]:
    if not isinstance(data_used, dict):
        return None
    raw = data_used.get(MISSION_GRAPH_KEY)
    if isinstance(raw, dict):
        return MissionGraph.from_dict(raw)
    return None


__all__ = [
    "MISSION_GRAPH_KEY",
    "MissionGraph",
    "MissionSegmentProfile",
    "SegmentKind",
    "build_mission_graph",
    "load_mission_graph",
    "save_mission_graph",
]
