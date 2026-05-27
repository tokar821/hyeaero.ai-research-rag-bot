"""
Segment authority — every rendered segment must bind to routes or segment-scoped constraints.

Fallback segments are internal-only; they never render to the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from services.mission.mission_graph import MissionSegmentProfile
from services.mission.segment_kinds import SegmentKind

AuthorityType = Literal[
    "direct_route",
    "constraint_inference",
    "continuation_route",
    "fallback",
]

RENDER_CONFIDENCE_MIN = 0.72

_PACIFIC_ANCHOR_RE = re.compile(
    r"\b(?:tokyo|osaka|seoul|hong\s+kong|singapore|sydney|beijing|shanghai|"
    r"manila|auckland|los\s+angeles|san\s+francisco|vancouver|pacific)\b",
    re.I,
)
_MOUNTAIN_PLACE_RE = re.compile(
    r"\b(?:aspen|kase|ktex|kege|jackson|telluride|sun\s+valley|eagle\s+county)\b",
    re.I,
)
_INDUSTRIAL_PLACE_RE = re.compile(
    r"\b(?:remote\s+drilling|arctic\s+oil|west\s+africa|oil\s+platform|"
    r"mining|gravel|unpaved|industrial\s+airport)\b",
    re.I,
)
_ME_RE = re.compile(
    r"\b(?:dubai|riyadh|doha|abu\s+dhabi|jeddah|middle\s+east)\b",
    re.I,
)


@dataclass
class DomainScopedConstraintSet:
    """Segment-local constraints — never globalized to the whole mission."""

    mountain: bool = False
    industrial: bool = False
    short_runway: bool = False
    gravel: bool = False
    cargo: bool = False
    continuation_ulr: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            k: v
            for k, v in (
                ("mountain", self.mountain),
                ("industrial", self.industrial),
                ("short_runway", self.short_runway),
                ("gravel", self.gravel),
                ("cargo", self.cargo),
                ("continuation_ulr", self.continuation_ulr),
            )
            if v
        }

    def constraint_keys(self) -> List[str]:
        return list(self.to_dict().keys())


@dataclass
class SegmentAuthority:
    segment_name: str
    source_routes: List[str] = field(default_factory=list)
    source_constraints: List[str] = field(default_factory=list)
    source_entities: List[str] = field(default_factory=list)
    confidence: float = 1.0
    authority_type: AuthorityType = "direct_route"
    why_exists: str = ""
    conflict_note: str = ""
    implication: str = ""

    @property
    def has_authority(self) -> bool:
        return bool(self.source_routes or self.source_constraints)

    @property
    def renderable(self) -> bool:
        if self.authority_type == "fallback":
            return False
        if not self.has_authority:
            return False
        return self.confidence >= RENDER_CONFIDENCE_MIN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_name": self.segment_name,
            "source_routes": list(self.source_routes),
            "source_constraints": list(self.source_constraints),
            "source_entities": list(self.source_entities),
            "confidence": round(self.confidence, 3),
            "authority_type": self.authority_type,
            "renderable": self.renderable,
            "why_exists": self.why_exists,
            "conflict_note": self.conflict_note,
            "implication": self.implication,
        }


def infer_domain_constraints_for_segment(
    seg: MissionSegmentProfile,
    *,
    cargo_required: bool = False,
) -> DomainScopedConstraintSet:
    """Map segment routes/kind to local constraints only."""
    routes_blob = " ".join(seg.route_labels).lower()
    scoped = DomainScopedConstraintSet()
    if seg.kind == SegmentKind.MOUNTAIN_FIELD or _MOUNTAIN_PLACE_RE.search(routes_blob):
        scoped.mountain = True
        scoped.short_runway = True
    if seg.kind == SegmentKind.INDUSTRIAL_FIELD or _INDUSTRIAL_PLACE_RE.search(routes_blob):
        scoped.industrial = True
        scoped.short_runway = True
    if seg.kind in (SegmentKind.ULR_CONTINUATION, SegmentKind.PACIFIC_ULR):
        scoped.continuation_ulr = True
    if seg.kind == SegmentKind.CARIBBEAN_REGIONAL:
        scoped.short_runway = True
    if cargo_required and seg.route_labels:
        scoped.cargo = True
    return scoped


def build_segment_authority(
    seg: MissionSegmentProfile,
    *,
    stage_nm_by_route: Optional[Dict[str, float]] = None,
) -> SegmentAuthority:
    """Derive authority record for one segment — used for render gating."""
    routes = list(seg.route_labels or [])
    constraints = infer_domain_constraints_for_segment(seg)
    constraint_keys = constraints.constraint_keys()
    nm_map = stage_nm_by_route or {}
    peak_nm = max((nm_map.get(r, 0.0) for r in routes), default=0.0)
    routes_blob = " ".join(routes)

    authority_type: AuthorityType = "direct_route"
    confidence = 0.95
    why = ""
    implication = ""
    entities: List[str] = []

    if not routes and constraint_keys:
        authority_type = "constraint_inference"
        confidence = 0.65
        why = f"Constraint-only segment ({', '.join(constraint_keys)}) — insufficient route binding."

    if seg.kind == SegmentKind.ULR_CONTINUATION:
        authority_type = "continuation_route"
        me_routes = [r for r in routes if _ME_RE.search(r)]
        if me_routes:
            confidence = 0.92
            why = f"Middle East / founder continuation authority: {me_routes[0]}"
            entities.extend(me_routes[:2])
        elif routes:
            confidence = 0.88
            why = f"Long-stage continuation: {routes[0]}"
        implication = (
            "Daily domestic utilization rarely supports dedicated ULR ownership on peak legs."
        )

    elif seg.kind == SegmentKind.PACIFIC_ULR:
        pacific_routes = [r for r in routes if _PACIFIC_ANCHOR_RE.search(r)]
        if not pacific_routes:
            authority_type = "fallback"
            confidence = 0.35
            why = "Pacific ULR band without Pacific corridor route proof."
        else:
            confidence = 0.9 if peak_nm >= 4800 else 0.78
            why = f"Pacific / Asia ULR corridor: {pacific_routes[0]}"
            entities.extend(pacific_routes[:2])
            implication = "Oceanic stages require worst-case leg planning — not a domestic jet problem."

    elif seg.kind == SegmentKind.MOUNTAIN_FIELD:
        mountain_routes = [r for r in routes if _MOUNTAIN_PLACE_RE.search(r)]
        if not mountain_routes:
            authority_type = "fallback"
            confidence = 0.3
            why = "Mountain segment without mountain airport route proof."
        else:
            confidence = 0.9
            why = f"Mountain / short-strip authority: {mountain_routes[0]}"
            entities.extend(mountain_routes[:2])
            implication = "Winter density altitude and runway length dominate — not cabin luxury."

    elif seg.kind == SegmentKind.INDUSTRIAL_FIELD:
        industrial_routes = [r for r in routes if _INDUSTRIAL_PLACE_RE.search(r)]
        if not industrial_routes:
            authority_type = "fallback"
            confidence = 0.35
            why = "Industrial segment without field-access route proof."
        else:
            confidence = 0.88
            why = f"Industrial / remote field authority: {industrial_routes[0]}"
            entities.extend(industrial_routes[:2])

    elif routes:
        why = f"Direct route authority: {routes[0]}"
        if len(routes) > 1:
            why += f" (+{len(routes) - 1} supporting legs)"
        confidence = 0.9

    elif (seg.operational_band or "").strip():
        authority_type = "fallback"
        confidence = 0.4
        why = f"Operational band only — no isolated routes: {(seg.operational_band or '')[:80]}"

    auth = SegmentAuthority(
        segment_name=seg.label or seg.segment_id,
        source_routes=routes,
        source_constraints=constraint_keys,
        source_entities=entities or routes[:2],
        confidence=confidence,
        authority_type=authority_type,
        why_exists=why,
        implication=implication,
    )
    if constraint_keys and routes:
        auth.source_constraints = constraint_keys
    return auth


def filter_renderable_segments(
    segments: List[MissionSegmentProfile],
    *,
    stage_nm_by_route: Optional[Dict[str, float]] = None,
) -> tuple[List[MissionSegmentProfile], List[SegmentAuthority]]:
    """Drop ghost / fallback segments; return surviving segments + authority records."""
    kept: List[MissionSegmentProfile] = []
    authorities: List[SegmentAuthority] = []
    for seg in segments:
        auth = build_segment_authority(seg, stage_nm_by_route=stage_nm_by_route)
        authorities.append(auth)
        if auth.renderable:
            seg.constraints = {
                **dict(seg.constraints or {}),
                **infer_domain_constraints_for_segment(seg).to_dict(),
            }
            seg.constraints["segment_authority"] = auth.to_dict()
            kept.append(seg)
    return kept, authorities


__all__ = [
    "RENDER_CONFIDENCE_MIN",
    "AuthorityType",
    "DomainScopedConstraintSet",
    "SegmentAuthority",
    "build_segment_authority",
    "filter_renderable_segments",
    "infer_domain_constraints_for_segment",
]
