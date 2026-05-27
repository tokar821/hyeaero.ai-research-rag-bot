"""
Operational band validator — bands need corridor proof before they become renderable segments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from services.mission.segment_kinds import SegmentKind

RENDER_CONFIDENCE_MIN = 0.72

_PACIFIC_ANCHOR_RE = re.compile(
    r"\b(?:tokyo|osaka|seoul|hong\s+kong|singapore|sydney|beijing|shanghai|"
    r"manila|auckland|los\s+angeles|san\s+francisco|vancouver|pacific)\b",
    re.I,
)
_ME_RE = re.compile(
    r"\b(?:dubai|riyadh|doha|abu\s+dhabi|jeddah|middle\s+east)\b",
    re.I,
)
_MOUNTAIN_RE = re.compile(
    r"\b(?:aspen|kase|ktex|kege|jackson|telluride|sun\s+valley)\b",
    re.I,
)
_INDUSTRIAL_RE = re.compile(
    r"\b(?:remote\s+drilling|arctic\s+oil|west\s+africa|oil\s+platform|mining|gravel)\b",
    re.I,
)
_CARIBBEAN_RE = re.compile(r"\bcaribbean\b", re.I)


@dataclass
class BandValidation:
    band: str
    segment_kind: SegmentKind
    confidence: float
    renderable: bool
    reason: str

    def to_dict(self):
        return {
            "band": self.band,
            "segment_kind": self.segment_kind.value,
            "confidence": round(self.confidence, 3),
            "renderable": self.renderable,
            "reason": self.reason,
        }


def validate_operational_band(
    band: str,
    *,
    route_labels: List[str],
    query: str = "",
) -> BandValidation:
    """Score whether an operational band may attach to a segment with route proof."""
    b = (band or "").strip()
    routes_blob = " ".join(route_labels)
    ql = (query or "").lower()
    blob = f"{routes_blob} {ql}".lower()

    kind = SegmentKind.DOMESTIC_EXECUTIVE
    confidence = 0.5
    reason = "Unvalidated band"

    if re.search(r"continuation|middle east", b, re.I):
        kind = SegmentKind.ULR_CONTINUATION
        if _ME_RE.search(blob) and route_labels:
            confidence = 0.9
            reason = "ME continuation corridor confirmed on routes"
        elif _ME_RE.search(blob):
            confidence = 0.75
            reason = "ME reference in mission text"
        else:
            confidence = 0.4
            reason = "Continuation band without ME route proof"

    elif re.search(r"multi-leg ultra|pacific", b, re.I):
        kind = SegmentKind.PACIFIC_ULR
        if _PACIFIC_ANCHOR_RE.search(blob) and route_labels:
            confidence = 0.88
            reason = "Pacific ULR corridor on routes"
        else:
            confidence = 0.35
            reason = "Pacific ULR band without Pacific route proof"

    elif re.search(r"mountain|short-strip", b, re.I):
        kind = SegmentKind.MOUNTAIN_FIELD
        if _MOUNTAIN_RE.search(blob) and route_labels:
            confidence = 0.9
            reason = "Mountain airport on routes"
        else:
            confidence = 0.3
            reason = "Mountain band without mountain route proof"

    elif re.search(r"field-access|industrial", b, re.I):
        kind = SegmentKind.INDUSTRIAL_FIELD
        if _INDUSTRIAL_RE.search(blob) and route_labels:
            confidence = 0.85
            reason = "Industrial/remote field on routes"
        else:
            confidence = 0.35
            reason = "Industrial band without field route proof"

    elif re.search(r"caribbean", b, re.I):
        kind = SegmentKind.CARIBBEAN_REGIONAL
        if _CARIBBEAN_RE.search(blob) and route_labels:
            confidence = 0.88
            reason = "Caribbean corridor on routes"
        else:
            confidence = 0.4
            reason = "Caribbean band without route proof"

    elif re.search(r"transatlantic|heavy-cabin", b, re.I):
        kind = SegmentKind.TRANSATLANTIC_EXECUTIVE
        if route_labels:
            confidence = 0.85
            reason = "Transatlantic routes present"
        else:
            confidence = 0.45
            reason = "Transatlantic band without route binding"

    else:
        if route_labels:
            confidence = 0.8
            reason = "Executive band with route binding"
        else:
            confidence = 0.45
            reason = "Band without routes"

    return BandValidation(
        band=b,
        segment_kind=kind,
        confidence=confidence,
        renderable=confidence >= RENDER_CONFIDENCE_MIN and bool(route_labels),
        reason=reason,
    )


def band_may_create_segment(
    band: str,
    route_labels: List[str],
    *,
    query: str = "",
) -> Tuple[bool, SegmentKind, float]:
    """Gate: bands alone never create segments — only strengthen route-born segments."""
    v = validate_operational_band(band, route_labels=route_labels, query=query)
    return v.renderable, v.segment_kind, v.confidence


__all__ = [
    "RENDER_CONFIDENCE_MIN",
    "BandValidation",
    "band_may_create_segment",
    "validate_operational_band",
]
