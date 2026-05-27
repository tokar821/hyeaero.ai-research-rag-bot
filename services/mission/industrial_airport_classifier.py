"""
Industrial / remote airport classifier — gravel, oil, mining, remote strips.

Output is structural metadata for pre-ranking graph and band proof, not response templates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

_INDUSTRIAL_AIRPORT_CLASSIFIER_KEY = "industrial_airport_profile"


class IndustrialAirportClass(str, Enum):
    GRAVEL = "gravel"
    OIL_FIELD = "oil_field"
    MINING = "mining"
    REMOTE_STRIP = "remote_strip"
    INDUSTRIAL = "industrial"


_CLASS_PATTERNS: tuple[tuple[IndustrialAirportClass, re.Pattern], ...] = (
    (
        IndustrialAirportClass.GRAVEL,
        re.compile(
            r"\b(?:gravel|unpaved|dirt|grass)\s+(?:strips?|runways?|fields?)\b"
            r"|\bshort\s+gravel\b",
            re.I,
        ),
    ),
    (
        IndustrialAirportClass.OIL_FIELD,
        re.compile(
            r"\b(?:oil\s+fields?|oil\s+sites?|remote\s+oil|petroleum|pipeline\s+site)\b",
            re.I,
        ),
    ),
    (
        IndustrialAirportClass.MINING,
        re.compile(
            r"\b(?:mining|mine\s+site|mineral|quarry)\b",
            re.I,
        ),
    ),
    (
        IndustrialAirportClass.REMOTE_STRIP,
        re.compile(
            r"\b(?:remote\s+strips?|northern\s+canada|bush\s+strip|isolated\s+field|"
            r"arctic\s+oil|drilling\s+sites?|mining\s+strips?|west\s+africa)\b",
            re.I,
        ),
    ),
    (
        IndustrialAirportClass.INDUSTRIAL,
        re.compile(
            r"\b(?:industrial\s+airports?|factory\s+site|plant\s+site|smaller\s+industrial)\b",
            re.I,
        ),
    ),
)


@dataclass
class IndustrialAirportProfile:
    classes: List[IndustrialAirportClass] = field(default_factory=list)
    runway_over_cabin: bool = False
    field_access_required: bool = False
    summary: str = ""

    @property
    def active(self) -> bool:
        return bool(self.classes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classes": [c.value for c in self.classes],
            "runway_over_cabin": self.runway_over_cabin,
            "field_access_required": self.field_access_required,
            "summary": self.summary,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> Optional["IndustrialAirportProfile"]:
        if not isinstance(raw, dict):
            return None
        classes: List[IndustrialAirportClass] = []
        for v in raw.get("classes") or []:
            try:
                classes.append(IndustrialAirportClass(str(v)))
            except ValueError:
                continue
        return cls(
            classes=classes,
            runway_over_cabin=bool(raw.get("runway_over_cabin")),
            field_access_required=bool(raw.get("field_access_required")),
            summary=str(raw.get("summary") or ""),
        )


def classify_industrial_airports(text: str) -> IndustrialAirportProfile:
    tl = text or ""
    found: List[IndustrialAirportClass] = []
    for kind, pat in _CLASS_PATTERNS:
        if pat.search(tl) and kind not in found:
            found.append(kind)

    runway_over = bool(
        found
        and re.search(
            r"\b(?:matters?\s+more\s+than\s+cabin|runway|reliability\s+into|dispatch\s+failures?)\b",
            tl,
            re.I,
        )
    )
    if found and re.search(r"\b(?:short|unpaved|gravel|strip)\b", tl, re.I):
        runway_over = True

    labels = [c.value.replace("_", " ") for c in found]
    summary = f"Field-access classes: {', '.join(labels)}" if labels else ""

    return IndustrialAirportProfile(
        classes=found,
        runway_over_cabin=runway_over,
        field_access_required=bool(found),
        summary=summary,
    )


def apply_industrial_profile_to_mission(
    profile,
    packet,
    industrial: IndustrialAirportProfile,
    *,
    data_used: Optional[Dict] = None,
) -> None:
    from services.mission.models import PriorityLevel

    if not industrial.active:
        return

    profile.runway_priority = PriorityLevel.HIGH
    profile.short_field_priority = PriorityLevel.HIGH
    for c in industrial.classes:
        tag = f"industrial_{c.value}"
        if tag not in profile.airport_constraints:
            profile.airport_constraints.append(tag)

    if packet is not None:
        packet.inferred_constraints["industrial_airport_access"] = True
        if industrial.runway_over_cabin:
            packet.inferred_constraints["runway_over_cabin"] = True
        if "Domestic field-access executive band" not in (
            packet.fallback_operational_band or []
        ):
            packet.fallback_operational_band.append(
                "Domestic field-access executive band"
            )

    if isinstance(data_used, dict):
        data_used[_INDUSTRIAL_AIRPORT_CLASSIFIER_KEY] = industrial.to_dict()


__all__ = [
    "IndustrialAirportClass",
    "IndustrialAirportProfile",
    "classify_industrial_airports",
    "apply_industrial_profile_to_mission",
    "_INDUSTRIAL_AIRPORT_CLASSIFIER_KEY",
]
