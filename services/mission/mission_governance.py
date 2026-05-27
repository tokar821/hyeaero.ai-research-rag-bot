"""
Governance resolution before ranking — CEO mandate vs utilization reality vs mission structure.

Runs after route graph + passenger distribution; sets deferral flags, not narrative templates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.consultant.mission_state import MissionState
from services.mission.mission_understanding_engine import MissionUnderstandingPacket
from services.mission.models import MissionProfile

MISSION_GOVERNANCE_KEY = "mission_governance"

_CEO_MANDATE_RE = re.compile(
    r"\b(?:ceo|chairman|founder|leadership)\b.*\b(?:requires?|needs?|insists?|mandate)\b"
    r"|\b(?:requires?|needs?|insists?)\b.*\b(?:ceo|chairman|founder|leadership)\b"
    r"|\bceo\s+insists\b",
    re.I,
)
_DOMESTIC_SHARE_RE = re.compile(
    r"\b(\d{1,3})\s*%\s+of\s+flights?\s+are\s+(?:domestic|short)\b"
    r"|\b(\d{1,3})\s*%\s+(?:short\s+)?hops?\b",
    re.I,
)
_FOUNDER_COMPANY_SPLIT_RE = re.compile(
    r"\b(?:founder|chairman)\b.*\b(?:nonstop|flies?)\b.*\b"
    r"(?:abu\s+dhabi|dubai|singapore|riyadh|tokyo)\b",
    re.I,
)
_COMPANY_NA_SCOPE_RE = re.compile(
    r"\b(?:rest of the company|never leaves north america|company never)\b",
    re.I,
)
_DOMESTIC_HOP_RE = re.compile(
    r"\b(?:domestic|short)\s+(?:hops?|legs?|trips?)\b"
    r"|\b2\s*[-–]\s*3\s+hour\s+(?:domestic|legs?)\b",
    re.I,
)
_SINGLE_AIRCRAFT_PREF_RE = re.compile(
    r"\b(?:ideally\s+one|want\s+simplicity|one\s+aircraft|single\s+aircraft|only\s+one\s+jet)\b",
    re.I,
)
_PRIOR_JET_MISMATCH_RE = re.compile(
    r"\b(?:too\s+(?:big|large|limited)|inefficient\s+for\s+daily|mostly\s+idle|underutil)\b",
    re.I,
)
_ULR_CITY_RE = re.compile(
    r"\b(?:new\s+york|nyc|teterboro)\b.*\b(?:dubai|abu\s+dhabi|riyadh|tokyo|singapore)\b"
    r"|\b(?:dubai|abu\s+dhabi|riyadh|tokyo|singapore)\b.*\b(?:new\s+york|nyc)\b"
    r"|\bnonstop\s+(?:tokyo|dubai|abu\s+dhabi|singapore|riyadh)\b",
    re.I,
)


@dataclass
class MissionGovernanceResolution:
    ceo_ulr_mandate: bool = False
    domestic_utilization_dominant: bool = False
    single_aircraft_preference: bool = False
    utilization_mission_conflict: bool = False
    founder_company_asymmetry: bool = False
    defer_global_aircraft_ranking: bool = False
    governance_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ceo_ulr_mandate": self.ceo_ulr_mandate,
            "domestic_utilization_dominant": self.domestic_utilization_dominant,
            "single_aircraft_preference": self.single_aircraft_preference,
            "utilization_mission_conflict": self.utilization_mission_conflict,
            "founder_company_asymmetry": self.founder_company_asymmetry,
            "defer_global_aircraft_ranking": self.defer_global_aircraft_ranking,
            "governance_notes": list(self.governance_notes),
        }


def resolve_mission_governance(
    text: str,
    profile: MissionProfile,
    packet: Optional[MissionUnderstandingPacket],
    *,
    mission: Optional[MissionState] = None,
) -> MissionGovernanceResolution:
    tl = text or ""
    res = MissionGovernanceResolution()

    res.ceo_ulr_mandate = bool(
        _CEO_MANDATE_RE.search(tl)
        or (_ULR_CITY_RE.search(tl) and re.search(r"\bnonstop\b", tl, re.I))
    )
    res.domestic_utilization_dominant = bool(
        _DOMESTIC_SHARE_RE.search(tl) or _DOMESTIC_HOP_RE.search(tl)
    )
    res.single_aircraft_preference = bool(_SINGLE_AIRCRAFT_PREF_RE.search(tl))
    prior_mismatch = bool(_PRIOR_JET_MISMATCH_RE.search(tl))

    if res.ceo_ulr_mandate and res.domestic_utilization_dominant:
        res.utilization_mission_conflict = True
        res.governance_notes.append(
            "CEO ULR mandate conflicts with dominant short domestic utilization — portfolio governance required."
        )
    elif res.ceo_ulr_mandate and prior_mismatch:
        res.utilization_mission_conflict = True
        res.governance_notes.append(
            "Prior large-aircraft underutilization on domestic legs — ULR mandate is asymmetric to daily ops."
        )

    if _FOUNDER_COMPANY_SPLIT_RE.search(tl) and (
        _COMPANY_NA_SCOPE_RE.search(tl)
        or re.search(r"\bdaily\s+flights?\b.*\b(?:chicago|san\s+francisco|nyc)\b", tl, re.I)
    ):
        res.founder_company_asymmetry = True
        res.utilization_mission_conflict = True
        res.governance_notes.append(
            "Founder-exclusive ULR legs vs company domestic scope — portfolio governance, not one shared tail number."
        )

    if res.single_aircraft_preference and res.utilization_mission_conflict:
        res.governance_notes.append(
            "Single-aircraft preference acknowledged but structurally constrained by utilization asymmetry."
        )
    elif res.single_aircraft_preference:
        res.governance_notes.append(
            "Operator prefers one aircraft for simplicity — structural proof still governs if bands conflict."
        )

    # Defer global ranked shortlist when governance tension + multi-band or multi-route mission
    multi_route = len(profile.routes) >= 2
    multi_band = bool(
        packet
        and len(packet.fallback_operational_band or []) >= 2
    )
    incompatible = bool(
        packet and packet.inferred_constraints.get("incompatible_mission_bands")
    )

    res.defer_global_aircraft_ranking = bool(
        res.utilization_mission_conflict
        or res.founder_company_asymmetry
        or (res.ceo_ulr_mandate and (multi_route or multi_band))
        or (res.single_aircraft_preference and incompatible)
    )

    return res


def apply_governance_resolution(
    resolution: MissionGovernanceResolution,
    packet: Optional[MissionUnderstandingPacket],
    *,
    data_used: Optional[Dict[str, Any]] = None,
) -> None:
    if packet is None:
        return
    if resolution.founder_company_asymmetry:
        packet.inferred_constraints["founder_company_asymmetry"] = True
    if resolution.utilization_mission_conflict:
        packet.inferred_constraints["governance_asymmetry"] = True
        packet.inferred_constraints["incompatible_mission_bands"] = True
        bands = packet.fallback_operational_band or []
        if "Middle East ULR continuation band" not in bands:
            packet.fallback_operational_band = list(bands) + [
                "Middle East ULR continuation band"
            ]
        if "Long-leg executive band" not in packet.fallback_operational_band:
            packet.fallback_operational_band.append("Long-leg executive band")
    if resolution.single_aircraft_preference:
        packet.inferred_constraints["single_aircraft_request"] = True
    if resolution.defer_global_aircraft_ranking:
        packet.inferred_constraints["defer_global_shortlist"] = True
        packet.recommend_aircraft = False
    if resolution.governance_notes:
        for note in resolution.governance_notes:
            if note not in packet.understanding_notes:
                packet.understanding_notes.append(note)
        synth = (packet.operational_synthesis or "").strip()
        for note in resolution.governance_notes:
            if note and note not in synth:
                synth = f"{synth} {note}".strip() if synth else note
        if synth:
            packet.operational_synthesis = synth
    if resolution.ceo_ulr_mandate and resolution.domestic_utilization_dominant:
        extra = (
            "Governance: CEO nonstop ULR mandate vs dominant short domestic utilization — "
            "portfolio structure required; daily hops do not justify a single Global-class platform."
        )
        if extra not in (packet.operational_synthesis or ""):
            packet.operational_synthesis = f"{(packet.operational_synthesis or '').strip()} {extra}".strip()

    if isinstance(data_used, dict):
        data_used[MISSION_GOVERNANCE_KEY] = resolution.to_dict()
        if resolution.defer_global_aircraft_ranking:
            data_used["ranking_defer_governance"] = True
