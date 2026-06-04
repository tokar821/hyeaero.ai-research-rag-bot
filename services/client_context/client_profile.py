"""Persistent client profile across a conversation thread."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ClientProfile:
    preferred_budget_musd: Optional[float] = None
    preferred_manufacturers: List[str] = field(default_factory=list)
    preferred_aircraft: List[str] = field(default_factory=list)
    mission_patterns: List[str] = field(default_factory=list)
    acquisition_stage: str = "EXPLORING"
    risk_tolerance: str = "moderate"  # conservative | moderate | aggressive
    inferred_preferences: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preferred_budget_musd": self.preferred_budget_musd,
            "preferred_manufacturers": list(self.preferred_manufacturers),
            "preferred_aircraft": list(self.preferred_aircraft),
            "mission_patterns": list(self.mission_patterns),
            "acquisition_stage": self.acquisition_stage,
            "risk_tolerance": self.risk_tolerance,
            "inferred_preferences": dict(self.inferred_preferences),
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> ClientProfile:
        if not isinstance(raw, dict):
            return cls()
        return cls(
            preferred_budget_musd=_float_or_none(raw.get("preferred_budget_musd")),
            preferred_manufacturers=_str_list(raw.get("preferred_manufacturers")),
            preferred_aircraft=_str_list(raw.get("preferred_aircraft")),
            mission_patterns=_str_list(raw.get("mission_patterns")),
            acquisition_stage=str(raw.get("acquisition_stage") or "EXPLORING"),
            risk_tolerance=str(raw.get("risk_tolerance") or "moderate"),
            inferred_preferences=dict(raw.get("inferred_preferences") or {}),
        )


def _float_or_none(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _str_list(val: Any) -> List[str]:
    if not isinstance(val, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for item in val:
        s = str(item or "").strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out[:12]


__all__ = ["ClientProfile"]
