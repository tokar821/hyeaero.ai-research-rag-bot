"""Resale strength — segment demand and depreciation posture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ResaleAssessment:
    model: str
    resale_strength: str  # strong | stable | soft | unknown
    commentary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "resale_strength": self.resale_strength,
            "commentary": self.commentary,
        }


_SEGMENT_RESALE = {
    "g650": ("strong", "ULR benchmark; resale supported by global demand."),
    "challenger 350": ("stable", "Super-mid with consistent secondary demand."),
    "citation cj3+": ("stable", "Light jet with deep buyer pool."),
}


def assess_resale_strength(model: str) -> ResaleAssessment:
    key = (model or "").strip().lower()
    strength, commentary = _SEGMENT_RESALE.get(
        key, ("unknown", "Resale posture requires serial-specific comp review.")
    )
    return ResaleAssessment(model=model, resale_strength=strength, commentary=commentary)
