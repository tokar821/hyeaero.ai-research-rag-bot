"""Assemble broker-facing conversation context for a turn."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.client_context.acquisition_stage_detector import (
    AcquisitionStage,
    detect_acquisition_stage,
    merge_stage,
)
from services.client_context.client_profile import ClientProfile
from services.client_context.conversation_memory import ConversationMemory


@dataclass
class BrokerConversationContext:
    remembered_budget_musd: Optional[float] = None
    remembered_targets: List[str] = field(default_factory=list)
    stage: str = AcquisitionStage.EXPLORING.value
    active_aircraft: Optional[str] = None
    active_constraints: Dict[str, Any] = field(default_factory=dict)
    preferred_manufacturers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "remembered_budget_musd": self.remembered_budget_musd,
            "remembered_targets": list(self.remembered_targets),
            "stage": self.stage,
            "active_aircraft": self.active_aircraft,
            "active_constraints": dict(self.active_constraints),
            "preferred_manufacturers": list(self.preferred_manufacturers),
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> BrokerConversationContext:
        if not isinstance(raw, dict):
            return cls()
        return cls(
            remembered_budget_musd=raw.get("remembered_budget_musd"),
            remembered_targets=list(raw.get("remembered_targets") or []),
            stage=str(raw.get("stage") or AcquisitionStage.EXPLORING.value),
            active_aircraft=raw.get("active_aircraft"),
            active_constraints=dict(raw.get("active_constraints") or {}),
            preferred_manufacturers=list(raw.get("preferred_manufacturers") or []),
        )


def build_broker_context(
    profile: ClientProfile,
    memory: ConversationMemory,
    *,
    query: str = "",
    intent_persistence: Optional[Dict[str, Any]] = None,
) -> BrokerConversationContext:
    """Build turn context from profile + memory + optional intent persistence."""
    ip = intent_persistence or {}
    active_aircraft = (
        str(ip.get("active_aircraft") or "").strip()
        or (profile.preferred_aircraft[0] if profile.preferred_aircraft else None)
    )

    detected = detect_acquisition_stage(query, prior_stage=profile.acquisition_stage)
    stage = merge_stage(profile.acquisition_stage, detected)

    constraints: Dict[str, Any] = {}
    if profile.preferred_budget_musd is not None:
        constraints["budget_musd"] = profile.preferred_budget_musd
    if memory.last_comparison_pair:
        constraints["comparison_pair"] = list(memory.last_comparison_pair)

    targets = list(profile.preferred_aircraft[:6])
    if active_aircraft and active_aircraft not in targets:
        targets.insert(0, active_aircraft)

    return BrokerConversationContext(
        remembered_budget_musd=profile.preferred_budget_musd,
        remembered_targets=targets,
        stage=stage,
        active_aircraft=active_aircraft,
        active_constraints=constraints,
        preferred_manufacturers=list(profile.preferred_manufacturers),
    )


__all__ = ["BrokerConversationContext", "build_broker_context"]
