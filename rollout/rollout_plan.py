"""
Rollout plan — staged percentage rollout with next-stage recommendations.

Operational guidance only — does not modify RolloutController logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from services.routing.unified_emergency_rollback import evaluate_emergency_rollback
from services.routing.unified_intent_production_metrics import get_production_metrics
from services.telemetry.unified_rollout_telemetry import get_rollout_telemetry_snapshot

ROLLOUT_STAGES: Tuple[int, ...] = (0, 5, 10, 25, 50, 100)


@dataclass(frozen=True)
class RolloutStage:
    stage_id: int
    percent: int
    label: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "percent": self.percent,
            "label": self.label,
        }


def get_rollout_stages() -> List[RolloutStage]:
    return [
        RolloutStage(stage_id=i, percent=pct, label=f"Stage {i}: {pct}%")
        for i, pct in enumerate(ROLLOUT_STAGES)
    ]


def _current_configured_percent() -> int:
    raw = (os.getenv("UNIFIED_INTENT_ROLLOUT_PERCENT") or "0").strip()
    try:
        return max(0, min(100, int(raw)))
    except ValueError:
        return 0


def _nearest_stage_index(percent: int) -> int:
    return min(range(len(ROLLOUT_STAGES)), key=lambda i: abs(ROLLOUT_STAGES[i] - percent))


@dataclass(frozen=True)
class StageRecommendation:
    current_stage: RolloutStage
    recommended_stage: RolloutStage
    action: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_stage": self.current_stage.to_dict(),
            "recommended_stage": self.recommended_stage.to_dict(),
            "action": self.action,
            "reason": self.reason,
        }


def recommend_next_stage(
    *,
    rollout_telemetry: Optional[Dict[str, Any]] = None,
    production_metrics: Optional[Dict[str, Any]] = None,
) -> StageRecommendation:
    """
    Recommend next rollout stage based on rollback alerts, hardening failures, and divergence.
    """
    rollout = rollout_telemetry or get_rollout_telemetry_snapshot()
    production = production_metrics or get_production_metrics()
    rollback = evaluate_emergency_rollback(
        production_metrics=production,
        rollout_telemetry=rollout,
    )

    current_pct = _current_configured_percent()
    current_idx = _nearest_stage_index(current_pct)
    current_stage = get_rollout_stages()[current_idx]

    total = int(rollout.get("total_rollout_events") or 0)
    divergence_rate = float(rollout.get("authority_divergence_rate") or 0.0)
    hardening_failures = int(production.get("hardening_failure_count") or 0)
    failure_rate = hardening_failures / max(total, 1)

    if rollback.active or failure_rate > 0.15 or divergence_rate > 0.25:
        target_idx = max(0, current_idx - 1)
        action = "ROLLBACK" if rollback.active else "HOLD_OR_REDUCE"
        reason = (
            f"Rollback active={rollback.active}, hardening_failure_rate={failure_rate:.2f}, "
            f"divergence_rate={divergence_rate:.2f}"
        )
    elif total < 50:
        target_idx = current_idx
        action = "HOLD"
        reason = f"Insufficient traffic sample (n={total}); collect more events before advancing."
    elif divergence_rate < 0.05 and failure_rate < 0.02 and current_idx < len(ROLLOUT_STAGES) - 1:
        target_idx = min(current_idx + 1, len(ROLLOUT_STAGES) - 1)
        action = "ADVANCE"
        reason = (
            f"Metrics within limits: divergence_rate={divergence_rate:.2f}, "
            f"hardening_failure_rate={failure_rate:.2f}"
        )
    else:
        target_idx = current_idx
        action = "HOLD"
        reason = (
            f"Stabilize before advance: divergence_rate={divergence_rate:.2f}, "
            f"hardening_failure_rate={failure_rate:.2f}"
        )

    recommended = get_rollout_stages()[target_idx]
    return StageRecommendation(
        current_stage=current_stage,
        recommended_stage=recommended,
        action=action,
        reason=reason,
    )


__all__ = [
    "ROLLOUT_STAGES",
    "RolloutStage",
    "StageRecommendation",
    "get_rollout_stages",
    "recommend_next_stage",
]
