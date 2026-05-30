"""
Unified intent rollout controller — deterministic traffic segmentation for production enablement.

Does NOT modify routing, execution_path resolution, or responder behavior.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from services.routing.unified_emergency_rollback import RollbackStatus


def _env_truthy(key: str) -> bool:
    return (os.getenv(key) or "").strip().lower() in ("1", "true", "yes")


def _rollout_percent() -> int:
    raw = (os.getenv("UNIFIED_INTENT_ROLLOUT_PERCENT") or "0").strip()
    try:
        value = int(raw)
    except ValueError:
        return 0
    return max(0, min(100, value))


def _deterministic_bucket(session_key: str) -> int:
    """Return stable bucket 0–99 for a session identifier."""
    digest = hashlib.sha256(session_key.encode("utf-8")).hexdigest()
    return int(digest, 16) % 100


def extract_rollout_session_keys(
    client_conversation_state: Optional[Dict[str, Any]],
) -> tuple[Optional[str], Optional[str]]:
    """
    Extract rollout segmentation keys from conversation state.

    Does not invent identifiers — returns None when unavailable.
    """
    if not isinstance(client_conversation_state, dict):
        return None, None
    user_id = client_conversation_state.get("user_id")
    user_key = str(user_id).strip() if user_id is not None and str(user_id).strip() else None
    conversation_id = (
        client_conversation_state.get("conversation_id")
        or client_conversation_state.get("session_id")
        or client_conversation_state.get("id")
    )
    conv_key = (
        str(conversation_id).strip()
        if conversation_id is not None and str(conversation_id).strip()
        else None
    )
    return user_key, conv_key


@dataclass(frozen=True)
class RolloutDecision:
    enabled: bool
    source: str
    reason: str
    rollout_percent: int = 0
    session_bucket: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "source": self.source,
            "reason": self.reason,
            "rollout_percent": self.rollout_percent,
            "session_bucket": self.session_bucket,
        }


def evaluate_rollout(
    *,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    rollback_status: Optional["RollbackStatus"] = None,
) -> RolloutDecision:
    """
    Determine whether unified intent pipeline authority applies for this request.

    Priority: force_legacy > force_unified > percentage > default.

    ``rollback_status`` is accepted for observability correlation only — Phase 7
    does not auto-disable unified authority from rollback signals (observe-only).
    """
    _ = rollback_status  # correlated in data_used; does not alter decision (Phase 7)
    percent = _rollout_percent()

    if _env_truthy("UNIFIED_INTENT_FORCE_LEGACY"):
        return RolloutDecision(
            enabled=False,
            source="force_legacy",
            reason="UNIFIED_INTENT_FORCE_LEGACY is active.",
            rollout_percent=percent,
        )

    if _env_truthy("UNIFIED_INTENT_FORCE_UNIFIED"):
        return RolloutDecision(
            enabled=True,
            source="force_unified",
            reason="UNIFIED_INTENT_FORCE_UNIFIED is active.",
            rollout_percent=percent,
        )

    if percent >= 100:
        return RolloutDecision(
            enabled=True,
            source="percentage_rollout",
            reason=f"Rollout percent {percent} includes all traffic.",
            rollout_percent=percent,
            session_bucket=0,
        )

    if percent <= 0:
        return RolloutDecision(
            enabled=False,
            source="default",
            reason="Rollout percent is 0 — legacy pipeline default.",
            rollout_percent=percent,
        )

    session_key = user_id or conversation_id
    if not session_key:
        return RolloutDecision(
            enabled=False,
            source="default",
            reason="No user_id or conversation_id available for deterministic percent rollout.",
            rollout_percent=percent,
        )

    bucket = _deterministic_bucket(session_key)
    in_rollout = bucket < percent
    return RolloutDecision(
        enabled=in_rollout,
        source="percentage_rollout",
        reason=(
            f"Deterministic bucket {bucket} "
            f"{'<' if in_rollout else '>='} rollout percent {percent}."
        ),
        rollout_percent=percent,
        session_bucket=bucket,
    )


def attach_rollout_decision(
    data_used: Dict[str, Any],
    decision: RolloutDecision,
) -> None:
    """Merge rollout decision into data_used for observability."""
    data_used["unified_rollout_decision"] = decision.to_dict()


__all__ = [
    "RolloutDecision",
    "attach_rollout_decision",
    "evaluate_rollout",
    "extract_rollout_session_keys",
]
