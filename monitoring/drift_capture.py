"""
Production drift capture — ring buffer of live routing events for regression analysis.

Does not modify routing or responders.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

MAX_DRIFT_EVENTS = 10_000


@dataclass(frozen=True)
class DriftEvent:
    query: str
    execution_path: str
    model: Optional[str]
    hardening_flags: Dict[str, bool]
    rollback_status: Dict[str, Any]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    qri_intent: str = ""
    rollout_enabled: bool = False
    unified_enforced: bool = False
    path_category: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query[:500],
            "execution_path": self.execution_path,
            "model": self.model,
            "hardening_flags": dict(self.hardening_flags),
            "rollback_status": dict(self.rollback_status),
            "timestamp": self.timestamp,
            "qri_intent": self.qri_intent,
            "rollout_enabled": self.rollout_enabled,
            "unified_enforced": self.unified_enforced,
            "path_category": self.path_category,
        }


class DriftCapture:
    """In-memory store of recent production drift events (last 10,000)."""

    def __init__(self, *, max_events: int = MAX_DRIFT_EVENTS) -> None:
        self._events: Deque[DriftEvent] = deque(maxlen=max_events)

    def append(self, event: DriftEvent) -> None:
        self._events.append(event)

    def count(self) -> int:
        return len(self._events)

    def export(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        items = list(self._events)
        if limit is not None:
            items = items[-limit:]
        return [e.to_dict() for e in items]

    def clear(self) -> None:
        self._events.clear()


_GLOBAL_CAPTURE = DriftCapture()


def get_drift_capture() -> DriftCapture:
    return _GLOBAL_CAPTURE


def capture_drift_event(event: DriftEvent) -> None:
    _GLOBAL_CAPTURE.append(event)


def reset_drift_capture() -> None:
    _GLOBAL_CAPTURE.clear()


__all__ = [
    "DriftCapture",
    "DriftEvent",
    "MAX_DRIFT_EVENTS",
    "capture_drift_event",
    "get_drift_capture",
    "reset_drift_capture",
]
