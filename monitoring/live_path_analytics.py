"""
Live path analytics — per-category request metrics from production traffic.

Does not modify routing or responders.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

PATH_CATEGORIES = (
    "FACT",
    "MARKET",
    "CAPABILITY",
    "COMPARISON",
    "ALTERNATIVE",
    "MISSION",
    "BUY_DECISION",
    "UNKNOWN",
)


@dataclass
class PathCategoryStats:
    category: str
    total_requests: int = 0
    successful_executions: int = 0
    fallback_executions: int = 0
    latency_sum_ms: float = 0.0

    @property
    def average_latency_ms(self) -> float:
        if self.total_requests <= 0:
            return 0.0
        return self.latency_sum_ms / self.total_requests

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "total_requests": self.total_requests,
            "successful_executions": self.successful_executions,
            "fallback_executions": self.fallback_executions,
            "average_latency_ms": round(self.average_latency_ms, 2),
        }


class LivePathAnalytics:
    """Aggregates live traffic by path category."""

    def __init__(self) -> None:
        self._stats: Dict[str, PathCategoryStats] = {
            cat: PathCategoryStats(category=cat) for cat in PATH_CATEGORIES
        }

    def record(
        self,
        category: str,
        *,
        unified_enforced: bool,
        fallback: bool,
        latency_ms: float,
    ) -> None:
        cat = category if category in self._stats else "UNKNOWN"
        row = self._stats[cat]
        row.total_requests += 1
        row.latency_sum_ms += max(0.0, float(latency_ms))
        if unified_enforced and not fallback:
            row.successful_executions += 1
        else:
            row.fallback_executions += 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            "by_category": {
                cat: self._stats[cat].to_dict()
                for cat in PATH_CATEGORIES
                if self._stats[cat].total_requests > 0
            },
            "totals": {
                "total_requests": sum(s.total_requests for s in self._stats.values()),
                "successful_executions": sum(
                    s.successful_executions for s in self._stats.values()
                ),
                "fallback_executions": sum(
                    s.fallback_executions for s in self._stats.values()
                ),
            },
        }

    def reset(self) -> None:
        for cat in PATH_CATEGORIES:
            self._stats[cat] = PathCategoryStats(category=cat)


_GLOBAL_ANALYTICS = LivePathAnalytics()


def get_live_path_analytics() -> LivePathAnalytics:
    return _GLOBAL_ANALYTICS


def reset_live_path_analytics() -> None:
    _GLOBAL_ANALYTICS.reset()


def infer_path_category(
    execution_path: str,
    *,
    qri_intent: str = "",
) -> str:
    """Map execution path + QRI to golden-style category labels."""
    path = (execution_path or "none").strip().lower()
    mapping = {
        "aircraft_fact": "FACT",
        "aircraft_market_fact": "MARKET",
        "capability": "CAPABILITY",
        "comparison": "COMPARISON",
        "alternative": "ALTERNATIVE",
    }
    if path in mapping:
        return mapping[path]
    qri = (qri_intent or "").lower()
    if qri in ("acquisition_recommendation", "shortlist_ranking", "ownership_economics"):
        if "acquisition" in qri or "shortlist" in qri:
            return "BUY_DECISION"
    if path == "none":
        return "MISSION"
    return "UNKNOWN"


__all__ = [
    "LivePathAnalytics",
    "PATH_CATEGORIES",
    "get_live_path_analytics",
    "infer_path_category",
    "reset_live_path_analytics",
]
