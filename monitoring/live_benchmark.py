"""
Live benchmark — metadata-only legacy vs unified comparison from production traffic.

Does not compare semantic text quality or modify pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class LiveBenchmarkAggregator:
    """Accumulates structural metadata comparisons per live request."""

    total_events: int = 0
    routing_agreement_count: int = 0
    path_agreement_count: int = 0
    unified_latency_sum_ms: float = 0.0
    legacy_latency_sum_ms: float = 0.0
    unified_length_sum: int = 0
    legacy_length_sum: int = 0
    length_delta_sum: int = 0

    def record(
        self,
        *,
        routing_agreement: bool,
        path_agreement: bool,
        unified_latency_ms: float,
        legacy_latency_ms: float,
        unified_output_length: int,
        legacy_output_length: int,
    ) -> None:
        self.total_events += 1
        if routing_agreement:
            self.routing_agreement_count += 1
        if path_agreement:
            self.path_agreement_count += 1
        self.unified_latency_sum_ms += max(0.0, float(unified_latency_ms))
        self.legacy_latency_sum_ms += max(0.0, float(legacy_latency_ms))
        self.unified_length_sum += max(0, int(unified_output_length))
        self.legacy_length_sum += max(0, int(legacy_output_length))
        self.length_delta_sum += abs(
            int(unified_output_length) - int(legacy_output_length)
        )

    def snapshot(self) -> Dict[str, Any]:
        n = self.total_events
        if n == 0:
            return {
                "total_events": 0,
                "routing_agreement_rate": 0.0,
                "path_agreement_rate": 0.0,
                "mean_unified_latency_ms": 0.0,
                "mean_legacy_latency_ms": 0.0,
                "mean_latency_delta_ms": 0.0,
                "mean_unified_output_length": 0.0,
                "mean_legacy_output_length": 0.0,
                "mean_length_delta": 0.0,
            }
        return {
            "total_events": n,
            "routing_agreement_rate": round(self.routing_agreement_count / n, 4),
            "path_agreement_rate": round(self.path_agreement_count / n, 4),
            "mean_unified_latency_ms": round(self.unified_latency_sum_ms / n, 2),
            "mean_legacy_latency_ms": round(self.legacy_latency_sum_ms / n, 2),
            "mean_latency_delta_ms": round(
                (self.unified_latency_sum_ms - self.legacy_latency_sum_ms) / n, 2
            ),
            "mean_unified_output_length": round(self.unified_length_sum / n, 1),
            "mean_legacy_output_length": round(self.legacy_length_sum / n, 1),
            "mean_length_delta": round(self.length_delta_sum / n, 1),
        }

    def reset(self) -> None:
        self.total_events = 0
        self.routing_agreement_count = 0
        self.path_agreement_count = 0
        self.unified_latency_sum_ms = 0.0
        self.legacy_latency_sum_ms = 0.0
        self.unified_length_sum = 0
        self.legacy_length_sum = 0
        self.length_delta_sum = 0


_GLOBAL_BENCHMARK = LiveBenchmarkAggregator()


def get_live_benchmark() -> LiveBenchmarkAggregator:
    return _GLOBAL_BENCHMARK


def reset_live_benchmark() -> None:
    _GLOBAL_BENCHMARK.reset()


def compare_live_metadata(
    *,
    unified_execution_path: str,
    legacy_qri_intent: str,
    authority_aligned: bool,
    unified_latency_ms: float = 0.0,
    legacy_latency_ms: float = 0.0,
    unified_output_length: int = 0,
    legacy_output_length: int = 0,
) -> Dict[str, Any]:
    """
    Structural routing/path agreement check — metadata only.
    """
    from monitoring.live_path_analytics import infer_path_category

    unified_cat = infer_path_category(unified_execution_path)
    legacy_cat = infer_path_category("none", qri_intent=legacy_qri_intent)

    path_agreement = unified_cat == legacy_cat or authority_aligned
    routing_agreement = authority_aligned or (
        unified_execution_path != "none" and legacy_qri_intent != ""
    )

    get_live_benchmark().record(
        routing_agreement=routing_agreement,
        path_agreement=path_agreement,
        unified_latency_ms=unified_latency_ms,
        legacy_latency_ms=legacy_latency_ms,
        unified_output_length=unified_output_length,
        legacy_output_length=legacy_output_length,
    )

    return {
        "routing_agreement": routing_agreement,
        "path_agreement": path_agreement,
        "unified_category": unified_cat,
        "legacy_category": legacy_cat,
    }


__all__ = [
    "LiveBenchmarkAggregator",
    "compare_live_metadata",
    "get_live_benchmark",
    "reset_live_benchmark",
]
