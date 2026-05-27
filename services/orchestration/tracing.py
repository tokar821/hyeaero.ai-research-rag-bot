"""Execution tracing for the consultant orchestration pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.orchestration.constants import ORCHESTRATION_STAGES
from services.orchestration.modes import OrchestrationMode, orchestration_mode, structured_logging_enabled

logger = logging.getLogger(__name__)


@dataclass
class StageRecord:
    stage: str
    status: str  # ok | skipped | failed | degraded
    duration_ms: float = 0.0
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self, *, verbose: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "stage": self.stage,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "confidence": round(self.confidence, 3),
        }
        if self.error:
            out["error"] = self.error
        if verbose and self.details:
            out["details"] = dict(self.details)
        elif self.details and not verbose:
            # Production: keep only compact summary keys
            compact = {}
            for key in (
                "feasible_count",
                "eliminated_count",
                "ranked_models",
                "reason",
                "image_count",
                "low_confidence",
                "elimination_stripped_from_ranking",
                "fleet_decomposition",
            ):
                if key in self.details:
                    compact[key] = self.details[key]
            if compact:
                out["details"] = compact
        return out


@dataclass
class OrchestrationTrace:
    mode: OrchestrationMode = OrchestrationMode.PRODUCTION
    stages: List[StageRecord] = field(default_factory=list)
    overall_confidence: float = 1.0
    low_confidence: bool = False
    decision_source: str = ""
    failures: List[str] = field(default_factory=list)

    def completed_stage_names(self) -> List[str]:
        return [s.stage for s in self.stages if s.status in ("ok", "degraded")]

    def record(
        self,
        stage: str,
        status: str,
        *,
        duration_ms: float = 0.0,
        confidence: float = 1.0,
        details: Optional[Dict[str, Any]] = None,
        error: str = "",
    ) -> None:
        rec = StageRecord(
            stage=stage,
            status=status,
            duration_ms=duration_ms,
            confidence=confidence,
            details=dict(details or {}),
            error=error,
        )
        self.stages.append(rec)
        if structured_logging_enabled():
            logger.info(
                "orchestration stage=%s status=%s duration_ms=%.1f confidence=%.2f%s",
                stage,
                status,
                duration_ms,
                confidence,
                f" error={error}" if error else "",
                extra={"orchestration": rec.to_dict(verbose=True)},
            )

    def to_dict(self) -> Dict[str, Any]:
        verbose = self.mode == OrchestrationMode.DEBUG
        return {
            "mode": self.mode.value,
            "decision_source": self.decision_source,
            "stages": [s.to_dict(verbose=verbose) for s in self.stages],
            "stages_completed": self.completed_stage_names(),
            "overall_confidence": round(self.overall_confidence, 3),
            "low_confidence": self.low_confidence,
            "failures": list(self.failures),
            "pipeline_order": list(ORCHESTRATION_STAGES),
        }


class StageRunner:
    """Context manager for timed stage execution with fail-safe status."""

    def __init__(self, trace: OrchestrationTrace, stage: str) -> None:
        self._trace = trace
        self._stage = stage
        self._start = 0.0
        self._finished = False
        self.confidence = 1.0
        self.details: Dict[str, Any] = {}

    def __enter__(self) -> "StageRunner":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._finished:
            return False
        duration_ms = (time.perf_counter() - self._start) * 1000.0
        if exc_type is None:
            self._trace.record(
                self._stage,
                "ok",
                duration_ms=duration_ms,
                confidence=self.confidence,
                details=self.details,
            )
            return False
        self._trace.record(
            self._stage,
            "failed",
            duration_ms=duration_ms,
            confidence=0.0,
            details=self.details,
            error=str(exc)[:500],
        )
        self._trace.failures.append(f"{self._stage}: {exc}")
        return False  # propagate — caller decides fail-safe vs abort

    def skip(self, reason: str, *, confidence: float = 1.0) -> None:
        if self._finished:
            return
        duration_ms = (time.perf_counter() - self._start) * 1000.0
        self._trace.record(
            self._stage,
            "skipped",
            duration_ms=duration_ms,
            confidence=confidence,
            details={"reason": reason},
        )
        self._finished = True

    def degraded(self, reason: str, *, confidence: float = 0.5) -> None:
        if self._finished:
            return
        duration_ms = (time.perf_counter() - self._start) * 1000.0
        self._trace.record(
            self._stage,
            "degraded",
            duration_ms=duration_ms,
            confidence=confidence,
            details={"reason": reason},
        )
        self._finished = True
