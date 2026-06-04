"""
Debug-only assertions that deterministic paths never leak into LLM / kernel execution.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

HARD_DETERMINISTIC_INTENTS = frozenset({"comparison", "alternative", "buy_decision"})

_KERNEL_MARKERS = (
    "operational synthesis",
    "mission_authority_kernel",
    "viability with compromises",
    "approved shortlist",
)


class DeterministicExecutionViolation(RuntimeError):
    """Raised in debug mode when deterministic/LLM paths both executed."""


def deterministic_assertion_enabled() -> bool:
    for key in (
        "CONSULTANT_DETERMINISTIC_ASSERT",
        "DETERMINISTIC_EXECUTION_ASSERT",
    ):
        if (os.getenv(key) or "").strip().lower() in ("1", "true", "yes"):
            return True
    return False


def assert_no_llm_leak(response_trace: Optional[Dict[str, Any]] = None) -> None:
    """
    Raise when a deterministic turn also executed LLM or mission kernel synthesis.

    No-op unless ``CONSULTANT_DETERMINISTIC_ASSERT=1`` (or ``DETERMINISTIC_EXECUTION_ASSERT``).
    """
    if not deterministic_assertion_enabled():
        return

    trace = response_trace if isinstance(response_trace, dict) else {}
    violations = _collect_violations(trace)
    if not violations:
        return

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "violations": violations,
        "trace": _safe_trace(trace),
    }
    _log_violation(payload)
    raise DeterministicExecutionViolation("; ".join(violations))


def _collect_violations(trace: Dict[str, Any]) -> List[str]:
    violations: List[str] = []
    intent = str(
        trace.get("deterministic_intent")
        or trace.get("authority_dispatch_kind")
        or (trace.get("deterministic_execution") or {}).get("deterministic_intent")
        or ""
    ).lower()

    llm_executed = bool(trace.get("llm_executed"))
    pre_llm_executed = bool(trace.get("pre_llm_executed"))
    kernel_present = bool(trace.get("kernel_synthesis_present")) or _kernel_text_detected(trace)

    if intent == "comparison" and llm_executed:
        violations.append("comparison + LLM both executed")
    if intent == "alternative" and (kernel_present or _alternative_kernel_leak(trace)):
        violations.append("alternative responder + kernel synthesis present")
    if intent == "buy_decision" and kernel_present:
        violations.append("buy_decision + mission kernel both executed")
    if intent in HARD_DETERMINISTIC_INTENTS and llm_executed:
        violations.append(f"{intent} deterministic path reached LLM")
    if intent in HARD_DETERMINISTIC_INTENTS and pre_llm_executed:
        violations.append(f"{intent} deterministic path reached pre-LLM mission pipeline")

    return violations


def _alternative_kernel_leak(trace: Dict[str, Any]) -> bool:
    block = str(trace.get("pipeline_authority_block") or "")
    answer = str(trace.get("answer") or "")
    blob = f"{block}\n{answer}".lower()
    return any(marker in blob for marker in _KERNEL_MARKERS)


def _kernel_text_detected(trace: Dict[str, Any]) -> bool:
    du = trace.get("data_used")
    if not isinstance(du, dict):
        du = {}
    block = str(trace.get("pipeline_authority_block") or "")
    answer = str(trace.get("answer") or "")
    blob = f"{block}\n{answer}\n{json.dumps(du, default=str)[:4000]}".lower()
    return any(marker in blob for marker in _KERNEL_MARKERS)


def _safe_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: trace[k]
        for k in (
            "query",
            "deterministic_intent",
            "authority_dispatch_kind",
            "llm_executed",
            "pre_llm_executed",
            "kernel_synthesis_present",
            "trigger_reason",
            "final_responder",
        )
        if k in trace
    }


def _log_violation(payload: Dict[str, Any]) -> None:
    try:
        base = Path(__file__).resolve().parents[2]
        log_dir = base / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "deterministic_violation.log"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


__all__ = [
    "DeterministicExecutionViolation",
    "assert_no_llm_leak",
    "deterministic_assertion_enabled",
]
