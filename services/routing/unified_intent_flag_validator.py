"""
Unified intent flag validator — observability-only flag/path consistency checks.

Does not modify routing behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from services.routing.unified_intent_router import UnifiedExecutionPath, UnifiedIntentRoute

_PATH_FLAG_MAP = {
    UnifiedExecutionPath.AIRCRAFT_FACT: "enforce_fact",
    UnifiedExecutionPath.AIRCRAFT_MARKET_FACT: "enforce_fact",
    UnifiedExecutionPath.CAPABILITY: "enforce_capability",
    UnifiedExecutionPath.COMPARISON: "enforce_comparison",
    UnifiedExecutionPath.ALTERNATIVE: "enforce_alternative",
}


@dataclass
class FlagValidationResult:
    valid: bool
    execution_path: str
    enabled_flags: Dict[str, bool]
    expected_flag: str | None
    invalid_combinations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "execution_path": self.execution_path,
            "enabled_flags": dict(self.enabled_flags),
            "expected_flag": self.expected_flag,
            "invalid_combinations": list(self.invalid_combinations),
        }


def validate_flag_consistency(
    route: UnifiedIntentRoute,
    *,
    enforce_fact: bool = False,
    enforce_capability: bool = False,
    enforce_comparison: bool = False,
    enforce_alternative: bool = False,
) -> FlagValidationResult:
    """
    Ensure enabled enforcement flags align with ``route.execution_path``.

    Logs mismatches only — never changes routing.
    """
    flags = {
        "enforce_fact": enforce_fact,
        "enforce_capability": enforce_capability,
        "enforce_comparison": enforce_comparison,
        "enforce_alternative": enforce_alternative,
    }
    path = route.execution_path
    expected = _PATH_FLAG_MAP.get(path)
    invalid: List[str] = []

    if not any(flags.values()):
        return FlagValidationResult(
            valid=True,
            execution_path=path.value,
            enabled_flags=flags,
            expected_flag=expected,
        )

    for flag_name, enabled in flags.items():
        if not enabled:
            continue
        if expected is None and path == UnifiedExecutionPath.NONE:
            invalid.append(f"{flag_name}_enabled_for_none_path")
        elif expected and flag_name != expected:
            invalid.append(f"{flag_name}_mismatch_for_{path.value}")

    if expected and flags.get(expected) is False and any(flags.values()):
        invalid.append(f"missing_{expected}_for_{path.value}")

    return FlagValidationResult(
        valid=not invalid,
        execution_path=path.value,
        enabled_flags=flags,
        expected_flag=expected,
        invalid_combinations=invalid,
    )


__all__ = ["FlagValidationResult", "validate_flag_consistency"]
