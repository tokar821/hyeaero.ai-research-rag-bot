"""
Comparison output facade — delegates to Comparison v2 structured layer.

Legacy markdown table builders are bypassed for explicit_comparison.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from services.comparison.comparison_pipeline_v2 import run_comparison_v2
from services.consultant.mission_state import MissionState

_INSUFFICIENT_JSON = (
    '{\n  "mode": "explicit_comparison",\n'
    '  "status": "INSUFFICIENT_DATA",\n'
    '  "reason": "missing canonical aircraft set"\n}'
)


def format_comparison_response(
    *,
    query: str,
    mission: MissionState,
    compare_models: Sequence[str],
    data_used: Optional[Dict[str, Any]] = None,
) -> str:
    """Render explicit_comparison as strict Comparison v2 JSON only."""
    return run_comparison_v2(
        query=query,
        mission=mission,
        compare_models=compare_models,
        data_used=data_used,
        mode="explicit_comparison",
    )


# Backward-compatible constant for tests/callers expecting a string marker
_INSUFFICIENT = _INSUFFICIENT_JSON


__all__ = ["format_comparison_response", "_INSUFFICIENT"]
