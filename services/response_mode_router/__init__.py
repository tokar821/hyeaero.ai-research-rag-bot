"""
Response Mode Router — specialized answer orchestration per turn.

Public API: :func:`route_response_mode`, :func:`response_mode_prompt_suffix`.
"""

from __future__ import annotations

from .enforce import enforce_from_data_used, enforce_mode_on_answer
from .prompts import response_mode_prompt_suffix
from .router import route_response_mode, router_result_json
from .schemas import ResponseMode, ResponseModeRouterResult

__all__ = [
    "ResponseMode",
    "ResponseModeRouterResult",
    "enforce_from_data_used",
    "enforce_mode_on_answer",
    "response_mode_prompt_suffix",
    "route_response_mode",
    "router_result_json",
]
