"""
Validation entrypoints for mission extraction payloads.
"""

from __future__ import annotations

import json
from typing import Any, Union

from pydantic import ValidationError

from services.mission_extraction.schema import MissionExtractionResult, validate_extraction_payload


def validate_extraction_json(raw_json: str) -> MissionExtractionResult:
    """Parse and validate a JSON string against the mission extraction schema."""
    try:
        data: Any = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    return validate_extraction_payload(data)


def safe_validate_extraction(
    data: Union[dict, MissionExtractionResult],
) -> tuple[MissionExtractionResult | None, str | None]:
    """Return ``(result, error_message)`` — does not raise."""
    try:
        return validate_extraction_payload(data), None
    except ValidationError as exc:
        return None, str(exc)
