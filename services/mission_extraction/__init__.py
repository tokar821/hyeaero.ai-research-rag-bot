"""
Mission Extraction Layer — requirements only, no aircraft recommendations.

Public API:
  - :func:`extract_mission_requirements`
  - :func:`extract_mission_requirements_json`
  - :class:`MissionExtractionResult`
"""

from services.mission_extraction.extractor import (
    extract_mission_requirements,
    extract_mission_requirements_json,
)
from services.mission_extraction.schema import MissionExtractionResult, validate_extraction_payload
from services.mission_extraction.validate import safe_validate_extraction, validate_extraction_json

__all__ = [
    "MissionExtractionResult",
    "extract_mission_requirements",
    "extract_mission_requirements_json",
    "validate_extraction_payload",
    "validate_extraction_json",
    "safe_validate_extraction",
]
