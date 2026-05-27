"""
Strict aircraft image verification — exact tail or exact model only.

Public API:
  - :func:`verify_aircraft_image_rows` — raw SearchAPI rows
  - :func:`verify_gallery_images` — consultant gallery items
  - :const:`VERIFIED_FAILURE_MESSAGE` — user-facing empty state
"""

from services.aircraft_image_verification.constants import (
    MIN_VERIFICATION_CONFIDENCE,
    VERIFIED_FAILURE_MESSAGE,
    strict_image_verification_enabled,
    verification_min_confidence,
)
from services.aircraft_image_verification.pipeline import (
    ImageVerificationAudit,
    RejectionRecord,
    VerificationPipelineResult,
    fallback_handling,
    verify_aircraft_image_rows,
    verify_gallery_images,
)
from services.aircraft_image_verification.source_ranking import SourceTier, classify_source_tier

__all__ = [
    "MIN_VERIFICATION_CONFIDENCE",
    "VERIFIED_FAILURE_MESSAGE",
    "ImageVerificationAudit",
    "RejectionRecord",
    "SourceTier",
    "VerificationPipelineResult",
    "classify_source_tier",
    "fallback_handling",
    "strict_image_verification_enabled",
    "verification_min_confidence",
    "verify_aircraft_image_rows",
    "verify_gallery_images",
]
