"""Thresholds and user-facing copy for strict aircraft image verification."""

from __future__ import annotations

import os

# Minimum composite confidence (0–1) to show an image in the gallery.
MIN_VERIFICATION_CONFIDENCE = 0.7

VERIFIED_FAILURE_MESSAGE = "No verified images found for this exact aircraft."

# Legacy / alternate copy used in some orchestrator paths — normalize to the strict message.
LEGACY_FAILURE_MESSAGES = frozenset(
    {
        VERIFIED_FAILURE_MESSAGE,
        "No verified images found for this aircraft.",
        "I cannot find verified images for this specific request.",
        "I couldn't find verified images for this specific aircraft.",
        "No verified images met quality and relevance thresholds.",
    }
)


def strict_image_verification_enabled() -> bool:
    """Default ON — set ``AIRCRAFT_IMAGE_VERIFICATION_STRICT=0`` to disable the final gate."""
    return (os.getenv("AIRCRAFT_IMAGE_VERIFICATION_STRICT") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def verification_min_confidence() -> float:
    try:
        return max(0.5, min(0.95, float((os.getenv("AIRCRAFT_IMAGE_MIN_CONFIDENCE") or "0.7").strip())))
    except ValueError:
        return MIN_VERIFICATION_CONFIDENCE
