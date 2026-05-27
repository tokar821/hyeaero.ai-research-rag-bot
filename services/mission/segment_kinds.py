"""Segment kind enum — shared without circular imports."""

from __future__ import annotations

from enum import Enum


class SegmentKind(str, Enum):
    DOMESTIC_EXECUTIVE = "domestic_executive"
    TRANSATLANTIC_EXECUTIVE = "transatlantic_executive"
    ULR_CONTINUATION = "ulr_continuation"
    MOUNTAIN_FIELD = "mountain_field"
    INDUSTRIAL_FIELD = "industrial_field"
    CARIBBEAN_REGIONAL = "caribbean_regional"
    PACIFIC_ULR = "pacific_ulr"


__all__ = ["SegmentKind"]
