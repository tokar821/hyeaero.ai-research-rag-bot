"""Response mode router schemas."""

from __future__ import annotations

from enum import Enum
from typing import Literal, TypedDict


class ResponseMode(str, Enum):
    IMAGE_SHOWCASE = "image_showcase"
    ADVISORY = "advisory"
    FOLLOWUP_CONTINUATION = "followup_continuation"
    COMPARISON_MODE = "comparison_mode"
    EDUCATIONAL_MODE = "educational_mode"
    TAIL_SPECIFIC = "tail_specific"
    INVALID_SANITY = "invalid_sanity"


Verbosity = Literal["minimal", "short", "detailed"]


class ResponseModeRouterResult(TypedDict):
    mode: str
    reason: str
    visual_priority: bool
    verbosity: Verbosity
    inherit_context: bool
    forbid_urls_in_text: bool
    max_sentences_hint: int
