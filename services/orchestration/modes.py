"""Debug vs production orchestration modes."""

from __future__ import annotations

import os
from enum import Enum


class OrchestrationMode(str, Enum):
    PRODUCTION = "production"
    DEBUG = "debug"


def orchestration_mode() -> OrchestrationMode:
    """
    ``CONSULTANT_ORCHESTRATION_MODE=debug`` enables verbose traces in ``data_used``.

    Default is production (sanitized traces, minimal client exposure).
    """
    raw = (os.getenv("CONSULTANT_ORCHESTRATION_MODE") or "production").strip().lower()
    if raw in ("debug", "dev", "verbose"):
        return OrchestrationMode.DEBUG
    return OrchestrationMode.PRODUCTION


def orchestration_enabled() -> bool:
    """Master switch — set ``CONSULTANT_ORCHESTRATION=0`` to use legacy inline paths."""
    return (os.getenv("CONSULTANT_ORCHESTRATION") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def structured_logging_enabled() -> bool:
    return (os.getenv("CONSULTANT_ORCHESTRATION_LOG") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
