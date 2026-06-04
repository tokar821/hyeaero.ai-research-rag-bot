"""Executive intelligence synthesis (Phase 27)."""

from services.synthesis.executive_intelligence_synthesis_engine import (
    ExecutiveIntelligenceSynthesis,
    attach_executive_synthesis_if_enabled,
    build_executive_synthesis,
    executive_synthesis_enabled,
)

__all__ = [
    "ExecutiveIntelligenceSynthesis",
    "attach_executive_synthesis_if_enabled",
    "build_executive_synthesis",
    "executive_synthesis_enabled",
]
