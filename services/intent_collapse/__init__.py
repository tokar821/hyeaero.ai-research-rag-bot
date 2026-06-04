"""Phase 45 — unified pre-reasoning intent collapse."""

from services.intent_collapse.intent_collapse_engine import apply_intent_collapse
from services.intent_collapse.canonical_intent_frame import CanonicalIntentFrame, PrimaryIntent

__all__ = ["apply_intent_collapse", "CanonicalIntentFrame", "PrimaryIntent"]
