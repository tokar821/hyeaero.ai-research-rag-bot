"""
Query archetype detection — shared by router hints and presentation intelligence.
"""

from __future__ import annotations

import re

_REPLACEMENT_RE = re.compile(
    r"\b(?:"
    r"credible\s+replacements?|"
    r"alternatives?\s+to|"
    r"alternative\s+to|"
    r"lower[- ]cost\s+alternative|"
    r"down[- ]market|"
    r"fall\s+too\s+far\s+down[- ]market|"
    r"want\s+(?:a\s+)?lower[- ]cost\s+alternative\s+to|"
    r"replace\s+(?:a|an|our|the)\s+"
    r")\b",
    re.I,
)

_COG_STRATEGIC_RE = re.compile(
    r"\b(?:"
    r"center[- ]of[- ]gravity|"
    r"procurement\s+center[- ]of[- ]gravity|"
    r"how\s+should\s+(?:the\s+)?(?:actual\s+)?procurement|"
    r"how\s+should\s+.*\s+be\s+represented|"
    r"operationally\s+rational|"
    r"network\s+being\s+distorted|"
    r"edge[- ]case\s+mission|"
    r"leadership\s+keeps\s+fixating|"
    r"strategically\s+important|"
    r"optimized\s+around\s+tokyo|"
    r"longest\s+route|"
    r"should\s+(?:hong\s+kong|aspen|geneva)\b.*\b(?:influence|drive|materially)|"
    r"planning\s+distortion\s+risks?|"
    r"materially\s+influence\s+fleet\s+sizing"
    r")\b",
    re.I,
)

_BROKER_ACQUISITION_RE = re.compile(
    r"\b(?:"
    r"broker[- ]style|"
    r"acquisition\s+summary|"
    r"acquisition\s+budget|"
    r"under\s+\$?\d+\s*m(?:illion)?\s+acquisition|"
    r"professional\s+aircraft\s+brokerage"
    r")\b",
    re.I,
)

_DOWNSIZING_STRATEGIC_RE = re.compile(
    r"\b(?:downsiz(?:e|ing)|step\s+down)\b.*\b(?:operational\s+sense|dispatch\s+credibility|make\s+sense)\b",
    re.I,
)

_CONFLICT_EXPLAIN_RE = re.compile(
    r"\b(?:"
    r"why\s+do\s+(?:these\s+)?requirements\s+start\s+conflicting|"
    r"(?:what\s+)?compromises?\s+become\s+unavoidable|"
    r"what\s+if\s+leadership\s+(?:suddenly\s+)?insists|"
    r"guaranteed\s+nonstop\s+(?:singapore|tokyo|dubai|asia)"
    r")\b",
    re.I,
)

_OWNERSHIP_STRUCTURE_RE = re.compile(
    r"\b(?:"
    r"ownership\s+structure\s+makes?\s+(?:the\s+)?most\s+sense|"
    r"fractional\s+ownership\s+(?:now\s+)?make\s+more\s+sense|"
    r"fractional\s+(?:vs\.?|versus)\s+full\s+ownership|"
    r"full\s+ownership\s+(?:vs\.?|versus)\s+fractional|"
    r"does\s+fractional\s+ownership\b|"
    r"fractional\s+ownership\s+stop\s+making\s+sense|"
    r"at\s+what\s+utilization\s+level\s+does\s+fractional"
    r")\b",
    re.I,
)

_IMAGE_EXPLICIT_RE = re.compile(
    r"\b(?:"
    r"show\s+(?:me\s+)?(?:only\s+)?(?:verified\s+)?(?:exterior[- ]only\s+)?(?:exterior\s+)?images?|"
    r"show\s+(?:only\s+)?verified\s+images?\s+of\s+the|"
    r"verified\s+exterior[- ]only\s+images?|"
    r"verified\s+exterior\s+images?|"
    r"find\s+verified\s+images?|"
    r"(?:cockpit|exterior)\s+and\s+exterior|"
    r"tail\s+number|"
    r"n\d{1,5}[a-z]{0,2}\b|"
    r"vp-[a-z]{3}\b"
    r")\b",
    re.I,
)


def is_replacement_query(query: str) -> bool:
    return bool(_REPLACEMENT_RE.search(query or ""))


def is_cog_strategic_query(query: str) -> bool:
    return bool(_COG_STRATEGIC_RE.search(query or ""))


def is_broker_acquisition_query(query: str) -> bool:
    return bool(_BROKER_ACQUISITION_RE.search(query or ""))


def is_downsizing_strategic_query(query: str) -> bool:
    return bool(_DOWNSIZING_STRATEGIC_RE.search(query or ""))


def is_conflict_explain_query(query: str) -> bool:
    return bool(_CONFLICT_EXPLAIN_RE.search(query or ""))


def is_ownership_structure_query(query: str) -> bool:
    return bool(_OWNERSHIP_STRUCTURE_RE.search(query or ""))


def is_image_request_query(query: str) -> bool:
    return bool(_IMAGE_EXPLICIT_RE.search(query or ""))


def is_mission_evolution_query(query: str) -> bool:
    try:
        from services.conversation_continuity.mission_evolution import (
            is_mission_evolution_query as _evo,
        )

        return _evo(query)
    except Exception:
        return False


__all__ = [
    "is_replacement_query",
    "is_cog_strategic_query",
    "is_broker_acquisition_query",
    "is_downsizing_strategic_query",
    "is_conflict_explain_query",
    "is_ownership_structure_query",
    "is_image_request_query",
    "is_mission_evolution_query",
]
