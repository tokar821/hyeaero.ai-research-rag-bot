"""Structured entity-detection output for consultant analytics and routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConsultantEntityDetection:
    """Output of the entity-detection stage (tokens + counts for SQL / Tavily anchoring).

    ``intent_primary`` is coarse routing; ``aviation_intent`` is the fine Ask Consultant label.
    """

    lookup_tokens: List[str]
    phlydata_row_count: int
    faa_lookup_token_count: int
    intent_primary: Optional[str] = None
    aviation_intent: Optional[str] = None
    intent_source: Optional[str] = None
    registry_sql_enabled: bool = True
    tail_candidates: List[str] = field(default_factory=list)
    serial_or_model_tokens: List[str] = field(default_factory=list)
    aviation_entities: Optional[Dict[str, List[str]]] = None
    """Output of :func:`~rag.entities.aviation_identifiers.detect_aviation_entities`."""

    def asdict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["lookup_tokens"] = d["lookup_tokens"][:48]
        d["tail_candidates"] = d["tail_candidates"][:24]
        d["serial_or_model_tokens"] = d["serial_or_model_tokens"][:32]
        return d
