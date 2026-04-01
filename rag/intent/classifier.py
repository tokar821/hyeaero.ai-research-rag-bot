"""
Intent classification — delegates to :mod:`rag.intent.aviation_classifier` then maps to
coarse :class:`~rag.intent.schemas.ConsultantIntent` for retrieval policy.
"""

from __future__ import annotations

from typing import List, Optional

from rag.intent.aviation_classifier import (
    aviation_to_consultant_coarse,
    classify_aviation_intent_detailed,
)
from rag.intent.schemas import ConsultantIntent, IntentClassification, query_kind_from_aviation_intent


def classify_consultant_intent(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> IntentClassification:
    """
    Full pipeline intent: fine ``aviation_intent`` + coarse ``primary`` for ranking / SQL gates,
    plus ``query_kind`` (mission / spec / comparison / ownership / market / listings / general).
    """
    av = classify_aviation_intent_detailed(query, history)
    primary = aviation_to_consultant_coarse(av.intent)
    return IntentClassification(
        primary=primary,
        source=av.source,
        confidence=av.confidence,
        notes=av.notes,
        aviation_intent=av.intent,
        query_kind=query_kind_from_aviation_intent(av.intent),
    )
