"""Intent → retrieval policy (registry SQL, optional Pinecone filters)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from rag.intent.schemas import AviationIntent, ConsultantIntent, IntentClassification
from rag.phlydata_consultant_lookup import extract_us_registration_tail_candidates


def registry_sql_enabled_for_intent(
    classification: IntentClassification,
    query: str = "",
    history: Optional[List[Dict[str, str]]] = None,
    *,
    force_always: Optional[bool] = None,
) -> bool:
    """
    Whether to query ``faa_master`` (registration / ownership database) for this turn.

    Enabled when:
    - Aviation intent is **registration_lookup** or **serial_lookup** (tail or MSN-style identity), or
    - A U.S. **N-number** appears in the latest message or recent chat (e.g. N123AB, N1234, N98765), or
    - Coarse intent is **registration_lookup** (legacy).

    ``CONSULTANT_REGISTRY_SQL_ALWAYS=1`` forces SQL on every turn.
    """
    if force_always is None:
        force_always = (os.getenv("CONSULTANT_REGISTRY_SQL_ALWAYS") or "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
    if force_always:
        return True
    if extract_us_registration_tail_candidates(query, history):
        return True
    if classification.aviation_intent in (
        AviationIntent.REGISTRATION_LOOKUP,
        AviationIntent.SERIAL_LOOKUP,
    ):
        return True
    if classification.primary == ConsultantIntent.REGISTRATION_LOOKUP:
        return True
    return False


def pinecone_filter_for_intent(intent: ConsultantIntent) -> Optional[Dict[str, Any]]:
    """
    Optional Pinecone metadata filter derived from intent.

    Returns ``None`` to use :func:`rag.pinecone_metadata.infer_pinecone_entity_filter` inside
    :meth:`RAGQueryService.retrieve`.
    """
    del intent
    return None
