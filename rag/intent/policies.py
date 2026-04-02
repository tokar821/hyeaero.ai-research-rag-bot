""" Intent → retrieval policy (registry SQL, optional Pinecone filters). """

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from rag.intent.schemas import ConsultantIntent, IntentClassification


def registry_sql_enabled_for_intent(
    classification: IntentClassification,
    query: str = "",
    history: Optional[List[Dict[str, str]]] = None,
    *,
    force_always: Optional[bool] = None,
    explicit_registry_sql: Optional[bool] = None,
) -> bool:
    """
    Whether to run ingested **FAA MASTER** (registration) SQL for this turn.

    Default: enable only when **strict civil registration** patterns appear
    (see :func:`rag.aviation_tail.find_strict_tail_candidates`). OEM / model tokens such as
    ``601``, ``604``, or ``550-0123`` do **not** enable registry SQL.

    ``explicit_registry_sql`` is set by :mod:`rag.consultant_fine_intent` tool router when present.

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
    if explicit_registry_sql is not None:
        return bool(explicit_registry_sql)

    from rag.aviation_tail import find_strict_tail_candidates

    strict_tails = find_strict_tail_candidates(query, history)
    return bool(strict_tails)


def pinecone_filter_for_intent(intent: ConsultantIntent) -> Optional[Dict[str, Any]]:
    """
    Optional Pinecone metadata filter derived from intent.

    Returns ``None`` to use :func:`rag.pinecone_metadata.infer_pinecone_entity_filter` inside
    :meth:`RAGQueryService.retrieve`.
    """
    del intent
    return None
