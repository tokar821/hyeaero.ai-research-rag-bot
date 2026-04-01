"""Merge tokens and slots for consultant entity detection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag.entities.aviation_identifiers import detect_aviation_entities
from rag.entities.schemas import ConsultantEntityDetection
from rag.phlydata_consultant_lookup import (
    consultant_merge_lookup_tokens,
    extract_phlydata_lookup_tokens,
    extract_us_registration_tail_candidates,
)


def summarize_consultant_entities(
    query: str,
    history: Optional[List[Dict[str, str]]],
    phly_meta: Dict[str, Any],
    phly_rows: List[Dict[str, Any]],
    intent_classification: Optional[Any] = None,
    *,
    registry_sql_enabled: bool = True,
) -> ConsultantEntityDetection:
    flt = phly_meta.get("faa_lookup_tokens")
    flist = flt if isinstance(flt, list) else []
    tokens = consultant_merge_lookup_tokens(query, history, flist)
    intent_primary = None
    intent_source = None
    aviation_label: Optional[str] = None
    if intent_classification is not None:
        intent_primary = intent_classification.primary.value
        intent_source = intent_classification.source
        if getattr(intent_classification, "aviation_intent", None) is not None:
            aviation_label = intent_classification.aviation_intent.value
    tails = list(dict.fromkeys(extract_us_registration_tail_candidates(query or "", history)))[:24]
    sm = list(dict.fromkeys(extract_phlydata_lookup_tokens(query or "")))[:32]
    av_entities = detect_aviation_entities(query or "", history)
    return ConsultantEntityDetection(
        lookup_tokens=list(tokens),
        phlydata_row_count=len(phly_rows or []),
        faa_lookup_token_count=len(flist),
        intent_primary=intent_primary,
        aviation_intent=aviation_label,
        intent_source=intent_source,
        registry_sql_enabled=registry_sql_enabled,
        tail_candidates=tails,
        serial_or_model_tokens=sm,
        aviation_entities=av_entities,
    )
