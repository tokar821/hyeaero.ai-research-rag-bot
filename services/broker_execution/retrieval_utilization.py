"""
Phase 55 — observability for retrieval vs answer reference (no user-facing change).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Set


def _collect_retrieved_entities(data_used: dict) -> Set[str]:
    found: Set[str] = set()

    for key in ("aviation_entities", "entity_detection"):
        blob = data_used.get(key)
        if isinstance(blob, dict):
            for m in blob.get("models") or blob.get("aircraft") or []:
                if m:
                    found.add(str(m).strip().lower())
            tails = blob.get("registrations") or blob.get("tails") or []
            for t in tails:
                if t:
                    found.add(str(t).strip().upper())

    ae = data_used.get("consultant_entity_summary")
    if isinstance(ae, dict):
        for m in ae.get("models") or []:
            if m:
                found.add(str(m).strip().lower())

    chunks = data_used.get("rag_chunks") or data_used.get("retrieved_chunks") or []
    if isinstance(chunks, list):
        for ch in chunks[:20]:
            if isinstance(ch, dict):
                text = str(ch.get("text") or ch.get("content") or "")
            else:
                text = str(ch)
            for m in re.findall(
                r"(?i)\b(?:gulfstream\s+g\d{3}|citation\s+\w+|challenger\s+\d+|falcon\s+\w+|longitude|g280)\b",
                text,
            ):
                found.add(m.lower())
            for t in re.findall(r"\bN[A-Z0-9]{3,6}\b", text):
                found.add(t.upper())

    auth = data_used.get("authority_payload") or {}
    if isinstance(auth, dict):
        model = auth.get("model") or auth.get("aircraft_model")
        if model:
            found.add(str(model).strip().lower())

    return found


def _count_referenced(answer: str, retrieved: Set[str]) -> int:
    if not answer or not retrieved:
        return 0
    low = answer.lower()
    count = 0
    for entity in retrieved:
        el = entity.lower()
        if el in low or el.replace(" ", "") in low.replace(" ", ""):
            count += 1
    return count


def attach_retrieval_utilization(answer: str, data_used: dict) -> None:
    """Mirror retrieval usage into data_used — does not alter answer text."""
    if not isinstance(data_used, dict):
        return
    retrieved = _collect_retrieved_entities(data_used)
    referenced = _count_referenced(answer or "", retrieved)
    flow = data_used.get("fact_flow") if isinstance(data_used.get("fact_flow"), dict) else {}
    selected = int(flow.get("selected_facts") or 0)
    rendered = int(flow.get("rendered_facts") or 0)
    if selected > 0:
        pct = round(100.0 * rendered / max(selected, 1), 1)
    elif retrieved:
        pct = round(100.0 * referenced / max(len(retrieved), 1), 1)
    else:
        pct = 100.0 if rendered else 0.0
    data_used["retrieved_entities_count"] = len(retrieved)
    data_used["referenced_entities_count"] = referenced
    data_used["retrieval_utilization_pct"] = pct
    data_used["retrieval_utilization_low"] = pct < 50.0 and (selected > 0 or len(retrieved) > 0)


__all__ = ["attach_retrieval_utilization"]
