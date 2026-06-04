"""
Phase 56 — fact flow observability (measurement only; does not change answers).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from services.broker_execution.tail_fact_renderer import (
    count_rendered_tail_facts,
    select_tail_facts,
)


def _count_retrieved_documents(data_used: dict) -> int:
    n = 0
    chunks = data_used.get("rag_chunks") or data_used.get("retrieved_chunks") or []
    if isinstance(chunks, list):
        n += len(chunks)
    if data_used.get("phly_authority") or data_used.get("phlydata_rows") or data_used.get("phly_rows"):
        n += 1
    meta = data_used.get("phly_meta") or {}
    if isinstance(meta, dict):
        n += int(meta.get("faa_master_owner_rows") or 0)
    if data_used.get("faa_master_row") or data_used.get("tail_facts"):
        n += 1
    return n


def _count_retrieved_entities(data_used: dict) -> int:
    found: Set[str] = set()
    for key in ("aviation_entities", "entity_detection"):
        blob = data_used.get(key)
        if isinstance(blob, dict):
            for m in blob.get("models") or blob.get("aircraft") or []:
                if m:
                    found.add(str(m).strip().lower())
            for t in blob.get("registrations") or blob.get("tails") or []:
                if t:
                    found.add(str(t).strip().upper())
    facts = data_used.get("tail_selected_facts") or []
    if isinstance(facts, list):
        found.add(str(data_used.get("tail_registration") or "").upper())
        for f in facts:
            if isinstance(f, dict) and f.get("value"):
                found.add(str(f["value"])[:40])
    return len(found)


def _count_selected_facts(data_used: dict) -> int:
    facts = data_used.get("tail_selected_facts") or []
    if isinstance(facts, list) and facts:
        return len(facts)
    listing = data_used.get("listing_parse_audit") or {}
    if isinstance(listing, dict) and listing.get("parse_success"):
        return sum(
            1
            for k in ("detected_model", "detected_year", "detected_price")
            if listing.get(k) is not None
        )
    br = data_used.get("broker_reasoning") or {}
    if isinstance(br, dict):
        comp = br.get("comparison") or {}
        if isinstance(comp, dict) and comp.get("models"):
            return len(comp.get("models") or [])
    return 0


def _count_rendered_facts(answer: str, data_used: dict) -> int:
    facts = data_used.get("tail_selected_facts") or []
    if isinstance(facts, list) and facts:
        return count_rendered_tail_facts(answer, facts)
    n = 0
    listing = data_used.get("listing_parse_audit") or {}
    if isinstance(listing, dict) and listing.get("parse_success"):
        low = (answer or "").lower()
        model = str(listing.get("detected_model") or "").lower()
        if model and model in low:
            n += 1
        year = listing.get("detected_year")
        if year is not None and str(year) in (answer or ""):
            n += 1
        price = listing.get("detected_price")
        if price is not None and re.search(r"\d", answer or ""):
            n += 1
    br = data_used.get("broker_reasoning") or {}
    if isinstance(br, dict):
        comp = br.get("comparison") or {}
        models = comp.get("models") if isinstance(comp, dict) else []
        low = (answer or "").lower()
        for m in models or []:
            if str(m).lower() in low:
                n += 1
        if re.search(r"(?is)\b(?:range|cabin|operating\s+cost|liquidity)\b", answer or ""):
            n += max(n, 1)
    return n


def build_fact_flow(query: str, answer: str, data_used: dict) -> Dict[str, Any]:
    selected = _count_selected_facts(data_used)
    rendered = _count_rendered_facts(answer or "", data_used)
    retrieved_entities = _count_retrieved_entities(data_used)
    fallback_used = bool(
        re.search(r"(?is)send\s+me\s+(?:the\s+)?listing\s+package", answer or "")
        and selected > 0
    ) or bool(
        data_used.get("tail_fallback_used")
        and selected > 0
    )
    return {
        "query": (query or "").strip(),
        "retrieved_documents": _count_retrieved_documents(data_used),
        "retrieved_entities": retrieved_entities,
        "selected_facts": selected,
        "rendered_facts": rendered,
        "fallback_used": fallback_used,
    }


def attach_fact_flow(query: str, answer: str, data_used: dict) -> Dict[str, Any]:
    """Store fact_flow on data_used — observability only."""
    if not isinstance(data_used, dict):
        return {}
    flow = build_fact_flow(query, answer, data_used)
    data_used["fact_flow"] = flow
    return flow


__all__ = ["attach_fact_flow", "build_fact_flow"]
