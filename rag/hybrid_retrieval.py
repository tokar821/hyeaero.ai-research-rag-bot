"""
Hybrid retrieval routing for Ask Consultant: structured Phly / listing SQL first,
Pinecone vector retrieval for semantic missions, comparisons, and general knowledge.

Structured path uses existing ``phlydata_aircraft``, ``faa_master``, ``aircraft_listings``,
and related SQL (not hypothetical ``aircraft_registrations`` tables).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from rag.consultant_fine_intent import ConsultantFineIntent, ConsultantFineIntentResult


class HybridRetrievalQueryKind(str, Enum):
    TAIL_NUMBER_LOOKUP = "tail_number_lookup"
    SERIAL_NUMBER_LOOKUP = "serial_number_lookup"
    AIRCRAFT_LISTING_QUERY = "aircraft_listing_query"
    AIRCRAFT_SPEC_QUERY = "aircraft_spec_query"
    MISSION_QUESTION = "mission_question"
    COMPARISON_QUESTION = "comparison_question"
    GENERAL_AVIATION_KNOWLEDGE = "general_aviation_knowledge"
    OWNERSHIP_LOOKUP = "ownership_lookup"
    AIRCRAFT_PRICE_LOOKUP = "aircraft_price_lookup"


def _env_truthy(key: str) -> bool:
    return (os.getenv(key) or "").strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class HybridRetrievalPlan:
    """Controls structured-SQL priority vs Pinecone depth for one consultant turn."""

    kind: HybridRetrievalQueryKind
    vector_primary: bool
    """When False, Pinecone is skipped or reduced once structured SQL context is non-empty."""

    @property
    def prefers_structured_sql(self) -> bool:
        return not self.vector_primary

    def max_vector_chunks(self, base_max: int, sql_context_nonempty: bool) -> int:
        """
        Return cap for ``max_results_total`` passed to :meth:`RAGQueryService._retrieve_multi`.

        - Vector-primary kinds: always ``base_max``.
        - Structured kinds: ``0`` when SQL already returned context (override via
          ``HYBRID_VECTOR_SUPPLEMENT_MAX``); full ``base_max`` when SQL is empty (fallback).
        """
        if _env_truthy("HYBRID_DISABLE_VECTOR_SUPPRESS"):
            return base_max
        if self.vector_primary:
            return base_max
        if not sql_context_nonempty:
            return base_max
        try:
            sup = int((os.getenv("HYBRID_VECTOR_SUPPLEMENT_MAX") or "0").strip())
        except ValueError:
            sup = 0
        return max(0, min(sup, base_max))


_SERIAL_WORD = re.compile(r"\b(serial|s/?n|msn)\b", re.I)
_LISTING_WORD = re.compile(
    r"\b(listing|controller|exchange|aircraftpost|marketplace|on controller)\b",
    re.I,
)


def classify_hybrid_retrieval(
    query: str,
    fine: ConsultantFineIntentResult,
    strict_tails: List[str],
) -> HybridRetrievalPlan:
    """
    Map fine intent + tail evidence to a hybrid query kind and vector policy.

    Structured-first (Phly / listings SQL authoritative; suppress vector when SQL hits):
    tail / serial / listing / ownership / price.

    Vector-primary (Pinecone for long-form context):
    mission, comparison, model capability / general specs without a registration anchor,
    general aviation knowledge.
    """
    q = query or ""
    ql = q.lower()
    has_tail = bool(strict_tails)
    fi = fine.intent

    if fi == ConsultantFineIntent.AIRCRAFT_COMPARISON:
        return HybridRetrievalPlan(HybridRetrievalQueryKind.COMPARISON_QUESTION, vector_primary=True)

    if fi == ConsultantFineIntent.AVIATION_MISSION:
        return HybridRetrievalPlan(HybridRetrievalQueryKind.MISSION_QUESTION, vector_primary=True)

    if fi == ConsultantFineIntent.GENERAL_QUESTION:
        return HybridRetrievalPlan(HybridRetrievalQueryKind.GENERAL_AVIATION_KNOWLEDGE, vector_primary=True)

    if fi == ConsultantFineIntent.AIRCRAFT_RECOMMENDATION:
        return HybridRetrievalPlan(HybridRetrievalQueryKind.MISSION_QUESTION, vector_primary=True)

    if fi == ConsultantFineIntent.OWNERSHIP_LOOKUP:
        if has_tail:
            return HybridRetrievalPlan(HybridRetrievalQueryKind.OWNERSHIP_LOOKUP, vector_primary=False)
        return HybridRetrievalPlan(HybridRetrievalQueryKind.GENERAL_AVIATION_KNOWLEDGE, vector_primary=True)

    if fi == ConsultantFineIntent.MARKET_QUESTION:
        if _LISTING_WORD.search(ql):
            return HybridRetrievalPlan(HybridRetrievalQueryKind.AIRCRAFT_LISTING_QUERY, vector_primary=False)
        return HybridRetrievalPlan(HybridRetrievalQueryKind.AIRCRAFT_PRICE_LOOKUP, vector_primary=False)

    if fi == ConsultantFineIntent.AIRCRAFT_SPECS:
        if _SERIAL_WORD.search(ql) and not has_tail:
            return HybridRetrievalPlan(HybridRetrievalQueryKind.SERIAL_NUMBER_LOOKUP, vector_primary=False)
        if has_tail:
            return HybridRetrievalPlan(HybridRetrievalQueryKind.TAIL_NUMBER_LOOKUP, vector_primary=False)
        if any(
            k in ql
            for k in (
                "capability",
                "capabilities",
                "suitable for",
                "good for",
                "explain what a",
                "what is a citation",
                "overview of the",
            )
        ):
            return HybridRetrievalPlan(HybridRetrievalQueryKind.AIRCRAFT_SPEC_QUERY, vector_primary=True)
        return HybridRetrievalPlan(HybridRetrievalQueryKind.AIRCRAFT_SPEC_QUERY, vector_primary=True)

    return HybridRetrievalPlan(HybridRetrievalQueryKind.GENERAL_AVIATION_KNOWLEDGE, vector_primary=True)


def build_hybrid_phly_structured_context_block(phly_rows: List[Dict[str, Any]]) -> str:
    """Compact, LLM-friendly lines from Phly rows (authoritative when present)."""
    if not phly_rows:
        return ""
    lines: List[str] = [
        "[HYBRID — Structured aircraft record (authoritative)]",
        "",
    ]
    for i, r in enumerate(phly_rows[:5], 1):
        lines.append(f"Aircraft record {i}:")
        reg = (r.get("registration_number") or "").strip()
        sn = (r.get("serial_number") or "").strip()
        mfr = (r.get("manufacturer") or "").strip()
        mdl = (r.get("model") or "").strip()
        yr = r.get("manufacturer_year")
        if yr is None:
            yr = r.get("delivery_year")
        cat = (r.get("category") or "").strip()
        status = (r.get("aircraft_status") or "").strip()
        ask = r.get("ask_price")
        take = r.get("take_price")
        sold = r.get("sold_price")
        rc = (r.get("registration_country") or "").strip()
        bc = (r.get("based_country") or "").strip()
        broker = (r.get("seller_broker") or "").strip()
        seller = (r.get("seller") or "").strip()
        if reg:
            lines.append(f"  Tail number: {reg}")
        if sn:
            lines.append(f"  Serial: {sn}")
        mm = " ".join(x for x in (mfr, mdl) if x).strip()
        if mm:
            lines.append(f"  Aircraft type: {mm}")
        if yr is not None:
            lines.append(f"  Year: {yr}")
        if cat:
            lines.append(f"  Category: {cat}")
        if status:
            lines.append(f"  Status: {status}")
        if ask is not None:
            lines.append(f"  Ask price: {ask}")
        if take is not None:
            lines.append(f"  Take price: {take}")
        if sold is not None:
            lines.append(f"  Sold price: {sold}")
        if seller:
            lines.append(f"  Owner / seller (export field): {seller}")
        if broker:
            lines.append(f"  Broker: {broker}")
        if bc or rc:
            lines.append(f"  Location / basis: {', '.join(x for x in (bc, rc) if x)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def prepend_hybrid_structured_context(
    phly_authority: str,
    phly_rows: Optional[List[Dict[str, Any]]],
    plan: HybridRetrievalPlan,
) -> str:
    if not phly_rows or not plan.prefers_structured_sql:
        return phly_authority or ""
    block = build_hybrid_phly_structured_context_block(phly_rows)
    if not block:
        return phly_authority or ""
    base = (phly_authority or "").strip()
    if not base:
        return block
    return f"{block}\n\n{base}"
