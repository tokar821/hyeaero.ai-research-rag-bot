"""
Ask Consultant pipeline — structured to match the intended RAG architecture::

    User Query
         │
    Entity Detection   ← :func:`summarize_consultant_entities`
         │
    Query Router       ← :func:`load_consultant_pipeline_config` + Tavily gate
         │
    SQL + Pinecone Retrieval   ← :mod:`rag.consultant_retrieval` (Phly/market SQL, ``_retrieve_multi``)
         │
    Reranker           ← ``SemanticRerankerService`` inside :meth:`RAGQueryService.retrieve`
         │
    Context Builder    ← :func:`build_consultant_llm_context`
         │
    LLM
         │
    Answer

End-to-end orchestration lives in :func:`rag.consultant_retrieval.run_consultant_retrieval_bundle`.
This module holds **router config**, **entity summary**, and **context assembly** helpers.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


def _env_truthy(key: str) -> bool:
    return (os.getenv(key) or "").strip().lower() in ("1", "true", "yes")


@dataclass
class ConsultantEntityDetection:
    """Output of the entity-detection stage (tokens + counts for SQL / Tavily anchoring)."""

    lookup_tokens: List[str]
    phlydata_row_count: int
    faa_lookup_token_count: int

    def asdict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["lookup_tokens"] = d["lookup_tokens"][:48]
        return d


def summarize_consultant_entities(
    query: str,
    history: Optional[List[Dict[str, str]]],
    phly_meta: Dict[str, Any],
    phly_rows: List[Dict[str, Any]],
) -> ConsultantEntityDetection:
    from rag.phlydata_consultant_lookup import consultant_merge_lookup_tokens

    flt = phly_meta.get("faa_lookup_tokens")
    flist = flt if isinstance(flt, list) else []
    tokens = consultant_merge_lookup_tokens(query, history, flist)
    return ConsultantEntityDetection(
        lookup_tokens=list(tokens),
        phlydata_row_count=len(phly_rows or []),
        faa_lookup_token_count=len(flist),
    )


@dataclass
class ConsultantPipelineConfig:
    """Query-router tuning from env (latency, Tavily, RAG limits)."""

    low_latency: bool
    fast_retrieval: bool
    skip_expand: bool
    single_tavily_pass: bool
    strict_market_sql: bool
    tavily_per_pass: int
    max_rag_variants: int
    enrich_rag_max: int
    rag_max_chunks: int
    tavily_timeout: float
    intent_model: str

    def query_router_snapshot(self) -> Dict[str, Any]:
        return {
            "low_latency": self.low_latency,
            "fast_retrieval": self.fast_retrieval,
            "skip_query_expand": self.skip_expand,
            "single_tavily_pass": self.single_tavily_pass,
            "strict_market_sql": self.strict_market_sql,
            "tavily_per_pass": self.tavily_per_pass,
            "max_rag_variants": self.max_rag_variants,
            "enrich_rag_max": self.enrich_rag_max,
            "rag_max_chunks": self.rag_max_chunks,
            "tavily_timeout_sec": self.tavily_timeout,
        }


def load_consultant_pipeline_config(default_chat_model: str) -> ConsultantPipelineConfig:
    """Load router config from environment (consultant fast paths + retrieval caps)."""
    low_latency = _env_truthy("CONSULTANT_LOW_LATENCY")
    fast_retrieval = _env_truthy("CONSULTANT_FAST_RETRIEVAL") or low_latency
    skip_expand = _env_truthy("CONSULTANT_SKIP_QUERY_EXPAND") or low_latency
    single_tavily_pass = _env_truthy("CONSULTANT_TAVILY_SINGLE_PASS") or fast_retrieval
    strict_market_sql = _env_truthy("CONSULTANT_MARKET_SQL_STRICT")

    try:
        tavily_per_pass = int((os.getenv("CONSULTANT_TAVILY_RESULTS_PER_PASS") or "8").strip())
        tavily_per_pass = max(4, min(10, tavily_per_pass))
    except ValueError:
        tavily_per_pass = 8
    if fast_retrieval:
        tavily_per_pass = min(tavily_per_pass, 6)

    try:
        max_rag_variants = int((os.getenv("CONSULTANT_RAG_QUERY_VARIANTS") or "5").strip())
        max_rag_variants = max(1, min(8, max_rag_variants))
    except ValueError:
        max_rag_variants = 5
    if fast_retrieval:
        max_rag_variants = min(max_rag_variants, 3)

    try:
        enrich_rag_max = int((os.getenv("CONSULTANT_RAG_ENRICH_MAX") or "8").strip())
        enrich_rag_max = max(3, min(12, enrich_rag_max))
    except ValueError:
        enrich_rag_max = 8
    if fast_retrieval:
        enrich_rag_max = min(enrich_rag_max, 5)

    try:
        rag_max_chunks = int((os.getenv("CONSULTANT_RAG_MAX_CHUNKS") or "18").strip())
        rag_max_chunks = max(8, min(24, rag_max_chunks))
    except ValueError:
        rag_max_chunks = 18
    if fast_retrieval:
        rag_max_chunks = min(rag_max_chunks, 14)

    try:
        tavily_timeout = float((os.getenv("CONSULTANT_TAVILY_TIMEOUT_SEC") or "28").strip())
        tavily_timeout = max(8.0, min(60.0, tavily_timeout))
    except ValueError:
        tavily_timeout = 28.0
    if low_latency:
        tavily_timeout = min(tavily_timeout, 20.0)

    intent_model = (os.getenv("CONSULTANT_INTENT_MODEL") or default_chat_model or "").strip()

    return ConsultantPipelineConfig(
        low_latency=low_latency,
        fast_retrieval=fast_retrieval,
        skip_expand=skip_expand,
        single_tavily_pass=single_tavily_pass,
        strict_market_sql=strict_market_sql,
        tavily_per_pass=tavily_per_pass,
        max_rag_variants=max_rag_variants,
        enrich_rag_max=enrich_rag_max,
        rag_max_chunks=rag_max_chunks,
        tavily_timeout=tavily_timeout,
        intent_model=intent_model,
    )


def build_consultant_llm_context(
    *,
    phly_authority: str,
    market_block: str,
    tavily_block: str,
    rag_results: List[Dict[str, Any]],
    max_context_chars: int,
) -> Tuple[str, List[str]]:
    """
    Context-builder stage: **SQL / authority first**, then web (Tavily), then **reranked** Pinecone rows.
    Order reduces hallucination risk vs. putting opaque chunks first.
    """
    parts: List[str] = []
    total = 0
    sep = 20

    def append_block(text: str) -> None:
        nonlocal total
        chunk = (text or "").strip()
        if not chunk:
            return
        if total + len(chunk) + sep > max_context_chars:
            chunk = chunk[: max(0, max_context_chars - total - sep)]
        if not chunk.strip():
            return
        parts.append(chunk)
        total += len(chunk) + sep

    append_block(phly_authority)
    append_block(market_block)
    append_block(tavily_block)

    for r in rag_results:
        text = (r.get("full_context") or r.get("chunk_text") or "").strip()
        if not text:
            continue
        if total + len(text) + sep > max_context_chars:
            text = text[: max(0, max_context_chars - total - sep)]
        if text:
            parts.append(text)
            total += len(text) + sep
        if total >= max_context_chars:
            break

    context = "\n\n---\n\n".join(parts) if parts else ""
    return context, parts
