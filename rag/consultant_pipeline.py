"""
Ask Consultant pipeline — router config and **re-exports** for the layered RAG layout::

    User Query → :mod:`rag.intent` → :mod:`rag.entities` → router (this module's config)
    → :mod:`rag.consultant_retrieval` (SQL + Pinecone + Tavily)
    → :mod:`rag.ranking` → :mod:`rag.context` → LLM (+ :mod:`rag.answer`)

End-to-end orchestration: :func:`rag.consultant_retrieval.run_consultant_retrieval_bundle`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

# Re-exports (stable import paths for tests and callers)
from rag.context import build_consultant_llm_context
from rag.entities import ConsultantEntityDetection, summarize_consultant_entities


def _env_truthy(key: str) -> bool:
    return (os.getenv(key) or "").strip().lower() in ("1", "true", "yes")


__all__ = [
    "ConsultantEntityDetection",
    "ConsultantPipelineConfig",
    "build_consultant_llm_context",
    "load_consultant_pipeline_config",
    "summarize_consultant_entities",
]


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
