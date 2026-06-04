"""Minimal service stub for Phase 33 E2E retrieval runs.

This avoids needing live Pinecone/Postgres/OpenAI while still exercising
`run_consultant_retrieval_bundle` end-to-end for deterministic/dispatch paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


class _StubDB:
    def execute_query(self, *_args: Any, **_kwargs: Any) -> List[Dict[str, Any]]:
        return []


@dataclass
class ResponseAuditService:
    db: Any = None
    openai_api_key: str = ""
    chat_model: str = "gpt-4o-mini"

    def __post_init__(self) -> None:
        if self.db is None:
            self.db = _StubDB()

    # --- Methods used by consultant retrieval bundle ---
    def _professional_search_answer(self, _query: str) -> Optional[Dict[str, Any]]:
        return None

    def _consultant_history_snippet(self, _history: List[Dict[str, str]]) -> str:
        return ""

    def _phlydata_authority_block(self, *_args: Any, **_kwargs: Any) -> str:
        return "", {}, []

    def _retrieve_multi(self, *_args: Any, **_kwargs: Any) -> List[Dict[str, Any]]:
        # No Pinecone in CI; allow downstream to proceed with structured-only or dispatch.
        return []

    def _filter_rag_results_for_phly_aircraft(self, results: List[Dict[str, Any]], *_args: Any, **_kwargs: Any):
        return results

    def _rerank_enabled_globally(self) -> bool:
        return False

