"""
Cross-encoder semantic reranking (BAAI bge-reranker-large) for RAG retrieval.

Loads lazily on first use. Disable with ``RAG_RERANK_ENABLED=0`` or if ``sentence-transformers`` is missing.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-large"


class SemanticRerankerService:
    """
    Rerank (query, passage) pairs with a BGE cross-encoder.

    Documents are taken from each item's ``full_context`` or ``chunk_text`` (truncated for ``max_length``).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self._model = None

    @classmethod
    def from_env(cls) -> "SemanticRerankerService":
        model = (os.getenv("RAG_RERANKER_MODEL") or DEFAULT_MODEL).strip()
        device = (os.getenv("RAG_RERANKER_DEVICE") or "").strip() or None
        try:
            max_len = int((os.getenv("RAG_RERANKER_MAX_LENGTH") or "512").strip())
            max_len = max(128, min(1024, max_len))
        except ValueError:
            max_len = 512
        try:
            bs = int((os.getenv("RAG_RERANKER_BATCH_SIZE") or "8").strip())
            bs = max(1, min(32, bs))
        except ValueError:
            bs = 8
        return cls(model_name=model, device=device, max_length=max_len, batch_size=bs)

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from api.logging_bootstrap import install_default_log_tuning

            install_default_log_tuning()
        except Exception:
            pass
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is required for semantic reranking. "
                "Install with: pip install sentence-transformers"
            ) from e

        kwargs: Dict[str, Any] = {"max_length": self.max_length}
        if self.device:
            kwargs["device"] = self.device
        logger.info("Loading reranker model %s (max_length=%s)", self.model_name, self.max_length)
        self._model = CrossEncoder(self.model_name, **kwargs)

    def rerank(
        self,
        query: str,
        items: List[Dict[str, Any]],
        *,
        top_k: int = 5,
        text_max_chars: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Score items by relevance to ``query`` and return the top ``top_k`` copies
        with ``rerank_score`` and ``pinecone_score`` (original vector score) set.
        """
        if not items:
            return []
        q = (query or "").strip()
        if not q:
            return list(items)[:top_k]

        char_cap = text_max_chars
        if char_cap is None:
            try:
                char_cap = int((os.getenv("RAG_RERANKER_TEXT_CHARS") or "3200").strip())
            except ValueError:
                char_cap = 3200
        char_cap = max(256, min(12000, char_cap))

        self._load()
        assert self._model is not None

        texts: List[str] = []
        for it in items:
            t = (it.get("full_context") or it.get("chunk_text") or "").strip()
            if len(t) > char_cap:
                t = t[: char_cap - 3] + "..."
            texts.append(t if t else " ")

        pairs = [(q, t) for t in texts]
        raw = self._model.predict(
            pairs,
            show_progress_bar=False,
            batch_size=self.batch_size,
            convert_to_numpy=True,
        )
        scores_list = raw.tolist() if hasattr(raw, "tolist") else list(raw)

        order = sorted(
            range(len(items)),
            key=lambda i: float(scores_list[i]),
            reverse=True,
        )
        top_idx = order[: max(0, top_k)]

        out: List[Dict[str, Any]] = []
        for i in top_idx:
            row = dict(items[i])
            row["pinecone_score"] = row.get("score")
            row["rerank_score"] = float(scores_list[i])
            row["score"] = float(scores_list[i])
            out.append(row)
        return out
