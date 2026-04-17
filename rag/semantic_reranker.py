"""
Cross-encoder semantic reranking (BAAI BGE) for RAG retrieval.

Loads lazily on first use. Disable with ``RAG_RERANK_ENABLED=0`` or if ``sentence-transformers`` is missing.

**Memory (small hosts / Render starter):** set ``RAG_RERANK_LIGHT=1`` (or ``auto`` — light when
``RENDER`` is truthy) to use ``bge-reranker-base`` and smaller batch / passage caps. Rerank
still runs; only the model footprint and peak activations shrink.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-large"
DEFAULT_MODEL_LIGHT = "BAAI/bge-reranker-base"


def _env_truthy(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on")


def _rerank_light_profile_enabled() -> bool:
    """
    Light profile: smaller cross-encoder + conservative tensor sizes for low-RAM hosts.

    - ``RAG_RERANK_LIGHT=1|true|yes|on|light`` → on
    - ``RAG_RERANK_LIGHT=0|false|no|off|full|standard`` → off (full/large model unless overridden)
    - ``RAG_RERANK_LIGHT=auto`` → on when ``RENDER`` is set (typical Render.com web service)
    - unset → off (backward compatible)
    """
    raw = (os.getenv("RAG_RERANK_LIGHT") or "").strip().lower()
    if raw in ("1", "true", "yes", "on", "light"):
        return True
    if raw in ("0", "false", "no", "off", "full", "standard"):
        return False
    if raw == "auto":
        return _env_truthy(os.getenv("RENDER") or "")
    return False


def effective_reranker_model_name_from_env() -> str:
    """Model id used for reranking given current env (explicit ``RAG_RERANKER_MODEL`` wins over light default)."""
    explicit = (os.getenv("RAG_RERANKER_MODEL") or "").strip()
    if explicit:
        return explicit
    if _rerank_light_profile_enabled():
        return DEFAULT_MODEL_LIGHT
    return DEFAULT_MODEL


def _apply_cpu_thread_cap_for_memory() -> None:
    """Reduce PyTorch thread pools on CPU — lowers peak RSS on small containers (e.g. Render)."""
    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass
    except Exception:
        pass


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
        light = _rerank_light_profile_enabled()
        explicit_model = (os.getenv("RAG_RERANKER_MODEL") or "").strip()
        if explicit_model:
            model = explicit_model
        elif light:
            model = DEFAULT_MODEL_LIGHT
        else:
            model = DEFAULT_MODEL

        device = (os.getenv("RAG_RERANKER_DEVICE") or "").strip() or None

        max_len_raw = os.getenv("RAG_RERANKER_MAX_LENGTH")
        if max_len_raw is not None and str(max_len_raw).strip() != "":
            try:
                max_len = int(str(max_len_raw).strip())
                max_len = max(128, min(1024, max_len))
            except ValueError:
                max_len = 384 if light else 512
        else:
            max_len = 384 if light else 512

        bs_raw = os.getenv("RAG_RERANKER_BATCH_SIZE")
        if bs_raw is not None and str(bs_raw).strip() != "":
            try:
                bs = int(str(bs_raw).strip())
                bs = max(1, min(32, bs))
            except ValueError:
                bs = 4 if light else 8
        else:
            bs = 4 if light else 8

        if light:
            logger.info(
                "Reranker light profile: model=%s max_length=%s batch_size=%s",
                model,
                max_len,
                bs,
            )
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

        dev = (self.device or "").strip().lower()
        if not dev or dev == "cpu":
            _apply_cpu_thread_cap_for_memory()

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
            light = _rerank_light_profile_enabled()
            default_chars = "2000" if light else "3200"
            raw = (os.getenv("RAG_RERANKER_TEXT_CHARS") or "").strip()
            try:
                char_cap = int((raw or default_chars).strip())
            except ValueError:
                char_cap = int(default_chars)
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
