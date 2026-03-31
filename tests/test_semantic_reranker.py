"""SemanticRerankerService without loading BGE weights (mocked CrossEncoder output)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from rag.semantic_reranker import SemanticRerankerService


def test_rerank_orders_by_score_and_top_k():
    svc = SemanticRerankerService(model_name="dummy", max_length=128, batch_size=4)
    mock_model = MagicMock()
    mock_model.predict = MagicMock(
        return_value=np.array([0.2, 0.9, 0.5], dtype=np.float32)
    )
    svc._model = mock_model

    items = [
        {"full_context": "low", "score": 0.99},
        {"full_context": "high", "score": 0.5},
        {"full_context": "mid", "score": 0.8},
    ]
    out = svc.rerank("query jet", items, top_k=2)

    assert len(out) == 2
    assert out[0]["full_context"] == "high"
    assert out[0]["rerank_score"] == pytest.approx(0.9)
    assert out[0]["pinecone_score"] == 0.5
    assert out[1]["full_context"] == "mid"
    assert out[1]["score"] == pytest.approx(0.5)


def test_rerank_empty_query_returns_slice():
    svc = SemanticRerankerService()
    svc._model = MagicMock()
    items = [{"full_context": "x", "score": 1.0}]
    out = svc.rerank("", items, top_k=1)
    assert len(out) == 1
