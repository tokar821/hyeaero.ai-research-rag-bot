"""Remove ``embeddings_metadata`` rows for entity types before a full Pinecone re-sync."""

from __future__ import annotations

import logging
from typing import Collection, Optional

from database.postgres_client import PostgresClient

logger = logging.getLogger(__name__)


def delete_embeddings_metadata_for_entity_types(
    db: PostgresClient,
    entity_types: Collection[str],
    *,
    embedding_model: Optional[str] = None,
) -> int:
    """
    Delete tracking rows so :class:`~rag.rag_pipeline.RAGPipeline` will re-embed all rows
    for those types (when combined with Pinecone delete + ``force_reembed``).

    If ``embedding_model`` is set, only rows for that model are removed.
    """
    types = [str(t) for t in entity_types if t]
    if not types:
        return 0
    if embedding_model:
        sql = """
            DELETE FROM embeddings_metadata
            WHERE entity_type = ANY(%s::text[])
              AND embedding_model = %s
        """
        n = db.execute_update(sql, (types, embedding_model))
    else:
        sql = """
            DELETE FROM embeddings_metadata
            WHERE entity_type = ANY(%s::text[])
        """
        n = db.execute_update(sql, (types,))
    logger.info("embeddings_metadata: deleted %s rows for entity types %s", n, types)
    return int(n or 0)
