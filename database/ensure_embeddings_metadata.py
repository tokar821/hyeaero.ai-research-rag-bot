"""
Apply ``ensure_embeddings_metadata.sql`` so PhlyData / RAG pipelines can use ``embeddings_metadata``.

Safe to call before ``run_embed_phlydata.py`` or ``rag_pipeline`` sync.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from database.postgres_client import PostgresClient

logger = logging.getLogger(__name__)

_MIGRATION = Path(__file__).resolve().parent / "migrations" / "ensure_embeddings_metadata.sql"


def _sql_statements(sql_text: str) -> List[str]:
    """Split migration file on semicolons (no dollar-quoted bodies in this file)."""
    lines_out: List[str] = []
    for line in sql_text.splitlines():
        if line.strip().startswith("--"):
            continue
        lines_out.append(line)
    blob = "\n".join(lines_out)
    parts = [p.strip() for p in blob.split(";")]
    return [p for p in parts if p]


def apply_embeddings_metadata_schema(db: PostgresClient) -> bool:
    """
    Create ``embeddings_metadata`` and indexes if missing; add ``entity_type`` / ``entity_id`` when absent.
    """
    if not _MIGRATION.is_file():
        logger.error("Migration file missing: %s", _MIGRATION)
        return False
    sql_text = _MIGRATION.read_text(encoding="utf-8")
    statements = _sql_statements(sql_text)
    for i, stmt in enumerate(statements, 1):
        try:
            db.execute_update(stmt)
        except Exception as e:
            msg = str(e).lower()
            if "already exists" in msg or "duplicate" in msg:
                logger.debug("Statement %s skipped (already applied): %s", i, e)
                continue
            logger.error("Migration statement %s failed: %s\n%s", i, e, stmt[:200])
            return False
    logger.info("embeddings_metadata schema ensured (%s statements)", len(statements))
    return True
