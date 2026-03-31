"""
Embed ``public.phlydata_aircraft`` rows from PostgreSQL into Pinecone (dedicated namespace).

One Phly aircraft row → one structured text chunk (via :class:`~rag.chunking_service.ChunkingService.chunk_for_entity`),
built from **all** table columns (including dynamic ``csv_*``). Use **SQL / format_phlydata**
for verbatim prices; vectors are for **semantic recall** only.

Pinecone namespace: ``phlydata_aircraft`` (same index as main RAG from config; isolate with namespace in queries).

Before first run, ``embeddings_metadata`` must exist — apply ``database/migrations/ensure_embeddings_metadata.sql``
via ``python runners/ensure_embeddings_metadata.py`` (or it runs automatically from ``run_embed_phlydata.py``).
"""

from __future__ import annotations

import logging
import math
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from database.postgres_client import PostgresClient
from rag.chunking_service import ChunkingService
from rag.embedding_service import EmbeddingService
from rag.aircraft_normalization import normalize_aircraft_identity
from rag.pinecone_metadata import build_vector_metadata, sanitize_pinecone_metadata_dict
from rag.phlydata_aircraft_schema import _DEFAULT_EXCLUDE, fetch_phlydata_aircraft_data_columns
from vector_store.pinecone_client import PineconeClient

logger = logging.getLogger(__name__)

ENTITY_TYPE = "phlydata_aircraft"
PINECONE_NAMESPACE = "phlydata_aircraft"
MAX_FIELD_CHARS = 12_000
MAX_EMBED_TEXT_CHARS = 850_000


def _scalar_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Decimal):
        s = format(value, "f").rstrip("0").rstrip(".")
        return s if s else ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return str(value)
    if isinstance(value, (datetime, date)):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    s = str(value).strip()
    return s


def phly_row_to_embedding_text(row: Dict[str, Any], column_order: List[str]) -> str:
    """
    Labeled text for embedding: identity columns first, then remaining typed fields, then ``csv_*`` alpha keys.
    Hyphens and spelling match Postgres (trim only on output lines).
    """
    lines: List[str] = [
        "PhlyData internal aircraft export (phlydata_aircraft).",
        "Use PostgreSQL authoritative row for exact ask_price and aircraft_status.",
        "",
    ]
    aid = row.get("aircraft_id")
    lines.append(f"aircraft_id: {aid}")
    cm, cmo = normalize_aircraft_identity(row.get("manufacturer"), row.get("model"))
    if cm and cmo:
        lines.append(f"rag_canonical_aircraft_type: {cm} {cmo}")
        lines.append("")

    # Preferred identity order first if present
    priority = (
        "serial_number",
        "registration_number",
        "manufacturer",
        "model",
        "manufacturer_year",
        "delivery_year",
        "category",
        "aircraft_status",
        "ask_price",
        "take_price",
        "sold_price",
    )
    seen: Set[str] = set()
    ordered: List[str] = []
    for c in priority:
        if c in column_order and c not in seen:
            ordered.append(c)
            seen.add(c)
    csv_cols = sorted(c for c in column_order if c.startswith("csv_") and c not in seen)
    rest = [c for c in column_order if c not in seen and not c.startswith("csv_")]
    ordered.extend(rest)
    ordered.extend(csv_cols)

    for col in ordered:
        raw = row.get(col)
        t = _scalar_to_text(raw)
        if not t:
            continue
        if len(t) > MAX_FIELD_CHARS:
            t = t[: MAX_FIELD_CHARS - 3] + "..."
        lines.append(f"{col}: {t}")

    return "\n".join(lines).strip()


def get_embedded_phly_aircraft_ids(
    db: PostgresClient,
    embedding_model: str,
) -> Set[str]:
    q = """
        SELECT DISTINCT CAST(entity_id AS TEXT) AS eid
        FROM embeddings_metadata
        WHERE entity_type = %s
          AND embedding_model = %s
          AND entity_id IS NOT NULL
    """
    rows = db.execute_query(q, (ENTITY_TYPE, embedding_model))
    return {str(r["eid"]) for r in rows if r.get("eid")}


def _insert_embedding_metadata(db: PostgresClient, metadata_list: List[Dict[str, Any]]) -> None:
    for metadata in metadata_list:
        try:
            check_query = """
                SELECT id FROM embeddings_metadata
                WHERE entity_type = %s
                  AND entity_id = %s
                  AND embedding_model = %s
                LIMIT 1
            """
            exists = db.execute_query(
                check_query,
                (metadata["entity_type"], metadata["entity_id"], metadata["embedding_model"]),
            )
            if exists:
                update_query = """
                    UPDATE embeddings_metadata SET
                        chunk_count = %s,
                        vector_store_id = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE entity_type = %s
                      AND entity_id = %s
                      AND embedding_model = %s
                """
                db.execute_update(
                    update_query,
                    (
                        metadata["chunk_count"],
                        metadata["vector_store_id"],
                        metadata["entity_type"],
                        metadata["entity_id"],
                        metadata["embedding_model"],
                    ),
                )
            else:
                insert_query = """
                    INSERT INTO embeddings_metadata (
                        entity_type, entity_id, document_id,
                        embedding_model, embedding_dimension,
                        chunk_count, vector_store, vector_store_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                db.execute_update(
                    insert_query,
                    (
                        metadata["entity_type"],
                        metadata["entity_id"],
                        metadata.get("document_id"),
                        metadata["embedding_model"],
                        metadata["embedding_dimension"],
                        metadata["chunk_count"],
                        metadata["vector_store"],
                        metadata["vector_store_id"],
                    ),
                )
        except Exception as e:
            logger.error("Failed to insert embedding metadata: %s", e, exc_info=True)


def sync_phlydata_aircraft_embeddings(
    db: PostgresClient,
    pinecone: PineconeClient,
    embedding_service: EmbeddingService,
    chunking_service: ChunkingService,
    *,
    embedding_model: str,
    embedding_dimension: int,
    limit: Optional[int] = None,
    force_reembed: bool = False,
    page_size: int = 200,
    upsert_batch: int = 100,
    pinecone_batch: int = 100,
    openai_embed_batch: int = 200,
) -> Dict[str, Any]:
    """
    Read all ``phlydata_aircraft`` rows from Postgres; embed; upsert to Pinecone namespace ``phlydata_aircraft``.
    """
    stats: Dict[str, Any] = {
        "entity_type": ENTITY_TYPE,
        "processed": 0,
        "embedded": 0,
        "skipped": 0,
        "errors": 0,
        "chunks_created": 0,
        "vectors_upserted": 0,
    }

    columns = [
        c
        for c in fetch_phlydata_aircraft_data_columns(db)
        if c not in _DEFAULT_EXCLUDE
    ]
    col_sql = ", ".join(f'"{c}"' if re.search(r"[^a-z0-9_]", c, re.I) else c for c in columns)
    # aircraft_id is PK; data columns list may duplicate — ensure aircraft_id selected once
    select_cols = f'aircraft_id, {col_sql}' if col_sql else "aircraft_id"

    embedded_ids: Set[str] = set()
    if not force_reembed:
        embedded_ids = get_embedded_phly_aircraft_ids(db, embedding_model)
        logger.info("Phly embed: %s aircraft already recorded in embeddings_metadata", len(embedded_ids))

    offset = 0
    total_limit = limit
    vectors_buf: List[Dict[str, Any]] = []
    meta_buf: List[Dict[str, Any]] = []

    while True:
        if total_limit is not None and stats["processed"] >= total_limit:
            break
        batch_lim = page_size
        if total_limit is not None:
            batch_lim = min(page_size, total_limit - stats["processed"])
        if batch_lim <= 0:
            break

        sql = f"""
            SELECT {select_cols}
            FROM public.phlydata_aircraft
            ORDER BY aircraft_id
            LIMIT %s OFFSET %s
        """
        rows = db.execute_query(sql, (batch_lim, offset))
        if not rows:
            break
        offset += len(rows)

        for row in rows:
            stats["processed"] += 1
            aid = row.get("aircraft_id")
            eid = str(aid) if aid is not None else ""

            if eid and eid in embedded_ids and not force_reembed:
                stats["skipped"] += 1
                continue

            text = phly_row_to_embedding_text(row, columns)
            if not text or len(text) < 20:
                stats["skipped"] += 1
                continue
            if len(text) > MAX_EMBED_TEXT_CHARS:
                logger.warning("Phly row %s: text truncated from %s chars", eid, len(text))
                text = text[: MAX_EMBED_TEXT_CHARS - 200] + "\n…(truncated)"

            try:
                chunks = chunking_service.chunk_for_entity(
                    ENTITY_TYPE,
                    text,
                    {},
                    chunk_id_prefix=f"{ENTITY_TYPE}_{eid}",
                )
            except Exception as e:
                logger.error("Chunk failed aircraft_id=%s: %s", eid, e)
                stats["errors"] += 1
                continue
            if not chunks:
                stats["skipped"] += 1
                continue

            texts = [c["text"] for c in chunks]
            eb = max(32, min(500, int(openai_embed_batch)))
            embeddings = embedding_service.embed_batch(texts, batch_size=eb)

            row_vectors: List[Dict[str, Any]] = []
            n_chunks = len(chunks)
            for chunk, emb in zip(chunks, embeddings):
                if emb is None:
                    continue
                cid = chunk["chunk_index"]
                vector_id = f"{ENTITY_TYPE}_{eid}_chunk_{cid}"
                std_meta = build_vector_metadata(
                    ENTITY_TYPE,
                    row,
                    entity_id_override=eid if eid else None,
                    chunk_index=int(cid),
                    total_chunks=n_chunks,
                )
                cstrat = (chunk.get("metadata") or {}).get("chunking_strategy")
                if cstrat:
                    std_meta["chunking_strategy"] = str(cstrat)[:48]
                meta = sanitize_pinecone_metadata_dict(std_meta)
                row_vectors.append({"id": vector_id, "values": emb, "metadata": meta})

            if not row_vectors:
                stats["skipped"] += 1
                continue

            stats["chunks_created"] += len(row_vectors)
            vectors_buf.extend(row_vectors)

            meta_buf.append(
                {
                    "entity_type": ENTITY_TYPE,
                    "entity_id": eid,
                    "document_id": None,
                    "embedding_model": embedding_model,
                    "embedding_dimension": embedding_dimension,
                    "chunk_count": len(chunks),
                    "vector_store": "pinecone",
                    "vector_store_id": f"{ENTITY_TYPE}_{eid}",
                }
            )
            stats["embedded"] += 1

            if len(meta_buf) >= upsert_batch:
                try:
                    n = pinecone.upsert_vectors(
                        vectors_buf, namespace=PINECONE_NAMESPACE, batch_size=pinecone_batch
                    )
                    stats["vectors_upserted"] += n
                    _insert_embedding_metadata(db, meta_buf)
                except Exception as e:
                    logger.error("Upsert batch failed: %s", e, exc_info=True)
                    stats["errors"] += len(meta_buf)
                vectors_buf = []
                meta_buf = []

        logger.info(
            "Phly embed progress: processed=%s embedded=%s skipped=%s vectors=%s",
            stats["processed"],
            stats["embedded"],
            stats["skipped"],
            stats["vectors_upserted"],
        )

    if vectors_buf:
        try:
            n = pinecone.upsert_vectors(
                vectors_buf, namespace=PINECONE_NAMESPACE, batch_size=pinecone_batch
            )
            stats["vectors_upserted"] += n
            _insert_embedding_metadata(db, meta_buf)
        except Exception as e:
            logger.error("Final upsert failed: %s", e, exc_info=True)
            stats["errors"] += len(meta_buf)

    return stats
