"""
Delete Pinecone vectors + embeddings_metadata for **structured aviation** datasets only, then re-embed
using current entity-based chunking and :mod:`rag.pinecone_metadata`.

Does **not** touch ``document`` embeddings (manuals, PDFs, articles). Includes **AircraftPost** as
``aircraftpost_fleet_aircraft``. Controller / Exchange-style feeds are usually embedded as ``aircraft_listing``.

Steps:
  1. Delete default-namespace vectors whose ``entity_type`` is one of the structured set.
  2. Delete all vectors in Pinecone namespace ``phlydata_aircraft``.
  3. Remove matching rows from ``embeddings_metadata`` (optional filter by embedding model).
  4. Run :class:`~rag.rag_pipeline.RAGPipeline` for the five default-namespace types with ``force_reembed``.
  5. Run :func:`~rag.phlydata_aircraft_embed.sync_phlydata_aircraft_embeddings` with ``force_reembed``.

Usage::

    python runners/run_reembed_structured_aviation.py
    python runners/run_reembed_structured_aviation.py --limit 50
    python runners/run_reembed_structured_aviation.py --dry-run
    python runners/run_reembed_structured_aviation.py --verify

Env:
  Typical ``POSTGRES_*``, ``PINECONE_*``, ``OPENAI_*``, ``RAG_BATCH_SIZE`` (Pinecone upsert batch).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import get_config
from database.ensure_embeddings_metadata import apply_embeddings_metadata_schema
from database.postgres_client import PostgresClient
from rag.chunking_service import ChunkingService
from rag.embeddings_metadata_cleanup import delete_embeddings_metadata_for_entity_types
from rag.embedding_service import EmbeddingService
from rag.phlydata_aircraft_embed import sync_phlydata_aircraft_embeddings
from rag.rag_pipeline import RAGPipeline
from rag.structured_reembed_constants import (
    PHLYDATA_AIRCRAFT_ENTITY_TYPE,
    PHLYDATA_AIRCRAFT_PINECONE_NAMESPACE,
    STRUCTURED_AVIATION_ALL_ENTITY_TYPES,
    STRUCTURED_AVIATION_DEFAULT_NS_ENTITY_TYPES,
)
from utils.logger import get_logger, setup_logging
from vector_store.pinecone_client import PineconeClient

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-embed structured aviation entities only (chunking + metadata schema refresh)"
    )
    parser.add_argument("--limit", type=int, default=None, help="Max rows per entity type (testing)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions only; no Pinecone/Postgres writes",
    )
    parser.add_argument(
        "--skip-pinecone-delete",
        action="store_true",
        help="Skip Pinecone deletes (only metadata clear + embed)",
    )
    parser.add_argument(
        "--skip-metadata-delete",
        action="store_true",
        help="Skip embeddings_metadata DELETE (still use --force-reembed on pipeline)",
    )
    parser.add_argument(
        "--pinecone-batch",
        type=int,
        default=200,
        help="Vectors per Pinecone upsert batch (100–500 recommended)",
    )
    parser.add_argument(
        "--record-batch",
        type=int,
        default=200,
        help="Postgres entities to accumulate before a Pinecone flush in RAGPipeline",
    )
    parser.add_argument(
        "--embedding-api-batch",
        type=int,
        default=200,
        help="Max texts per OpenAI embeddings.create call (100–500)",
    )
    parser.add_argument(
        "--phly-page-size",
        type=int,
        default=200,
        help="Phly DB page size",
    )
    parser.add_argument(
        "--phly-upsert-batch",
        type=int,
        default=100,
        help="Phly aircraft rows before Pinecone flush",
    )
    parser.add_argument(
        "--skip-phly",
        action="store_true",
        help="Do not re-embed phlydata_aircraft namespace",
    )
    parser.add_argument(
        "--skip-default-ns",
        action="store_true",
        help="Do not re-embed default-namespace structured types",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After sync, print Pinecone stats (run without --dry-run for real counts)",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-file", type=str, default="logs/reembed_structured_log.txt")
    args = parser.parse_args()

    setup_logging(log_level=args.log_level, log_file=args.log_file, log_file_overwrite=True)

    pc_batch = max(50, min(500, args.pinecone_batch))
    rec_batch = max(50, min(500, args.record_batch))
    emb_batch = max(50, min(500, args.embedding_api_batch))

    try:
        config = get_config()
        if not config.postgres_connection_string:
            logger.error("PostgreSQL not configured")
            return 1
        if not config.pinecone_api_key:
            logger.error("Pinecone not configured")
            return 1
        if not config.openai_api_key:
            logger.error("OpenAI not configured")
            return 1

        db = PostgresClient(config.postgres_connection_string)
        if not db.test_connection():
            logger.error("PostgreSQL connection failed")
            return 1

        if args.dry_run:
            logger.info("DRY RUN — would delete Pinecone filter entity_type $in %s", list(STRUCTURED_AVIATION_DEFAULT_NS_ENTITY_TYPES))
            logger.info("DRY RUN — would delete_all namespace %r", PHLYDATA_AIRCRAFT_PINECONE_NAMESPACE)
            logger.info("DRY RUN — would DELETE embeddings_metadata for %s", sorted(STRUCTURED_AVIATION_ALL_ENTITY_TYPES))
            logger.info("DRY RUN — would sync RAG default-ns types: %s", list(STRUCTURED_AVIATION_DEFAULT_NS_ENTITY_TYPES))
            if not args.skip_phly:
                logger.info("DRY RUN — would sync Phly namespace %r", PHLYDATA_AIRCRAFT_PINECONE_NAMESPACE)
            return 0

        if not apply_embeddings_metadata_schema(db):
            logger.error("embeddings_metadata schema missing; run ensure_embeddings_metadata")
            return 1

        pinecone = PineconeClient(
            api_key=config.pinecone_api_key,
            index_name=config.pinecone_index_name,
            dimension=config.pinecone_dimension,
            metric=config.pinecone_metric,
            host=config.pinecone_host,
        )
        if not pinecone.connect():
            logger.error("Pinecone connect failed")
            return 1

        if not args.skip_pinecone_delete:
            flt = {"entity_type": {"$in": list(STRUCTURED_AVIATION_DEFAULT_NS_ENTITY_TYPES)}}
            logger.info("Pinecone: deleting default-namespace vectors with structured entity_type filter…")
            if not pinecone.delete_by_metadata_filter(flt, namespace=None):
                logger.error("Pinecone structured delete (default ns) failed")
                return 1
            if not args.skip_phly:
                logger.info("Pinecone: delete_all in namespace %r…", PHLYDATA_AIRCRAFT_PINECONE_NAMESPACE)
                if not pinecone.delete_all_in_namespace(PHLYDATA_AIRCRAFT_PINECONE_NAMESPACE):
                    logger.error("Pinecone phly namespace delete failed")
                    return 1
        else:
            logger.warning("Skipping Pinecone deletes (--skip-pinecone-delete)")

        if not args.skip_metadata_delete:
            delete_embeddings_metadata_for_entity_types(
                db,
                STRUCTURED_AVIATION_ALL_ENTITY_TYPES,
                embedding_model=config.openai_embedding_model,
            )
        else:
            logger.warning("Skipping embeddings_metadata delete")

        embed = EmbeddingService(
            api_key=config.openai_api_key,
            model=config.openai_embedding_model,
            dimension=config.openai_embedding_dimension,
        )
        chunker = ChunkingService(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)

        if not args.skip_default_ns:
            rag = RAGPipeline(
                db_client=db,
                pinecone_client=pinecone,
                embedding_service=embed,
                chunking_service=chunker,
                embedding_model=config.openai_embedding_model,
                embedding_dimension=config.openai_embedding_dimension,
                batch_size=pc_batch,
                embedding_batch_size=emb_batch,
            )
            rag.upsert_record_batch = rec_batch
            summary = rag.sync_all(
                entity_types=list(STRUCTURED_AVIATION_DEFAULT_NS_ENTITY_TYPES),
                limit=args.limit,
                force_reembed=True,
            )
            logger.info("Default-namespace structured sync summary: %s", summary)
        else:
            logger.warning("Skipping default-namespace RAG sync")

        if not args.skip_phly:
            phly_stats = sync_phlydata_aircraft_embeddings(
                db,
                pinecone,
                embed,
                chunker,
                embedding_model=config.openai_embedding_model,
                embedding_dimension=config.openai_embedding_dimension,
                limit=args.limit,
                force_reembed=True,
                page_size=args.phly_page_size,
                upsert_batch=args.phly_upsert_batch,
                pinecone_batch=pc_batch,
                openai_embed_batch=emb_batch,
            )
            logger.info("Phly sync stats: %s", phly_stats)
        else:
            logger.warning("Skipping phlydata_aircraft embed")

        if args.verify:
            stats = pinecone.get_stats()
            logger.info("Pinecone index stats: %s", stats)
            for qrow in db.execute_query(
                "SELECT entity_type, COUNT(*) AS c FROM embeddings_metadata GROUP BY entity_type ORDER BY entity_type"
            ):
                logger.info("embeddings_metadata: %s -> %s rows", qrow.get("entity_type"), qrow.get("c"))

        logger.info("Structured aviation re-embed finished.")
        return 0
    except Exception as e:
        logger.error("reembed structured failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
