"""Embed ``public.phlydata_aircraft`` from PostgreSQL into Pinecone namespace ``phlydata_aircraft``."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import get_config
from database.ensure_embeddings_metadata import apply_embeddings_metadata_schema
from database.postgres_client import PostgresClient
from rag.chunking_service import ChunkingService
from rag.embedding_service import EmbeddingService
from rag.phlydata_aircraft_embed import PINECONE_NAMESPACE, sync_phlydata_aircraft_embeddings
from utils.logger import get_logger, setup_logging
from vector_store.pinecone_client import PineconeClient

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Embed phlydata_aircraft rows (Postgres) → Pinecone dedicated namespace"
    )
    parser.add_argument("--limit", type=int, default=None, help="Max rows to embed (test)")
    parser.add_argument(
        "--force-reembed",
        action="store_true",
        help="Re-embed even if embeddings_metadata already has phlydata_aircraft",
    )
    parser.add_argument("--page-size", type=int, default=200, help="Rows per DB page")
    parser.add_argument("--upsert-batch", type=int, default=50, help="Aircraft rows before Pinecone upsert")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", type=str, default="logs/embed_phlydata_log.txt")
    parser.add_argument(
        "--skip-ensure-table",
        action="store_true",
        help="Do not run ensure_embeddings_metadata migration (table must already exist)",
    )
    args = parser.parse_args()

    setup_logging(log_level=args.log_level, log_file=args.log_file, log_file_overwrite=True)

    logger.info("PhlyData aircraft embed — Pinecone namespace: %s (same index as main RAG)", PINECONE_NAMESPACE)

    try:
        config = get_config()
        if not config.postgres_connection_string:
            logger.error("POSTGRES connection not configured")
            return 1
        db = PostgresClient(config.postgres_connection_string)
        if not db.test_connection():
            logger.error("PostgreSQL connection failed")
            return 1

        if not args.skip_ensure_table:
            if not apply_embeddings_metadata_schema(db):
                logger.error("Failed to ensure embeddings_metadata table; fix DB or use --skip-ensure-table")
                return 1

        if not config.pinecone_api_key:
            logger.error("Pinecone API key not configured")
            return 1
        pinecone = PineconeClient(
            api_key=config.pinecone_api_key,
            index_name=config.pinecone_index_name,
            dimension=config.pinecone_dimension,
            metric=config.pinecone_metric,
            host=config.pinecone_host,
        )
        if not pinecone.connect():
            logger.error("Pinecone connect failed — index must exist: %s", config.pinecone_index_name)
            return 1
        logger.info(
            "Pinecone: index=%s metric=%s dimension=%s — upserting to namespace=%r only",
            config.pinecone_index_name,
            config.pinecone_metric,
            config.pinecone_dimension,
            PINECONE_NAMESPACE,
        )

        if not config.openai_api_key:
            logger.error("OpenAI API key not configured")
            return 1
        embed = EmbeddingService(
            api_key=config.openai_api_key,
            model=config.openai_embedding_model,
            dimension=config.openai_embedding_dimension,
        )
        chunker = ChunkingService(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        stats = sync_phlydata_aircraft_embeddings(
            db,
            pinecone,
            embed,
            chunker,
            embedding_model=config.openai_embedding_model,
            embedding_dimension=config.openai_embedding_dimension,
            limit=args.limit,
            force_reembed=args.force_reembed,
            page_size=args.page_size,
            upsert_batch=args.upsert_batch,
        )

        logger.info("Done: %s", stats)
        return 0
    except Exception as e:
        logger.error("embed phlydata failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
