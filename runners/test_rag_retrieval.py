"""Test RAG retrieval: embed a question, query Pinecone, print top matches.

Run after:
  1. Data is in PostgreSQL (ETL pipeline)
  2. Data is embedded in Pinecone (run_rag_pipeline.py)

Usage:
  cd backend
  python runners/test_rag_retrieval.py "What Citation jets are for sale?"
  python runners/test_rag_retrieval.py "Gulfstream G650" --top-k 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import Config
from database.postgres_client import PostgresClient
from vector_store.pinecone_client import PineconeClient
from rag.embedding_service import EmbeddingService
from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test RAG retrieval: query Pinecone with a question")
    parser.add_argument(
        "query",
        nargs="?",
        default="What aircraft are available?",
        help="Question or search phrase to embed and search (default: What aircraft are available?)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    setup_logging(log_level=args.log_level)
    logger.info("=" * 60)
    logger.info("RAG Retrieval Test - Query Pinecone with a question")
    logger.info("=" * 60)

    try:
        config = Config.from_env()
        if not config.pinecone_api_key or not config.openai_api_key:
            logger.error("Configure .env: PINECONE_API_KEY, OPENAI_API_KEY (and PINECONE_INDEX_NAME)")
            return 1

        # Embedding service (same model as pipeline)
        embedding_service = EmbeddingService(
            api_key=config.openai_api_key,
            model=config.openai_embedding_model,
            dimension=config.openai_embedding_dimension,
        )
        logger.info(f"Embedding query with {config.openai_embedding_model}...")
        query_vector = embedding_service.embed_text(args.query)
        if not query_vector:
            logger.error("Failed to embed query (check OPENAI_API_KEY and model)")
            return 1
        logger.info(f"Query embedded (dim={len(query_vector)})")

        # Pinecone
        pinecone = PineconeClient(
            api_key=config.pinecone_api_key,
            index_name=config.pinecone_index_name,
            dimension=config.pinecone_dimension,
            metric=config.pinecone_metric,
            host=config.pinecone_host,
        )
        if not pinecone.connect():
            logger.error("Failed to connect to Pinecone")
            return 1
        logger.info("Connected to Pinecone")

        # Search
        matches = pinecone.query(vector=query_vector, top_k=args.top_k)
        logger.info(f"Found {len(matches)} matches for: \"{args.query}\"")
        print()
        for i, m in enumerate(matches, 1):
            score = getattr(m, "score", None) if hasattr(m, "score") else (m.get("score") if isinstance(m, dict) else None)
            meta = getattr(m, "metadata", None) if hasattr(m, "metadata") else (m.get("metadata") if isinstance(m, dict) else {})
            meta = meta or {}
            text = (meta.get("text") or "")[:200]
            entity = meta.get("entity_type", "")
            eid = meta.get("entity_id", "")
            score_str = f"{score:.4f}" if score is not None else "N/A"
            print(f"  [{i}] score={score_str} | {entity} {eid}")
            if text:
                print(f"      {text}...")
            print()
        logger.info("=" * 60)
        return 0
    except Exception as e:
        logger.error(f"Retrieval test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
