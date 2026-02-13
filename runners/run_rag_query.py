"""Run full RAG query: user question → Pinecone search → PostgreSQL details → LLM answer.

Flow:
  1. Embed the user query
  2. Vector search on Pinecone (top-k similar chunks)
  3. For each match, fetch full record from PostgreSQL (by entity_type + entity_id)
  4. Build context from those details and send to LLM
  5. Print the answer (and optional sources)

Usage:
  cd backend
  python runners/run_rag_query.py "What Citation jets are for sale?"
  python runners/run_rag_query.py "Gulfstream G650" --top-k 5
  python runners/run_rag_query.py "Tell me about N12345" --retrieve-only   # no LLM, just retrieval
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import Config
from database.postgres_client import PostgresClient
from vector_store.pinecone_client import PineconeClient
from rag.embedding_service import EmbeddingService
from rag.query_service import RAGQueryService
from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="RAG query: Pinecone search → Postgres details → LLM answer"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="What aircraft are available?",
        help="User question",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of chunks to retrieve (default: 10)")
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Only run retrieval (Pinecone + Postgres), do not call LLM",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(log_level=args.log_level)
    logger.info("RAG Query: %s", args.query)

    try:
        config = Config.from_env()
        if not all([config.pinecone_api_key, config.openai_api_key, config.postgres_connection_string]):
            logger.error("Set .env: PINECONE_API_KEY, OPENAI_API_KEY, POSTGRES_CONNECTION_STRING")
            return 1

        embedding_service = EmbeddingService(
            api_key=config.openai_api_key,
            model=config.openai_embedding_model,
            dimension=config.openai_embedding_dimension,
        )
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
        db = PostgresClient(config.postgres_connection_string)

        service = RAGQueryService(
            embedding_service=embedding_service,
            pinecone_client=pinecone,
            postgres_client=db,
            openai_api_key=config.openai_api_key,
        )

        if args.retrieve_only:
            results = service.retrieve(args.query, top_k=args.top_k)
            print(f"\nRetrieved {len(results)} items (no LLM):\n")
            for i, r in enumerate(results, 1):
                print(f"  [{i}] {r['entity_type']} id={r['entity_id']} score={r.get('score')}")
                ctx = (r.get("full_context") or r.get("chunk_text") or "")[:300]
                if ctx:
                    print(f"      {ctx}...")
                print()
            return 0

        out = service.answer(args.query, top_k=args.top_k)
        if out.get("error"):
            logger.error("RAG error: %s", out["error"])
            return 1
        print("\nAnswer:\n")
        print(out["answer"])
        if out.get("sources"):
            print("\nSources (entity_type, entity_id, score):")
            for s in out["sources"][:10]:
                print(f"  - {s['entity_type']} {s['entity_id']} ({s.get('score')})")
        return 0
    except Exception as e:
        logger.error("Run failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
