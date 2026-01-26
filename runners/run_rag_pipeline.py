"""Run RAG pipeline to sync PostgreSQL data to Pinecone."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import get_config
from database.postgres_client import PostgresClient
from vector_store.pinecone_client import PineconeClient
from rag.embedding_service import EmbeddingService
from rag.chunking_service import ChunkingService
from rag.rag_pipeline import RAGPipeline
from rag.entity_extractors import EXTRACTORS
from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    """Run RAG pipeline."""
    parser = argparse.ArgumentParser(description="RAG Pipeline - Sync PostgreSQL to Pinecone")
    parser.add_argument(
        '--entities',
        nargs='+',
        choices=list(EXTRACTORS.keys()) + ['all'],
        default=['all'],
        help='Entity types to process (default: all)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of records per entity type (for testing)'
    )
    parser.add_argument(
        '--force-reembed',
        action='store_true',
        help='Force re-embedding even if already embedded'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default='logs/rag_pipeline_log.txt',
        help='Log file path (default: logs/rag_pipeline_log.txt)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Log level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        log_level=args.log_level,
        log_file=args.log_file,
        log_file_overwrite=True
    )
    
    logger.info("=" * 60)
    logger.info("RAG Pipeline - Syncing PostgreSQL to Pinecone")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config = get_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize clients
        logger.info("Initializing clients...")
        
        # PostgreSQL client
        if not config.postgres_connection_string:
            logger.error("PostgreSQL connection string not configured")
            return 1
        
        db_client = PostgresClient(config.postgres_connection_string)
        if not db_client.test_connection():
            logger.error("Failed to connect to PostgreSQL")
            return 1
        logger.info("Connected to PostgreSQL")
        
        # Pinecone client
        if not config.pinecone_api_key:
            logger.error("Pinecone API key not configured")
            return 1
        
        pinecone_client = PineconeClient(
            api_key=config.pinecone_api_key,
            index_name=config.pinecone_index_name,
            dimension=config.pinecone_dimension,
            metric=config.pinecone_metric,
            host=config.pinecone_host
        )
        
        if not pinecone_client.connect():
            logger.error("Failed to connect to Pinecone")
            return 1
        logger.info("Connected to Pinecone")
        
        # Embedding service
        if not config.openai_api_key:
            logger.error("OpenAI API key not configured")
            return 1
        
        embedding_service = EmbeddingService(
            api_key=config.openai_api_key,
            model=config.openai_embedding_model,
            dimension=config.openai_embedding_dimension
        )
        logger.info(f"Initialized embedding service: {config.openai_embedding_model}")
        
        # Chunking service
        chunking_service = ChunkingService(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        logger.info(f"Initialized chunking service (size: {config.chunk_size}, overlap: {config.chunk_overlap})")
        
        # RAG Pipeline
        rag_pipeline = RAGPipeline(
            db_client=db_client,
            pinecone_client=pinecone_client,
            embedding_service=embedding_service,
            chunking_service=chunking_service,
            embedding_model=config.openai_embedding_model,
            embedding_dimension=config.openai_embedding_dimension,
            batch_size=config.batch_size
        )
        
        # Determine entity types
        entity_types = None
        if 'all' not in args.entities:
            entity_types = args.entities
        
        if args.limit:
            logger.info(f"TEST MODE - Processing limited data (limit: {args.limit} per entity type)")
        
        if args.force_reembed:
            logger.info("FORCE RE-EMBED MODE - Will re-embed all records")
        
        # Run pipeline
        logger.info("Starting RAG pipeline sync...")
        summary = rag_pipeline.sync_all(
            entity_types=entity_types,
            limit=args.limit,
            force_reembed=args.force_reembed
        )
        
        # Print summary
        logger.info("=" * 60)
        logger.info("RAG Pipeline Completed!")
        logger.info("=" * 60)
        logger.info(f"Total Processed: {summary['total_processed']}")
        logger.info(f"Total Embedded: {summary['total_embedded']}")
        logger.info(f"Total Skipped: {summary['total_skipped']}")
        logger.info(f"Total Errors: {summary['total_errors']}")
        logger.info(f"Total Chunks Created: {summary['total_chunks']}")
        logger.info(f"Total Vectors Upserted: {summary['total_vectors']}")
        logger.info("")
        logger.info("Per Entity Type:")
        for entity_type, stats in summary['entity_stats'].items():
            logger.info(f"  {entity_type}:")
            logger.info(f"    - Processed: {stats.get('processed', 0)}")
            logger.info(f"    - Embedded: {stats.get('embedded', 0)}")
            logger.info(f"    - Skipped: {stats.get('skipped', 0)}")
            logger.info(f"    - Errors: {stats.get('errors', 0)}")
            logger.info(f"    - Chunks: {stats.get('chunks_created', 0)}")
            logger.info(f"    - Vectors: {stats.get('vectors_upserted', 0)}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"RAG pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
