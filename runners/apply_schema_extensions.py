"""Apply schema extensions for RAG pipeline.

Extends embeddings_metadata table to support multiple entity types.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import get_config
from database.postgres_client import PostgresClient
from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    """Apply schema extensions."""
    setup_logging(
        log_level="INFO",
        log_file="logs/schema_extensions_log.txt",
        log_file_overwrite=True
    )
    
    logger.info("Applying schema extensions for RAG pipeline...")
    
    try:
        config = get_config()
        
        if not config.postgres_connection_string:
            logger.error("PostgreSQL connection string not configured")
            return 1
        
        db_client = PostgresClient(config.postgres_connection_string)
        
        if not db_client.test_connection():
            logger.error("Failed to connect to PostgreSQL")
            return 1
        
        # Read schema extensions
        schema_file = Path(__file__).parent.parent / "database" / "schema_extensions.sql"
        if not schema_file.exists():
            logger.error(f"Schema extensions file not found: {schema_file}")
            return 1
        
        sql_content = schema_file.read_text(encoding="utf-8")
        
        # Execute schema extensions
        logger.info("Executing schema extensions...")
        try:
            db_client.execute_update(sql_content)
            logger.info("Schema extensions applied successfully!")
            return 0
        except Exception as e:
            logger.error(f"Failed to apply schema extensions: {e}", exc_info=True)
            # Check if columns already exist (this is OK)
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                logger.info("Schema extensions already applied (this is OK)")
                return 0
            return 1
        
    except Exception as e:
        logger.error(f"Schema extension failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
