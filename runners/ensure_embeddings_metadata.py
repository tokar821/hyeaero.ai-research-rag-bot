"""Ensure ``embeddings_metadata`` exists (same contract as main RAG + PhlyData embed tracking)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import get_config
from database.ensure_embeddings_metadata import apply_embeddings_metadata_schema
from database.postgres_client import PostgresClient
from utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create/extend embeddings_metadata if missing")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    setup_logging(log_level=args.log_level, log_file="logs/ensure_embeddings_metadata_log.txt", log_file_overwrite=True)

    config = get_config()
    if not config.postgres_connection_string:
        logger.error("POSTGRES connection not configured")
        return 1
    db = PostgresClient(config.postgres_connection_string)
    if not db.test_connection():
        logger.error("PostgreSQL connection failed")
        return 1
    return 0 if apply_embeddings_metadata_schema(db) else 1


if __name__ == "__main__":
    sys.exit(main())
