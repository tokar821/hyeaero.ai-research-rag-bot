"""Configuration loader for RAG pipeline.

Handles loading of environment settings for PostgreSQL, Pinecone, and OpenAI.
"""

import os
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


@dataclass
class Config:
    """Configuration for RAG pipeline."""
    
    # PostgreSQL configuration
    postgres_connection_string: Optional[str] = None
    postgres_host: Optional[str] = None
    postgres_port: Optional[int] = None
    postgres_database: Optional[str] = None
    postgres_user: Optional[str] = None
    postgres_password: Optional[str] = None
    
    # Pinecone configuration
    pinecone_api_key: Optional[str] = None
    pinecone_host: Optional[str] = None
    pinecone_index_name: Optional[str] = None
    pinecone_dimension: int = 1024
    pinecone_metric: str = "cosine"
    pinecone_region: Optional[str] = None
    
    # OpenAI configuration
    openai_api_key: Optional[str] = None
    openai_embedding_model: str = "text-embedding-3-large"
    openai_embedding_dimension: int = 1024
    
    # RAG Pipeline configuration
    chunk_size: int = 1000  # Characters per chunk
    chunk_overlap: int = 200  # Overlap between chunks
    batch_size: int = 100  # Batch size for Pinecone upserts
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        # PostgreSQL
        postgres_connection_string = os.getenv("POSTGRES_CONNECTION_STRING")
        postgres_host = os.getenv("POSTGRES_HOST")
        postgres_port = int(os.getenv("POSTGRES_PORT", "5432")) if os.getenv("POSTGRES_PORT") else None
        postgres_database = os.getenv("POSTGRES_DATABASE")
        postgres_user = os.getenv("POSTGRES_USER")
        postgres_password = os.getenv("POSTGRES_PASSWORD")
        
        # Pinecone
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_host = os.getenv("PINECONE_HOST")
        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "hyeaero-ai")
        pinecone_dimension = int(os.getenv("PINECONE_DIMENSION", "1024"))
        pinecone_metric = os.getenv("PINECONE_METRIC", "cosine")
        pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")
        
        # OpenAI
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        openai_embedding_dimension = int(os.getenv("OPENAI_EMBEDDING_DIMENSION", "1024"))
        
        # RAG Pipeline
        chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
        batch_size = int(os.getenv("RAG_BATCH_SIZE", "100"))
        
        return cls(
            postgres_connection_string=postgres_connection_string,
            postgres_host=postgres_host,
            postgres_port=postgres_port,
            postgres_database=postgres_database,
            postgres_user=postgres_user,
            postgres_password=postgres_password,
            pinecone_api_key=pinecone_api_key,
            pinecone_host=pinecone_host,
            pinecone_index_name=pinecone_index_name,
            pinecone_dimension=pinecone_dimension,
            pinecone_metric=pinecone_metric,
            pinecone_region=pinecone_region,
            openai_api_key=openai_api_key,
            openai_embedding_model=openai_embedding_model,
            openai_embedding_dimension=openai_embedding_dimension,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
        )
    
    def validate(self) -> None:
        """Validate that required configuration is present.
        
        Raises:
            ValueError: If required configuration is missing.
        """
        errors = []
        
        if not self.postgres_connection_string and not all([
            self.postgres_host, self.postgres_database, self.postgres_user
        ]):
            errors.append("PostgreSQL connection string or individual credentials required")
        
        if not self.pinecone_api_key:
            errors.append("PINECONE_API_KEY is required")
        
        if not self.pinecone_index_name:
            errors.append("PINECONE_INDEX_NAME is required")
        
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")


# Global config instance (lazy-loaded)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance.
    
    Returns:
        Config: The loaded configuration instance.
    """
    global _config
    if _config is None:
        _config = Config.from_env()
        _config.validate()
    return _config


def reload_config() -> Config:
    """Reload configuration from environment.
    
    Returns:
        Config: The newly loaded configuration instance.
    """
    global _config
    _config = Config.from_env()
    _config.validate()
    return _config
