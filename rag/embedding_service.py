"""Embedding service using OpenAI text-embedding-3-large."""

import logging
from typing import List, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using OpenAI."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        dimension: int = 1024
    ):
        """Initialize embedding service.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model name
            dimension: Embedding dimension
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimension = dimension
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if error
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text.strip(),
                dimensions=self.dimension
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for API calls
            
        Returns:
            List of embedding vectors (None for failed embeddings)
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Filter out empty texts
            valid_batch = [(idx, text) for idx, text in enumerate(batch) if text and text.strip()]
            
            if not valid_batch:
                embeddings.extend([None] * len(batch))
                continue
            
            try:
                # Prepare texts for batch embedding
                batch_texts = [text for _, text in valid_batch]
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts,
                    dimensions=self.dimension
                )
                
                # Map results back to original positions
                batch_embeddings = [None] * len(batch)
                for (orig_idx, _), embedding_data in zip(valid_batch, response.data):
                    batch_embeddings[orig_idx] = embedding_data.embedding
                
                embeddings.extend(batch_embeddings)
                logger.debug(f"Generated embeddings for batch {i//batch_size + 1}: {len(valid_batch)}/{len(batch)}")
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1}: {e}")
                embeddings.extend([None] * len(batch))
        
        return embeddings
