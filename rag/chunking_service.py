"""Text chunking service for long documents."""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ChunkingService:
    """Service for chunking long texts into smaller pieces."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """Initialize chunking service.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_id_prefix: str = "chunk"
    ) -> List[Dict[str, Any]]:
        """Chunk text into smaller pieces.
        
        Args:
            text: Text to chunk
            metadata: Base metadata to include in each chunk
            chunk_id_prefix: Prefix for chunk IDs
            
        Returns:
            List of chunks with text and metadata
        """
        if not text or not text.strip():
            return []
        
        text = text.strip()
        chunks = []
        
        # If text is shorter than chunk_size, return as single chunk
        if len(text) <= self.chunk_size:
            return [{
                'text': text,
                'chunk_index': 0,
                'metadata': metadata or {}
            }]
        
        # Chunk with overlap
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary (prefer period, then newline, then space)
            if end < len(text):
                # Look for sentence boundary near the end
                for boundary in ['. ', '.\n', '\n\n', '\n', ' ']:
                    boundary_pos = text.rfind(boundary, start, end)
                    if boundary_pos != -1:
                        end = boundary_pos + len(boundary)
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_metadata = (metadata or {}).copy()
                chunk_metadata['chunk_index'] = chunk_index
                chunk_metadata['chunk_start'] = start
                chunk_metadata['chunk_end'] = end
                chunk_metadata['total_chunks'] = None  # Will be set after all chunks
                
                chunks.append({
                    'text': chunk_text,
                    'chunk_index': chunk_index,
                    'metadata': chunk_metadata
                })
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        # Update total_chunks in all chunks
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)
        
        logger.debug(f"Chunked text into {len(chunks)} chunks (size: {self.chunk_size}, overlap: {self.chunk_overlap})")
        return chunks
