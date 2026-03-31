"""Chunking service: entity-based routing for structured rows, character windows for documents.

Structured entity types (aircraft, listings, FAA, sales, Aviacost, AircraftPost, Phly export)
are embedded as **one** text block per record so field context stays together.

Long unstructured text (``document``) still uses sliding character windows with overlap.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, FrozenSet, List, Optional

logger = logging.getLogger(__name__)

# RAG pipeline entity_type values that must stay a single embedding per DB row.
STRUCTURED_SINGLE_CHUNK_ENTITY_TYPES: FrozenSet[str] = frozenset(
    {
        "aircraft",
        "aircraft_listing",
        "aircraft_sale",
        "faa_registration",
        "aviacost_aircraft_detail",
        "aircraftpost_fleet_aircraft",
        "phlydata_aircraft",
    }
)


class ChunkingService:
    """Chunking: one structured block per entity for tabular sources; multi-chunk for documents."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_for_entity(
        self,
        entity_type: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_id_prefix: str = "chunk",
    ) -> List[Dict[str, Any]]:
        """
        Route by ``entity_type``:

        - Structured tables → exactly **one** chunk (full extracted text), regardless of length
          (upstream ``RAGPipeline`` still skips rows over ``MAX_TEXT_LENGTH``).
        - ``document`` and any unknown type → :meth:`chunk_text` (character-based, backward compatible).
        """
        if not text or not str(text).strip():
            return []
        et = (entity_type or "").strip()
        if et in STRUCTURED_SINGLE_CHUNK_ENTITY_TYPES:
            return self._single_structured_chunk(text, metadata)
        return self.chunk_text(text, metadata, chunk_id_prefix=chunk_id_prefix)

    def _single_structured_chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        text = text.strip()
        chunk_metadata = (metadata or {}).copy()
        chunk_metadata["chunk_index"] = 0
        chunk_metadata["chunk_start"] = 0
        chunk_metadata["chunk_end"] = len(text)
        chunk_metadata["total_chunks"] = 1
        chunk_metadata["chunking_strategy"] = "entity_single"
        return [
            {
                "text": text,
                "chunk_index": 0,
                "metadata": chunk_metadata,
            }
        ]

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_id_prefix: str = "chunk",
    ) -> List[Dict[str, Any]]:
        """Legacy character-based chunking (documents and generic long text).

        The ``chunk_id_prefix`` argument is accepted for API compatibility; it is not stored on chunks.
        """
        if not text or not text.strip():
            return []

        text = text.strip()
        chunks: List[Dict[str, Any]] = []

        if len(text) <= self.chunk_size:
            md = (metadata or {}).copy()
            md["chunk_index"] = 0
            md["chunk_start"] = 0
            md["chunk_end"] = len(text)
            md["total_chunks"] = 1
            md["chunking_strategy"] = "character_window"
            return [{"text": text, "chunk_index": 0, "metadata": md}]

        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            if end < len(text):
                for boundary in [". ", ".\n", "\n\n", "\n", " "]:
                    boundary_pos = text.rfind(boundary, start, end)
                    if boundary_pos != -1:
                        end = boundary_pos + len(boundary)
                        break

            piece = text[start:end].strip()

            if piece:
                chunk_metadata = (metadata or {}).copy()
                chunk_metadata["chunk_index"] = chunk_index
                chunk_metadata["chunk_start"] = start
                chunk_metadata["chunk_end"] = end
                chunk_metadata["total_chunks"] = None
                chunk_metadata["chunking_strategy"] = "character_window"
                chunks.append(
                    {
                        "text": piece,
                        "chunk_index": chunk_index,
                        "metadata": chunk_metadata,
                    }
                )
                chunk_index += 1

            next_start = end - self.chunk_overlap
            start = max(next_start, start + 1) if next_start <= start else next_start
            if start >= len(text):
                break

        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)

        logger.debug(
            "Character chunking: %s chunks (size=%s overlap=%s prefix=%r)",
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
            chunk_id_prefix,
        )
        return chunks
