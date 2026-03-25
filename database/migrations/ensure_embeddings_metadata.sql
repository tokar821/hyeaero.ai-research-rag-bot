-- Idempotent: embeddings_metadata for RAG + PhlyData Pinecone tracking.
-- Safe to run multiple times. Pinecone vectors for phlydata_aircraft use namespace "phlydata_aircraft"
-- on the index from config (PINECONE_INDEX_NAME / config.pinecone_index_name).

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS embeddings_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID,
    embedding_model VARCHAR(100) NOT NULL,
    embedding_dimension INTEGER,
    chunk_count INTEGER,
    vector_store VARCHAR(100),
    vector_store_id VARCHAR(255),
    entity_type VARCHAR(50),
    entity_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE embeddings_metadata
    ADD COLUMN IF NOT EXISTS entity_type VARCHAR(50),
    ADD COLUMN IF NOT EXISTS entity_id UUID;

CREATE INDEX IF NOT EXISTS idx_embeddings_entity_type_id
    ON embeddings_metadata (entity_type, entity_id)
    WHERE entity_type IS NOT NULL AND entity_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_embeddings_document_id
    ON embeddings_metadata (document_id)
    WHERE document_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_embeddings_entity_model
    ON embeddings_metadata (entity_type, embedding_model)
    WHERE entity_type IS NOT NULL;

COMMENT ON TABLE embeddings_metadata IS 'Tracks embeddings synced to Pinecone (documents, listings, phlydata_aircraft namespace, etc.).';
COMMENT ON COLUMN embeddings_metadata.entity_type IS 'e.g. document, aircraft_listing, phlydata_aircraft';
COMMENT ON COLUMN embeddings_metadata.entity_id IS 'Primary key of the source row (UUID) when applicable';
COMMENT ON COLUMN embeddings_metadata.vector_store IS 'e.g. pinecone';
COMMENT ON COLUMN embeddings_metadata.vector_store_id IS 'Logical document id in vector store (e.g. phlydata_aircraft_{uuid})';
