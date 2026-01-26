-- Schema extensions for RAG pipeline
-- Extends embeddings_metadata to support multiple entity types

-- Add entity_type and entity_id columns to support tracking embeddings for:
-- - documents (existing)
-- - aircraft_listings
-- - aircraft (master data)
-- - aircraft_sales
-- - faa_registrations
-- etc.

ALTER TABLE embeddings_metadata 
ADD COLUMN IF NOT EXISTS entity_type VARCHAR(50),
ADD COLUMN IF NOT EXISTS entity_id UUID;

-- Create index for efficient lookups
CREATE INDEX IF NOT EXISTS idx_embeddings_entity_type_id 
ON embeddings_metadata(entity_type, entity_id) 
WHERE entity_type IS NOT NULL AND entity_id IS NOT NULL;

-- Create index for document_id (existing)
CREATE INDEX IF NOT EXISTS idx_embeddings_document_id 
ON embeddings_metadata(document_id) 
WHERE document_id IS NOT NULL;

-- Add comment
COMMENT ON COLUMN embeddings_metadata.entity_type IS 'Type of entity: document, aircraft_listing, aircraft, aircraft_sale, faa_registration, etc.';
COMMENT ON COLUMN embeddings_metadata.entity_id IS 'ID of the entity (references the entity table based on entity_type)';
