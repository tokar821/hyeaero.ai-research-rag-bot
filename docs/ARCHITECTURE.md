# RAG Pipeline Architecture

## Overview

The RAG (Retrieval-Augmented Generation) pipeline syncs structured data from PostgreSQL to Pinecone vector database for semantic search and retrieval.

## Data Flow

```
PostgreSQL Tables
    │
    ├── aircraft_listings
    ├── documents
    ├── aircraft
    ├── aircraft_sales
    └── faa_registrations
    │
    ▼
Entity Extractors
    │
    ├── Extract text from records
    ├── Generate metadata
    └── Chunk long texts
    │
    ▼
Embedding Service (OpenAI)
    │
    ├── text-embedding-3-large
    ├── Dimension: 1024
    └── Generate embeddings
    │
    ▼
Pinecone Vector Store
    │
    ├── Index: hyeaero-ai
    ├── Dimension: 1024
    ├── Metric: cosine
    └── Store vectors with metadata
    │
    ▼
embeddings_metadata (PostgreSQL)
    │
    └── Track what's been embedded
```

## Components

### 1. Entity Extractors (`rag/entity_extractors.py`)

Converts database records into text suitable for embedding:

- **AircraftListingExtractor**: Extracts listing descriptions, features, specs
- **DocumentExtractor**: Extracts text from PDF/TXT documents
- **AircraftExtractor**: Extracts aircraft master data
- **AircraftSaleExtractor**: Extracts sales information
- **FAARegistrationExtractor**: Extracts FAA registration data

### 2. Chunking Service (`rag/chunking_service.py`)

Splits long texts into smaller chunks:
- Configurable chunk size (default: 1000 characters)
- Overlap between chunks (default: 200 characters)
- Smart sentence boundary detection
- Preserves metadata across chunks

### 3. Embedding Service (`rag/embedding_service.py`)

Generates embeddings using OpenAI:
- Model: `text-embedding-3-large`
- Dimension: 1024
- Batch processing support
- Error handling for failed embeddings

### 4. Pinecone Client (`vector_store/pinecone_client.py`)

Manages Pinecone vector database:
- Connection management
- Batch upserts
- Query interface
- Statistics retrieval

### 5. RAG Pipeline (`rag/rag_pipeline.py`)

Orchestrates the entire process:
- Fetches records from PostgreSQL
- Checks `embeddings_metadata` for existing embeddings
- Processes only new/updated records
- Generates embeddings and stores in Pinecone
- Updates `embeddings_metadata` table

## Incremental Update Logic

### How It Prevents Duplicates

1. **Query Existing**: Before processing, queries `embeddings_metadata`:
   ```sql
   SELECT entity_id FROM embeddings_metadata
   WHERE entity_type = 'aircraft_listing' 
     AND embedding_model = 'text-embedding-3-large'
   ```

2. **Filter Records**: Only processes records not in the existing set

3. **Track New**: After embedding, inserts into `embeddings_metadata`:
   ```sql
   INSERT INTO embeddings_metadata (
       entity_type, entity_id, embedding_model, 
       chunk_count, vector_store_id
   ) VALUES (...)
   ```

### How It Prevents Missing Data

- Processes all records that aren't in `embeddings_metadata`
- Can be run multiple times safely (idempotent)
- Tracks by `entity_type + entity_id + embedding_model`

## Vector ID Format

Vectors in Pinecone use this ID format:
```
{entity_type}_{entity_id}_chunk_{chunk_index}
```

Example:
- `aircraft_listing_123e4567-e89b-12d3-a456-426614174000_chunk_0`
- `document_789e4567-e89b-12d3-a456-426614174000_chunk_1`

## Metadata Structure

Each vector in Pinecone includes metadata:

```json
{
  "entity_type": "aircraft_listing",
  "entity_id": "123e4567-e89b-12d3-a456-426614174000",
  "source_platform": "controller",
  "manufacturer": "Cessna",
  "model": "Citation CJ3+",
  "year": 2020,
  "listing_status": "For Sale",
  "chunk_index": 0,
  "chunk_start": 0,
  "chunk_end": 1000,
  "total_chunks": 3
}
```

## Database Schema Extensions

The `embeddings_metadata` table is extended with:

- `entity_type`: Type of entity (aircraft_listing, document, etc.)
- `entity_id`: UUID reference to the entity
- Existing `document_id` column (for backward compatibility)

This allows tracking embeddings for any entity type, not just documents.

## Error Handling

- **Failed Embeddings**: Logged and skipped, doesn't stop pipeline
- **Pinecone Errors**: Logged with full traceback
- **Database Errors**: Rolled back, logged, pipeline continues
- **Missing Text**: Records with no extractable text are skipped

## Performance Considerations

1. **Batch Processing**: Embeddings generated in batches (default: 100)
2. **Chunking**: Long texts split to stay within embedding limits
3. **Incremental**: Only processes new records
4. **Parallel**: Can process multiple entity types sequentially

## Future Enhancements

- Parallel processing of entity types
- Incremental updates based on `updated_at` timestamps
- Vector similarity search API
- RAG query interface
- Automatic re-embedding of updated records
