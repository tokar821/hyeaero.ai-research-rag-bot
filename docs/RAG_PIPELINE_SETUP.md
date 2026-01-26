# RAG Pipeline Setup Guide

Complete guide for setting up and running the RAG pipeline that syncs PostgreSQL data to Pinecone.

## Architecture

```
PostgreSQL ──► RAG Pipeline ──► Pinecone
   ▲                              │
   └──────── embeddings_metadata ─┘
```

The RAG pipeline:
1. Reads data from PostgreSQL (aircraft listings, documents, sales, etc.)
2. Checks `embeddings_metadata` table to see what's already embedded
3. Only processes new/updated records (incremental updates)
4. Generates embeddings using OpenAI text-embedding-3-large
5. Stores vectors in Pinecone
6. Updates `embeddings_metadata` to track what's been embedded

## Prerequisites

1. **PostgreSQL Database**: Must have the ETL pipeline schema with `embeddings_metadata` table
2. **Pinecone Index**: Must exist with:
   - Dimension: 1024
   - Metric: cosine
   - Name: hyeaero-ai (or configured name)
3. **OpenAI API Key**: For generating embeddings
4. **Python 3.12+**: Backend language

## Setup Steps

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Apply Schema Extensions

The `embeddings_metadata` table needs to be extended to support multiple entity types:

```bash
python runners/apply_schema_extensions.py
```

This adds:
- `entity_type` column (e.g., 'aircraft_listing', 'document', 'aircraft')
- `entity_id` column (UUID reference to the entity)
- Indexes for efficient lookups

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your actual values:

```bash
cp .env.example .env
# Then edit .env with your actual credentials
```

The `.env.example` file contains all required variables. Here's what you need to configure:

```env
# PostgreSQL (from ETL pipeline)
POSTGRES_CONNECTION_STRING=postgres://user:password@host:port/database?sslmode=require

# Pinecone
PINECONE_API_KEY=pcsk_52gxSM_MPppEGNGoeER4uJ5EqfQr3ErfNnTbWHd63nmVy8pYiGyEWzDRB3Jtc8GC2vJFq4
PINECONE_HOST=https://hyeaero-ai-iibvc99.svc.aped-4627-b74a.pinecone.io
PINECONE_INDEX_NAME=hyeaero-ai
PINECONE_DIMENSION=1024
PINECONE_METRIC=cosine
PINECONE_REGION=us-east-1

# OpenAI
OPENAI_API_KEY=your-openai-api-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_EMBEDDING_DIMENSION=1024

# RAG Pipeline (optional)
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200
RAG_BATCH_SIZE=100
```

### 4. Verify Pinecone Index

Ensure your Pinecone index exists and matches:
- **Index Name**: `hyeaero-ai`
- **Dimension**: `1024`
- **Metric**: `cosine`
- **Region**: `us-east-1`

## Running the RAG Pipeline

### Full Sync (All Entity Types)

```bash
cd backend
python runners/run_rag_pipeline.py
```

This will:
- Process all entity types (listings, documents, aircraft, sales, faa_registrations)
- Only embed new/updated records (checks `embeddings_metadata`)
- Generate embeddings and store in Pinecone
- Update `embeddings_metadata` table

### Sync Specific Entity Types

```bash
# Only process aircraft listings and documents
python runners/run_rag_pipeline.py --entities aircraft_listing document

# Only process documents
python runners/run_rag_pipeline.py --entities document
```

### Test Mode (Limited Records)

```bash
# Process only 10 records per entity type
python runners/run_rag_pipeline.py --limit 10

# Process 50 listings and 20 documents
python runners/run_rag_pipeline.py --entities aircraft_listing document --limit 50
```

### Force Re-embedding

```bash
# Re-embed all records (even if already embedded)
python runners/run_rag_pipeline.py --force-reembed
```

## Entity Types

The pipeline supports these entity types:

1. **aircraft_listing** - Aircraft listings from Controller/AircraftExchange
2. **document** - PDF/TXT documents with extracted text
3. **aircraft** - Master aircraft data
4. **aircraft_sale** - Historical sales data
5. **faa_registration** - FAA registration data

## Incremental Updates

The pipeline is designed for **incremental updates**:

- ✅ **No Duplicates**: Checks `embeddings_metadata` before embedding
- ✅ **No Missing Data**: Processes all new/updated records
- ✅ **Idempotent**: Safe to run multiple times
- ✅ **Efficient**: Only processes what's needed

### How It Works

1. **Check Existing**: Queries `embeddings_metadata` for already embedded entities
2. **Fetch New**: Gets records from PostgreSQL that aren't in `embeddings_metadata`
3. **Process**: Extracts text, chunks if needed, generates embeddings
4. **Store**: Upserts vectors to Pinecone
5. **Track**: Updates `embeddings_metadata` with embedding info

## Monitoring

### Check Embedding Status

```sql
-- Count embedded entities by type
SELECT entity_type, COUNT(*) as count
FROM embeddings_metadata
WHERE entity_type IS NOT NULL
GROUP BY entity_type;

-- Check specific entity type
SELECT entity_type, entity_id, chunk_count, created_at
FROM embeddings_metadata
WHERE entity_type = 'aircraft_listing'
ORDER BY created_at DESC
LIMIT 10;
```

### Check Pinecone Stats

The pipeline logs Pinecone statistics. You can also query directly:

```python
from backend.vector_store.pinecone_client import PineconeClient
from backend.config.config_loader import get_config

config = get_config()
client = PineconeClient(
    api_key=config.pinecone_api_key,
    index_name=config.pinecone_index_name,
    dimension=config.pinecone_dimension
)
client.connect()
stats = client.get_stats()
print(stats)
```

## Troubleshooting

### "Pinecone index does not exist"
- Verify index name in `.env` matches your Pinecone index
- Check that index is in the correct region

### "No extractable text"
- Some records may not have text to embed (e.g., empty descriptions)
- This is normal - those records are skipped

### "Failed to generate embedding"
- Check OpenAI API key is valid
- Check API quota/rate limits
- Verify model name is correct

### "Database connection error"
- Verify PostgreSQL connection string
- Check database is accessible
- Ensure schema extensions are applied

## Performance Tips

1. **Batch Size**: Adjust `RAG_BATCH_SIZE` for optimal throughput
2. **Chunk Size**: Larger chunks = fewer vectors but more context per vector
3. **Limit Records**: Use `--limit` for testing before full sync
4. **Entity Types**: Process specific types if you only need certain data

## Next Steps

After running the RAG pipeline:
1. Verify data in Pinecone (check stats)
2. Test retrieval queries
3. Build RAG query interface
4. Integrate with frontend
