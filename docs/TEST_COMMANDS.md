# RAG Pipeline Test Commands

Quick reference for testing the RAG pipeline.

## Prerequisites

1. Apply schema extensions:
```bash
cd backend
python runners/apply_schema_extensions.py
```

2. Configure `.env` file with all required credentials

## Basic Tests

### Test Connection Only
```bash
# This will test connections but not process data
python -c "from config.config_loader import get_config; from database.postgres_client import PostgresClient; from vector_store.pinecone_client import PineconeClient; c = get_config(); db = PostgresClient(c.postgres_connection_string); pc = PineconeClient(c.pinecone_api_key, c.pinecone_index_name); print('✅ Config OK'); print('✅ DB:', db.test_connection()); print('✅ Pinecone:', pc.connect())"
```

### Test with Limited Records (Recommended First Run)
```bash
# Process only 5 records per entity type
python runners/run_rag_pipeline.py --limit 5
```

### Test Specific Entity Type
```bash
# Only process documents
python runners/run_rag_pipeline.py --entities document --limit 10

# Only process aircraft listings
python runners/run_rag_pipeline.py --entities aircraft_listing --limit 10
```

### Test Multiple Entity Types
```bash
# Process listings and documents only
python runners/run_rag_pipeline.py --entities aircraft_listing document --limit 20
```

## Full Production Run

```bash
# Process all entity types, all records
python runners/run_rag_pipeline.py
```

## Force Re-embedding

```bash
# Re-embed all records (useful for testing or after model change)
python runners/run_rag_pipeline.py --force-reembed --limit 10
```

## Check Results

### In PostgreSQL
```sql
-- Count embedded entities by type
SELECT entity_type, COUNT(*) as count, SUM(chunk_count) as total_chunks
FROM embeddings_metadata
WHERE entity_type IS NOT NULL
GROUP BY entity_type;

-- View recent embeddings
SELECT entity_type, entity_id, chunk_count, created_at
FROM embeddings_metadata
ORDER BY created_at DESC
LIMIT 20;
```

### In Pinecone
Check Pinecone dashboard or use Python:
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
print(f"Total vectors: {stats.get('total_vector_count', 0)}")
```

## Verification Checklist

After running the pipeline:

- [ ] Check logs for errors
- [ ] Verify embeddings_metadata has new records
- [ ] Check Pinecone stats show new vectors
- [ ] Test query (if query interface available)
- [ ] Verify no duplicate vectors (check vector IDs)

## Troubleshooting

### "No records to process"
- Check PostgreSQL has data in the tables
- Verify entity types are correct
- Check if all records are already embedded

### "Pinecone connection failed"
- Verify API key is correct
- Check index name matches
- Verify index exists in Pinecone dashboard

### "OpenAI API error"
- Check API key is valid
- Verify quota/rate limits
- Check model name is correct
