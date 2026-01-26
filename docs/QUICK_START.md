# RAG Pipeline Quick Start

Quick guide to get the RAG pipeline running.

## 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

## 2. Configure Environment

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
# Then edit .env with your actual credentials
```

## 3. Apply Schema Extensions

```bash
python runners/apply_schema_extensions.py
```

## 4. Run RAG Pipeline

```bash
# Full sync (all entity types)
python runners/run_rag_pipeline.py

# Test with limited records
python runners/run_rag_pipeline.py --limit 10

# Specific entity types
python runners/run_rag_pipeline.py --entities aircraft_listing document
```

## Expected Output

```
2026-01-25 XX:XX:XX | __main__ | INFO | RAG Pipeline - Syncing PostgreSQL to Pinecone
2026-01-25 XX:XX:XX | __main__ | INFO | Connected to PostgreSQL
2026-01-25 XX:XX:XX | __main__ | INFO | Connected to Pinecone
2026-01-25 XX:XX:XX | __main__ | INFO | Processing entity type: aircraft_listing
2026-01-25 XX:XX:XX | rag.rag_pipeline | INFO | Found 0 already embedded aircraft_listing records
2026-01-25 XX:XX:XX | rag.rag_pipeline | INFO | Fetched 100 aircraft_listing records to process
...
2026-01-25 XX:XX:XX | __main__ | INFO | RAG Pipeline Completed!
2026-01-25 XX:XX:XX | __main__ | INFO | Total Embedded: 100
2026-01-25 XX:XX:XX | __main__ | INFO | Total Vectors Upserted: 250
```

## Verify in Database

```sql
SELECT entity_type, COUNT(*) 
FROM embeddings_metadata 
GROUP BY entity_type;
```

## Verify in Pinecone

Check Pinecone dashboard or use the client to query stats.
