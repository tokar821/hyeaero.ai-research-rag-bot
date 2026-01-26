# HyeAero Backend - RAG Pipeline

RAG (Retrieval-Augmented Generation) system that syncs data from PostgreSQL to Pinecone vector database.

## Architecture

```
PostgreSQL ──► RAG Pipeline ──► Pinecone
   ▲                              │
   └──────── embeddings_metadata ─┘
```

## Components

- **Vector Store**: Pinecone client wrapper
- **Embedding Service**: OpenAI text-embedding-3-large integration
- **RAG Pipeline**: Incremental sync from PostgreSQL to Pinecone
- **Chunking Service**: Text chunking for long documents
- **Database Client**: PostgreSQL connection for reading data

## Features

- ✅ Incremental updates (no duplicates, no missing data)
- ✅ Tracks embedding status in `embeddings_metadata` table
- ✅ Supports multiple entity types (listings, documents, aircraft, sales)
- ✅ Automatic chunking for long texts
- ✅ Idempotent operations (safe to run multiple times)
- ✅ Comprehensive logging to `logs/` folder

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your actual values:
```env
# PostgreSQL (from ETL pipeline)
POSTGRES_CONNECTION_STRING=postgres://...

# Pinecone
PINECONE_API_KEY=
PINECONE_HOST=
PINECONE_INDEX_NAME=
PINECONE_DIMENSION=124
PINECONE_METRIC=

# OpenAI (for embeddings)
OPENAI_API_KEY=your-openai-api-key
OPENAI_EMBEDDING_MODEL=
```

3. Run RAG pipeline:
```bash
python runners/run_rag_pipeline.py
```

## Usage

### Full Sync
```bash
python runners/run_rag_pipeline.py
```

### Sync Specific Entity Types
```bash
python runners/run_rag_pipeline.py --entities listings documents
```

### Limit Records (for testing)
```bash
python runners/run_rag_pipeline.py --limit 100
```

## Data Sources

The RAG pipeline processes:
1. **Aircraft Listings** - Descriptions, features, specifications
2. **Documents** - PDF/TXT extracted text
3. **Aircraft Master Data** - Manufacturer, model, specifications
4. **Sales Data** - Historical sales information
