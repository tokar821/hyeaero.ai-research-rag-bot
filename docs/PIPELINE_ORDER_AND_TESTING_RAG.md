# Pipeline Order and Testing RAG

## 1. After saving data in PostgreSQL, do we have to embed?

**Yes.** Saving to PostgreSQL (via the ETL loader) only puts data in the relational database. To use it for semantic search and Q&A you must run the **embed step**, which:

- Reads the five RAG entity types from PostgreSQL (aircraft_listing, document, aircraft, aircraft_sale, faa_registration)
- Chunks and embeds text with OpenAI
- Upserts vectors into **Pinecone**
- Records what was embedded in `embeddings_metadata`

So the order is:

1. **Save to PostgreSQL**  
   Run the ETL loader, e.g.  
   `python etl-pipeline/runners/run_database_loader.py` (or with source-specific flags).

2. **Embed (sync to Pinecone)**  
   From `backend`:  
   `python runners/run_rag_pipeline.py`  
   Optionally: `--entities ...` or `--limit N` for testing.

3. **After embed**  
   There is no further embed step. You use Pinecone (and any RAG/chat API you have) for:
   - **Semantic search**: embed a query → search Pinecone → get similar chunks.
   - **Q&A / RAG**: same retrieval + pass the chunks to an LLM to generate an answer.

Re-run the RAG pipeline whenever you want to refresh or add new data from PostgreSQL.

---

## 2. How can I test RAG logic?

### Option A: Test that embedding and sync work

1. **Embed a small set**  
   From `backend`:  
   `python runners/run_rag_pipeline.py --limit 50`  
   This embeds a few records per entity type. Check logs and DB/Pinecone (see TEST_COMMANDS.md).

2. **Check PostgreSQL**  
   ```sql
   SELECT entity_type, COUNT(*) FROM embeddings_metadata GROUP BY entity_type;
   ```

3. **Check Pinecone**  
   Use the Pinecone dashboard or the client’s `get_stats()` to see vector counts.

### Option B: Test retrieval (query → Pinecone → top chunks)

After data is in Pinecone, run the retrieval test script:

```bash
cd backend
python runners/test_rag_retrieval.py "What Citation jets are for sale?"
python runners/test_rag_retrieval.py "Gulfstream G650" --top-k 5
```

This script:

- Embeds your question with the same OpenAI model used by the pipeline
- Queries the Pinecone index for the top-k similar chunks
- Prints each match’s score, entity type, entity id, and a short text snippet

If you see relevant chunks with reasonable scores, **retrieval** is working.

### Option C: Full RAG (Pinecone → Postgres details → LLM answer)

If you have a backend API that does “ask a question → retrieve from Pinecone → call LLM → return answer”, test that API with a few questions and check that:

Run the full RAG query (retrieval + Postgres enrichment + LLM):

```bash
cd backend
python runners/run_rag_query.py "What Citation jets are for sale?"
python runners/run_rag_query.py "Gulfstream G650" --top-k 5
```

To test only retrieval (no LLM): `python runners/run_rag_query.py "Tell me about N12345" --retrieve-only`

This uses `rag/query_service.py`: it embeds the question, queries Pinecone, then for each match loads the full row from PostgreSQL (by `entity_type` + `entity_id`), builds context, and calls the LLM to generate an answer.

( If such an API does not exist yet, testing RAG today means: run the pipeline → run the RAG query script (see above) and verify retrieval. Adding a small “ask a question” endpoint that does retrieve + LLM is the next step for full RAG testing.

---

## Summary

| Step | What to run | What it does |
|------|-------------|--------------|
| 1. Save | ETL loader (`run_database_loader.py`) | Puts data into PostgreSQL |
| 2. Embed | RAG pipeline (`run_rag_pipeline.py`) | Syncs selected entities from PostgreSQL to Pinecone (chunk + embed + upsert) |
| 3. After embed | Your app / API / `test_rag_retrieval.py` | Search Pinecone and (optionally) generate answers with an LLM |

To **test RAG logic**: run the embed step with `--limit N`, then run `python runners/test_rag_retrieval.py "your question"` and check that the returned chunks are relevant.
