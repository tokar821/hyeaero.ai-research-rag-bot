# Logging Configuration

The RAG pipeline includes comprehensive logging that saves to files in the `logs/` folder.

## Log Files

### RAG Pipeline Logs
- **File**: `logs/rag_pipeline_log.txt`
- **Default**: Created automatically when running `run_rag_pipeline.py`
- **Content**: All pipeline execution logs including:
  - Connection status
  - Entity processing progress
  - Embedding generation
  - Pinecone upserts
  - Errors and warnings
  - Summary statistics

### Schema Extensions Logs
- **File**: `logs/schema_extensions_log.txt`
- **Default**: Created automatically when running `apply_schema_extensions.py`
- **Content**: Schema extension execution logs

## Log Format

All logs use a consistent format:
```
YYYY-MM-DD HH:MM:SS | module.name | LEVEL | message
```

Example:
```
2026-01-25 14:30:15 | rag.rag_pipeline | INFO | Processing entity type: aircraft_listing
2026-01-25 14:30:16 | rag.rag_pipeline | INFO | Found 0 already embedded aircraft_listing records
2026-01-25 14:30:17 | rag.embedding_service | DEBUG | Generated embeddings for batch 1: 50/50
```

## Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages (default)
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages with full tracebacks

## Configuration

### Default Behavior

By default, logs are:
- Written to console (stdout)
- Saved to file in `logs/` folder
- Overwritten on each run (not appended)

### Customize Log File Location

```bash
# Use custom log file path
python runners/run_rag_pipeline.py --log-file logs/custom_log.txt

# Use absolute path
python runners/run_rag_pipeline.py --log-file /path/to/logs/rag.log
```

### Change Log Level

```bash
# Show debug messages
python runners/run_rag_pipeline.py --log-level DEBUG

# Only show warnings and errors
python runners/run_rag_pipeline.py --log-level WARNING
```

## Log File Management

### Automatic Folder Creation

The `logs/` folder is created automatically if it doesn't exist. No manual setup required.

### Log Rotation

Logs are overwritten by default on each run. To append instead, modify the runner script or use a log rotation tool.

### Log File Size

Monitor log file sizes, especially with `DEBUG` level logging. Large log files can be:
- Archived periodically
- Rotated using system tools (logrotate on Linux)
- Deleted after review

## Example Log Output

```
2026-01-25 14:30:10 | __main__ | INFO | ============================================================
2026-01-25 14:30:10 | __main__ | INFO | RAG Pipeline - Syncing PostgreSQL to Pinecone
2026-01-25 14:30:10 | __main__ | INFO | ============================================================
2026-01-25 14:30:10 | __main__ | INFO | Configuration loaded successfully
2026-01-25 14:30:10 | __main__ | INFO | Initializing clients...
2026-01-25 14:30:11 | __main__ | INFO | Connected to PostgreSQL
2026-01-25 14:30:12 | __main__ | INFO | Connected to Pinecone
2026-01-25 14:30:12 | __main__ | INFO | Initialized embedding service: text-embedding-3-large
2026-01-25 14:30:12 | __main__ | INFO | Starting RAG pipeline sync...
2026-01-25 14:30:12 | rag.rag_pipeline | INFO | Processing entity type: aircraft_listing
2026-01-25 14:30:12 | rag.rag_pipeline | INFO | Found 0 already embedded aircraft_listing records
2026-01-25 14:30:13 | rag.rag_pipeline | INFO | Fetched 100 aircraft_listing records to process
2026-01-25 14:30:15 | rag.rag_pipeline | INFO | Successfully upserted 250 vectors to Pinecone
2026-01-25 14:30:15 | __main__ | INFO | RAG Pipeline Completed!
2026-01-25 14:30:15 | __main__ | INFO | Total Embedded: 100
2026-01-25 14:30:15 | __main__ | INFO | Total Vectors Upserted: 250
```

## Troubleshooting

### Logs Not Being Created

1. Check write permissions in the `backend/` directory
2. Verify the `logs/` folder can be created
3. Check disk space

### Too Much Logging

- Use `--log-level WARNING` to reduce log volume
- Review and adjust log levels in individual modules if needed

### Logs Too Verbose

- Use `--log-level INFO` (default) instead of `DEBUG`
- Logs are overwritten by default, so old logs won't accumulate

## Best Practices

1. **Review Logs After Each Run**: Check for errors or warnings
2. **Monitor Log File Size**: Especially with DEBUG logging
3. **Archive Important Logs**: Before running with `--force-reembed` or major changes
4. **Use Appropriate Log Levels**: DEBUG for development, INFO for production
