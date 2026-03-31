-- Idempotent: log user questions to Ask Consultant (sync + stream) for internal analytics.
-- Applied automatically on API startup when PostgreSQL is configured (optional opt-out: CONSULTANT_QUERY_ANALYTICS_ENABLED=0).

CREATE TABLE IF NOT EXISTS consultant_query_log (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    query_text TEXT NOT NULL,
    endpoint VARCHAR(16) NOT NULL,
    history_turn_count INT NOT NULL DEFAULT 0,
    client_ip TEXT NULL,
    user_agent TEXT NULL
);

CREATE INDEX IF NOT EXISTS consultant_query_log_created_at_idx
    ON consultant_query_log (created_at DESC);

CREATE INDEX IF NOT EXISTS consultant_query_log_endpoint_created_idx
    ON consultant_query_log (endpoint, created_at DESC);

COMMENT ON TABLE consultant_query_log IS 'Opt-in log of user questions to /api/rag/answer and /api/rag/answer/stream';
