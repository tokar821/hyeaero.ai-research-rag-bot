"""
Verify Pinecone after structured re-embed:

- Index / namespace vector counts (:meth:`~vector_store.pinecone_client.PineconeClient.get_stats`)
- Sample query metadata keys (entity_type, entity_id, aircraft_model, …)
- Optional retrieval smoke test (:meth:`~rag.query_service.RAGQueryService.retrieve`)

Examples::

    python runners/verify_structured_pinecone_embeddings.py
    python runners/verify_structured_pinecone_embeddings.py --queries "Gulfstream G650 range" "N123AB registration"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import get_config
from database.postgres_client import PostgresClient
from rag.embedding_service import EmbeddingService
from rag.query_service import RAGQueryService
from rag.structured_reembed_constants import (
    PHLYDATA_AIRCRAFT_PINECONE_NAMESPACE,
    STRUCTURED_AVIATION_DEFAULT_NS_ENTITY_TYPES,
)
from vector_store.pinecone_client import PineconeClient

REQUIRED_META_KEYS = (
    "entity_type",
    "entity_id",
    "aircraft_model",
    "manufacturer",
    "serial_number",
    "tail_number",
    "year",
    "source_table",
)


def _approx_db_counts(db: PostgresClient) -> dict:
    out = {}
    mapping = [
        ("aircraft", "SELECT COUNT(*) AS c FROM aircraft"),
        ("aircraft_listing", "SELECT COUNT(*) AS c FROM aircraft_listings"),
        ("aircraft_sale", "SELECT COUNT(*) AS c FROM aircraft_sales"),
        ("faa_registration", "SELECT COUNT(*) AS c FROM faa_registrations"),
        ("aviacost_aircraft_detail", "SELECT COUNT(*) AS c FROM aviacost_aircraft_details"),
        ("aircraftpost_fleet_aircraft", "SELECT COUNT(*) AS c FROM aircraftpost_fleet_aircraft"),
        ("phlydata_aircraft", "SELECT COUNT(*) AS c FROM public.phlydata_aircraft"),
    ]
    for label, sql in mapping:
        try:
            rows = db.execute_query(sql)
            out[label] = int(rows[0]["c"]) if rows else 0
        except Exception:
            out[label] = None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify structured Pinecone embeddings + optional RAG retrieve")
    parser.add_argument(
        "--queries",
        nargs="*",
        default=[
            "Gulfstream G650 range",
            "Citation X listing price",
            "N123AB registration",
        ],
        help="Smoke-test queries for RAGQueryService.retrieve",
    )
    parser.add_argument("--skip-retrieve", action="store_true")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    config = get_config()
    if not all([config.postgres_connection_string, config.pinecone_api_key, config.openai_api_key]):
        print("Configure POSTGRES, PINECONE_API_KEY, OPENAI_API_KEY", file=sys.stderr)
        return 1

    db = PostgresClient(config.postgres_connection_string)
    if not db.test_connection():
        print("Postgres connection failed", file=sys.stderr)
        return 1

    pc = PineconeClient(
        api_key=config.pinecone_api_key,
        index_name=config.pinecone_index_name,
        dimension=config.pinecone_dimension,
        metric=config.pinecone_metric,
        host=config.pinecone_host,
    )
    if not pc.connect():
        print("Pinecone connect failed", file=sys.stderr)
        return 1

    stats = pc.get_stats()
    print("=== Pinecone describe_index_stats ===")
    print("total_vector_count:", stats.get("total_vector_count"))
    ns = stats.get("namespaces") or {}
    print("namespaces:", {k: getattr(v, "vector_count", v) for k, v in ns.items()})

    print("\n=== Approximate Postgres row counts (expect vector counts same order / ≥ for multi-chunk) ===")
    for k, v in _approx_db_counts(db).items():
        print(f"  {k}: {v}")

    print("\n=== Structured default-namespace entity types (metadata filter reference) ===")
    print(list(STRUCTURED_AVIATION_DEFAULT_NS_ENTITY_TYPES))
    print("Phly namespace:", PHLYDATA_AIRCRAFT_PINECONE_NAMESPACE)

    emb = EmbeddingService(
        api_key=config.openai_api_key,
        model=config.openai_embedding_model,
        dimension=config.openai_embedding_dimension,
    )
    q = "Gulfstream G650 business jet aircraft"
    vec = emb.embed_text(q)
    if not vec:
        print("Embedding failed", file=sys.stderr)
        return 1
    matches = pc.query(vector=vec, top_k=3, namespace="", filter=None)
    print("\n=== Sample vectors (default namespace, first 3 matches) ===")
    for i, m in enumerate(matches or [], 1):
        meta = getattr(m, "metadata", None) or {}
        print(f"  [{i}] id={getattr(m, 'id', '')} score={getattr(m, 'score', None)}")
        missing = [k for k in REQUIRED_META_KEYS if k not in meta or meta.get(k) in (None, "")]
        print(f"      metadata keys present: {sorted(meta.keys())}")
        if missing:
            print(f"      NOTE empty/missing: {missing}")
        else:
            print("      All canonical metadata keys non-empty.")

    if args.skip_retrieve:
        return 0

    rag = RAGQueryService(
        embedding_service=emb,
        pinecone_client=pc,
        postgres_client=db,
        openai_api_key=config.openai_api_key or "",
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
    )
    print("\n=== RAG retrieve smoke tests ===")
    for q in args.queries:
        rows = rag.retrieve(q, top_k=args.top_k, max_results=min(5, args.top_k))
        print(f"  Query: {q!r} -> {len(rows)} hydrated rows")
        for j, r in enumerate(rows[:3], 1):
            print(
                f"    [{j}] {r.get('entity_type')} id={r.get('entity_id')} "
                f"score={r.get('score')}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
