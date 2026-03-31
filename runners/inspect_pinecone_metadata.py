#!/usr/bin/env python3
"""Sample Pinecone query and print vector metadata (verify upsert schema).

From backend/:
  python runners/inspect_pinecone_metadata.py
  python runners/inspect_pinecone_metadata.py --query "Gulfstream G650 listing"

Requires PINECONE_* and OPENAI_API_KEY in .env.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config_loader import get_config
from rag.embedding_service import EmbeddingService
from rag.pinecone_metadata import infer_pinecone_entity_filter
from vector_store.pinecone_client import PineconeClient


def main() -> int:
    p = argparse.ArgumentParser(description="Print Pinecone match metadata sample")
    p.add_argument("--query", default="Gulfstream G650 aircraft", help="Query text to embed")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    cfg = get_config()
    if not cfg.openai_api_key or not cfg.pinecone_api_key:
        print("Missing OPENAI_API_KEY or PINECONE_API_KEY")
        return 1

    emb = EmbeddingService(cfg.openai_api_key, dimension=cfg.openai_embedding_dimension)
    pc = PineconeClient(
        api_key=cfg.pinecone_api_key,
        index_name=cfg.pinecone_index_name or "hyeaero-ai",
        dimension=cfg.pinecone_dimension,
        metric=cfg.pinecone_metric,
        host=cfg.pinecone_host,
    )
    if not pc.connect():
        print("Pinecone connect failed")
        return 1

    vec = emb.embed_text(args.query)
    if not vec:
        print("Embedding failed")
        return 1

    filt = infer_pinecone_entity_filter(args.query)
    print("query:", repr(args.query))
    print("inferred filter:", filt)
    matches = pc.query(vector=vec, top_k=args.top_k, filter=filt)
    print("matches:", len(matches))
    for i, m in enumerate(matches, 1):
        meta = getattr(m, "metadata", None) or {}
        vid = getattr(m, "id", None)
        score = getattr(m, "score", None)
        print(f"\n--- match {i} id={vid!r} score={score!r} ---")
        if not meta:
            print("(no metadata)")
            continue
        for k in sorted(meta.keys()):
            v = meta[k]
            vs = repr(v)
            if len(vs) > 200:
                vs = vs[:197] + "..."
            print(f"  {k}: {vs}")
        approx = sum(len(str(k)) + len(str(v)) for k, v in meta.items())
        print(f"  (~{approx} bytes metadata payload)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
