#!/usr/bin/env python3
"""
Sequential stress test for the Hye Aero Ask Consultant API (POST /api/rag/answer).

Loads queries from JSON (default: consultant_stress_test_queries.json next to this script),
sends each query in order, records latency, writes consultant_test_results.json.

Environment:
  CONSULTANT_API_URL / API_URL — base URL (default: http://127.0.0.1:8000)
  CONSULTANT_TEST_TOKEN — optional Bearer token if your deployment requires auth
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    raise


def _default_queries_path() -> Path:
    return Path(__file__).resolve().parent / "consultant_stress_test_queries.json"


def load_queries(path: Path) -> List[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Queries file must be a JSON array of strings")
    out: List[str] = []
    for i, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Invalid query at index {i}")
        out.append(item.strip())
    return out


def run(
    base_url: str,
    queries: List[str],
    timeout_sec: float,
    delay_sec: float,
    token: str | None,
) -> List[Dict[str, Any]]:
    base = base_url.rstrip("/")
    url = f"{base}/api/rag/answer"
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    session = requests.Session()
    results: List[Dict[str, Any]] = []

    for i, query in enumerate(queries, 1):
        payload = {"query": query}
        t0 = time.perf_counter()
        try:
            r = session.post(url, json=payload, headers=headers, timeout=timeout_sec)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            if r.ok:
                data = r.json()
                answer = data.get("answer")
                response_text = answer if isinstance(answer, str) else json.dumps(data, ensure_ascii=False)
            else:
                detail = ""
                try:
                    j = r.json()
                    detail = str(j.get("detail") or j.get("error") or j)
                except Exception:
                    detail = (r.text or "")[:2000]
                response_text = f"[HTTP {r.status_code}] {detail}"
        except requests.RequestException as e:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            response_text = f"[request error] {e}"

        results.append(
            {
                "query": query,
                "response": response_text,
                "latency_ms": elapsed_ms,
            }
        )
        if delay_sec > 0 and i < len(queries):
            time.sleep(delay_sec)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run consultant stress test against /api/rag/answer")
    parser.add_argument(
        "--base-url",
        default=os.getenv("CONSULTANT_API_URL") or os.getenv("API_URL") or "http://127.0.0.1:8000",
        help="API base URL (no trailing slash)",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=_default_queries_path(),
        help="JSON array of query strings",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("consultant_test_results.json"),
        help="Output JSON file (array of {query, response, latency_ms})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional delay in seconds between requests",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("CONSULTANT_TEST_TOKEN") or "",
        help="Bearer token (or set CONSULTANT_TEST_TOKEN)",
    )
    args = parser.parse_args()

    qpath = args.queries_file
    if not qpath.is_file():
        print(f"Queries file not found: {qpath}", file=sys.stderr)
        sys.exit(1)

    queries = load_queries(qpath)
    print(f"Loaded {len(queries)} queries from {qpath}")
    print(f"POST {args.base_url.rstrip('/')}/api/rag/answer")

    token = (args.token or "").strip() or None
    results = run(
        base_url=args.base_url,
        queries=queries,
        timeout_sec=args.timeout,
        delay_sec=args.delay,
        token=token,
    )

    out_path = args.output
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} results to {out_path.resolve()}")


if __name__ == "__main__":
    main()
