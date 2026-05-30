"""
Live system integration audit — orchestration, RAG retrieval, images, multi-turn.

Requires backend/.env with keys for full pass (OPENAI_API_KEY, PINECONE_*, POSTGRES_*,
SEARCHAPI_API_KEY optional for image URL checks).

Usage:
  cd backend && set PYTHONPATH=. && python runners/run_live_system_integration_audit.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from services.orchestration.pipeline_orchestrator import run_consultant_orchestration  # noqa: E402


def _env_ok(name: str) -> bool:
    return bool((os.getenv(name) or "").strip())


def _check(label: str, ok: bool, detail: str = "") -> Dict[str, Any]:
    return {"check": label, "pass": ok, "detail": detail}


def audit_env() -> List[Dict[str, Any]]:
    rows = [
        _check("OPENAI_API_KEY", _env_ok("OPENAI_API_KEY")),
        _check("PINECONE_API_KEY", _env_ok("PINECONE_API_KEY")),
        _check("POSTGRES_CONNECTION_STRING", _env_ok("POSTGRES_CONNECTION_STRING")),
        _check("SEARCHAPI_API_KEY", _env_ok("SEARCHAPI_API_KEY"), "optional — images"),
    ]
    return rows


def audit_orchestration() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    q31 = (
        "Could a Praetor 600 realistically fly San Diego to Tokyo westbound winter "
        "8 passengers NBAA reserves without becoming dispatch-fragile?"
    )
    r31 = run_consultant_orchestration(q31)
    a31 = (r31.answer or "").lower()
    du31 = r31.data_used_patch or {}
    rows.append(
        _check(
            "Q31 Praetor SD–Tokyo not FEASIBLE",
            "verdict**: not realistic" in a31 or "not realistic" in a31,
            f"renderer={du31.get('orchestration_v2_renderer')} snippet={(r31.answer or '')[:120]}",
        )
    )
    rows.append(
        _check(
            "Q31 no directional disclaimer",
            "directional rather than catalog" not in a31,
        )
    )

    conv: Dict[str, Any] = {}
    r48a = run_consultant_orchestration(
        "We currently charter about 180 hours annually.", conversation_state=conv
    )
    conv.update(r48a.data_used_patch or {})
    r48 = run_consultant_orchestration(
        "Still assuming we prioritize low operating complexity, does fractional ownership "
        "now make more sense than full ownership?",
        conversation_state={"history": [], **conv},
    )
    a48 = (r48.answer or "").lower()
    du48 = r48.data_used_patch or {}
    rows.append(
        _check(
            "Q48 fractional ownership economics",
            du48.get("orchestration_v2_renderer") == "ownership_economics"
            or "ownership economics" in a48
            or "fractional" in a48,
            f"renderer={du48.get('orchestration_v2_renderer')}",
        )
    )

    conv2: Dict[str, Any] = {}
    r47a = run_consultant_orchestration(
        "We mostly operate Texas and Florida routes with occasional London travel.",
        conversation_state=conv2,
    )
    conv2.update(r47a.data_used_patch or {})
    r47 = run_consultant_orchestration(
        "Would your answer change if Tokyo became a quarterly mission?",
        conversation_state={"history": [], **conv2},
    )
    a47 = (r47.answer or "").lower()
    rows.append(
        _check(
            "Q47 Tokyo evolution (not Aspen-only)",
            "tokyo" in a47 and "aspen" not in a47[:200],
            (r47.answer or "")[:160],
        )
    )

    r43 = run_consultant_orchestration(
        "Create a broker-style acquisition brief for New York to London executive travel "
        "10 passengers lower operating cost than a Global 7500 emphasis on dispatch consistency."
    )
    a43 = (r43.answer or "").lower()
    rows.append(
        _check(
            "Q43 broker shortlist",
            "rank" in a43 or "shortlist" in a43 or "challenger" in a43,
        )
    )

    conv3: Dict[str, Any] = {}
    r49a = run_consultant_orchestration(
        "We operate mostly domestic U.S. flying.", conversation_state=conv3
    )
    conv3.update(r49a.data_used_patch or {})
    r49 = run_consultant_orchestration(
        "What if leadership suddenly insists on guaranteed nonstop Singapore capability?",
        conversation_state={"history": [], **conv3},
    )
    a49 = (r49.answer or "").lower()
    rows.append(
        _check(
            "Q49 Singapore follow-up strategic",
            "singapore" in a49
            and "no aircraft in the current validated catalog" not in a49,
            (r49.data_used_patch or {}).get("orchestration_v2_renderer", ""),
        )
    )

    return rows


def audit_images() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not _env_ok("SEARCHAPI_API_KEY"):
        rows.append(
            _check("SearchAPI images", False, "SEARCHAPI_API_KEY not set — skip live URLs")
        )
        return rows

    q = "Show verified exterior images of the Falcon 8X only."
    r = run_consultant_orchestration(q)
    imgs = r.aircraft_images or []
    rows.append(
        _check(
            "Falcon 8X image URLs returned",
            len(imgs) >= 1,
            f"count={len(imgs)}",
        )
    )
    if imgs:
        url = str(imgs[0].get("url") or "")
        rows.append(
            _check("Image URL http(s)", url.startswith("http"), url[:80]),
        )
    prose = (r.answer or "").lower()
    rows.append(
        _check(
            "Image prose not fail-closed only",
            "not available" not in prose or len(imgs) >= 1,
            (r.answer or "")[:100],
        )
    )
    return rows


def _build_rag_service():
    from config.config_loader import Config
    from database.postgres_client import PostgresClient
    from rag.embedding_service import EmbeddingService
    from rag.query_service import RAGQueryService
    from vector_store.pinecone_client import PineconeClient

    config = Config.from_env()
    if not all([config.openai_api_key, config.pinecone_api_key, config.postgres_connection_string]):
        raise RuntimeError("RAG not configured (OpenAI, Pinecone, Postgres)")
    embedding_service = EmbeddingService(
        api_key=config.openai_api_key,
        model=config.openai_embedding_model,
        dimension=config.openai_embedding_dimension,
    )
    pinecone = PineconeClient(
        api_key=config.pinecone_api_key,
        index_name=config.pinecone_index_name,
        dimension=config.pinecone_dimension,
        metric=config.pinecone_metric,
        host=config.pinecone_host,
    )
    if not pinecone.connect():
        raise RuntimeError("Pinecone connection failed")
    db = PostgresClient(config.postgres_connection_string)
    return RAGQueryService(
        embedding_service=embedding_service,
        pinecone_client=pinecone,
        postgres_client=db,
        openai_api_key=config.openai_api_key,
    )


def audit_rag() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not _env_ok("OPENAI_API_KEY"):
        rows.append(_check("RAG query service", False, "OPENAI_API_KEY missing"))
        return rows
    try:
        svc = _build_rag_service()
        out = svc.answer(
            "What is the typical range of a Gulfstream G650ER for broker planning?",
            top_k=12,
        )
        if out.get("error"):
            rows.append(_check("RAG answer", False, str(out["error"])[:200]))
            return rows
        ans = (out.get("answer") or "").strip()
        rows.append(
            _check(
                "RAG LLM answer non-empty",
                len(ans) > 80,
                f"len={len(ans)}",
            )
        )
        sources = out.get("sources") or []
        rows.append(
            _check(
                "RAG Pinecone/SQL sources",
                len(sources) >= 1,
                f"sources={len(sources)}",
            )
        )
        retrieved = svc.retrieve(
            "Gulfstream G650ER range nautical miles",
            top_k=8,
        )
        rows.append(
            _check(
                "RAG retrieve() returns chunks",
                len(retrieved) >= 1,
                f"chunks={len(retrieved)}",
            )
        )
    except Exception as exc:
        rows.append(_check("RAG query service", False, str(exc)[:200]))
    return rows


def main() -> int:
    sections: Dict[str, List[Dict[str, Any]]] = {
        "environment": audit_env(),
        "orchestration": audit_orchestration(),
        "images": audit_images(),
        "rag": audit_rag(),
    }
    all_rows: List[Dict[str, Any]] = []
    for sec_rows in sections.values():
        all_rows.extend(sec_rows)

    passed = sum(1 for r in all_rows if r["pass"])
    total = len(all_rows)

    out_path = _ROOT / "evals" / "live_system_integration_audit.json"
    payload = {
        "summary": {"pass": f"{passed}/{total}", "graded": total},
        "sections": sections,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload["summary"], indent=2))
    for sec, sec_rows in sections.items():
        print(f"\n## {sec}")
        for row in sec_rows:
            mark = "OK" if row["pass"] else "FAIL"
            detail = f" — {row['detail']}" if row.get("detail") else ""
            print(f"  [{mark}] {row['check']}{detail}")

    print(f"\nWrote {out_path}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
