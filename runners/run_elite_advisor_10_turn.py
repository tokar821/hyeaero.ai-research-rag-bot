"""
Continuous 10-turn elite-advisor stress chat — exercises intent persistence,
conversation state engine, response mode router, and relevance-first answers.

Usage:
  cd backend
  python runners/run_elite_advisor_10_turn.py
  python runners/run_elite_advisor_10_turn.py --json-out results/elite_10_turn.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import Config
from database.postgres_client import PostgresClient
from rag.embedding_service import EmbeddingService
from rag.query_service import RAGQueryService
from utils.logger import setup_logging, get_logger
from vector_store.pinecone_client import PineconeClient

logger = get_logger(__name__)

TURNS: List[Dict[str, Any]] = [
    {
        "turn": 1,
        "q": "How many seats does a G650 have?",
        "kind": "factual_pinpoint",
        "expect_any": ("g650", "seat", "passenger"),
        "expect_short_words": 120,
        "forbid_in_answer": ("nm", "knot", "mach", "ownership program"),
    },
    {
        "turn": 2,
        "q": "What's the range of a Falcon 8X?",
        "kind": "factual_pinpoint",
        "expect_any": ("falcon", "8x", "nm", "range", "nautical"),
        "expect_short_words": 100,
        "forbid_in_answer": ("seat", "passenger", "price", "$"),
    },
    {
        "turn": 3,
        "q": "Price of a used Challenger 350?",
        "kind": "factual_pinpoint",
        "expect_any": ("challenger", "350", "$", "million", "price", "market"),
        "expect_short_words": 180,
    },
    {
        "turn": 4,
        "q": "Best jet for SF to London with 12 people?",
        "kind": "mission_advisory",
        "expect_any": ("london", "san francisco", "sf", "12", "ultra", "global", "g650", "g700", "falcon"),
        "expect_short_words": 350,
    },
    {
        "turn": 5,
        "q": "Show me modern cabin under $10M.",
        "kind": "visual_gallery",
        "expect_any": ("challenger", "praetor", "modern", "cabin", "10"),
        "wants_images": True,
        "expect_short_words": 120,
        "forbid_in_answer": ("nautical", " mach", "http"),
    },
    {
        "turn": 6,
        "q": "Something less corporate.",
        "kind": "refinement_followup",
        "expect_any": ("praetor", "falcon", "global", "corporate", "wood", "lighting", "lounge"),
        "inherits_budget": "$10",
        "expect_short_words": 120,
        "forbid_in_answer": ("what do you mean", "passenger count", "route"),
    },
    {
        "turn": 7,
        "q": "Bigger.",
        "kind": "refinement_followup",
        "expect_any": ("global", "falcon", "g500", "bigger", "cabin"),
        "inherits_context": True,
        "expect_short_words": 120,
        "forbid_in_answer": ("how many passengers", "phenom 300"),
    },
    {
        "turn": 8,
        "q": "Compare G700 vs Global 7500.",
        "kind": "comparison",
        "expect_any": ("g700", "global", "presence", "comfort", "refined", "dramatic"),
        "expect_both_models": ("g700", "global"),
        "expect_short_words": 200,
        "forbid_in_answer": ("nautical miles", " mach", "baggage", "cabin pressure"),
    },
    {
        "turn": 9,
        "q": "Show cockpit too.",
        "kind": "visual_followup",
        "expect_any": ("cockpit", "flight deck", "g700", "global"),
        "wants_images": True,
        "inherits_comparison": True,
    },
    {
        "turn": 10,
        "q": "I care more about cabin feel than speed.",
        "kind": "preference_followup",
        "expect_any": ("cabin", "feel", "atmosphere", "quiet", "lounge", "falcon", "global"),
        "forbid_in_answer": ("mach", "knots", "ktas", "runway", "climb", "dispatch"),
        "inherits_comparison": True,
        "expect_short_words": 200,
    },
]


def _safe(s: Any) -> str:
    raw = str(s or "")
    try:
        return raw.encode("cp1252", errors="replace").decode("cp1252", errors="replace")
    except Exception:
        return "".join(ch if ord(ch) < 128 else "?" for ch in raw)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def _client_state_from_response(data_used: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(data_used, dict):
        return None
    cs = data_used.get("consultant_conversation_state")
    if isinstance(cs, dict) and cs:
        return dict(cs)
    return None


def _engine_snapshot(data_used: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    du = data_used if isinstance(data_used, dict) else {}
    ip = du.get("intent_persistence") if isinstance(du.get("intent_persistence"), dict) else {}
    cse = du.get("conversation_state_engine") if isinstance(du.get("conversation_state_engine"), dict) else {}
    cs = du.get("consultant_conversation_state") if isinstance(du.get("consultant_conversation_state"), dict) else {}
    mem = cs.get("conversation_memory") if isinstance(cs.get("conversation_memory"), dict) else {}
    router = du.get("consultant_response_router") if isinstance(du.get("consultant_response_router"), dict) else {}

    return {
        "response_mode": du.get("consultant_response_mode_canonical") or du.get("consultant_response_mode"),
        "response_mode_legacy": du.get("consultant_response_mode"),
        "router_decision": router.get("decision") or router.get("mode"),
        "intent_routing": ip.get("routing_decision") or ip.get("routing"),
        "intent_inherited": ip.get("inherited_fields") or [],
        "resolved_aircraft": (ip.get("resolved_intent") or {}).get("active_aircraft")
        if isinstance(ip.get("resolved_intent"), dict)
        else None,
        "memory_active_aircraft": mem.get("active_aircraft"),
        "memory_active_category": mem.get("active_category"),
        "memory_last_visual": mem.get("last_visual_context"),
        "memory_budget": mem.get("active_budget"),
        "memory_comparison": mem.get("comparison_target"),
        "cse_inherited": cse.get("inherited_fields") or [],
        "current_aircraft_reference": cs.get("current_aircraft_reference"),
        "current_budget": cs.get("current_budget"),
        "image_count": len(du.get("aircraft_images") or []) if isinstance(du.get("aircraft_images"), list) else 0,
        "fine_intent": du.get("consultant_fine_intent") or du.get("fine_intent"),
    }


def _score_turn(spec: Dict[str, Any], out: Dict[str, Any]) -> Tuple[str, List[str], Dict[str, float]]:
    """Return grade PASS/PARTIAL/FAIL, notes, subscores."""
    notes: List[str] = []
    answer = (out.get("answer") or "").strip()
    al = answer.lower()
    du = out.get("data_used") if isinstance(out.get("data_used"), dict) else {}
    eng = _engine_snapshot(du)
    wc = _word_count(answer)
    sub = {"relevance": 1.0, "engine": 1.0, "visual": 1.0}

    exp_any = tuple(x.lower() for x in (spec.get("expect_any") or ()))
    if exp_any and not any(x in al for x in exp_any):
        sub["relevance"] = 0.0
        notes.append("answer missing expected topic cues")

    max_w = spec.get("expect_short_words")
    if max_w and wc > int(max_w):
        sub["relevance"] = min(sub["relevance"], 0.5)
        notes.append(f"verbose for pinpoint ask ({wc} words > {max_w})")

    for bad in spec.get("forbid_in_answer") or ():
        if bad.lower() in al:
            sub["relevance"] = min(sub["relevance"], 0.5)
            notes.append(f"likely over-answered: contains '{bad}'")

    both = spec.get("expect_both_models")
    if both:
        for m in both:
            if m.lower() not in al:
                sub["relevance"] = min(sub["relevance"], 0.5)
                notes.append(f"comparison missing '{m}'")

    if spec.get("wants_images") and eng["image_count"] <= 0:
        sub["visual"] = 0.0
        notes.append("no images on visual turn")

    if spec.get("inherits_budget"):
        bud = str(eng.get("memory_budget") or eng.get("current_budget") or "")
        if "10" not in bud and "$10" not in (du.get("consultant_conversation_state") or {}).__str__().lower():
            sub["engine"] = min(sub["engine"], 0.5)
            notes.append("budget context may not have persisted")

    if spec.get("inherits_context") or spec.get("inherits_comparison"):
        if not (eng.get("memory_active_aircraft") or eng.get("resolved_aircraft") or eng.get("intent_inherited")):
            if spec.get("inherits_comparison"):
                if not any(x in al for x in ("g700", "global", "7500")):
                    sub["engine"] = min(sub["engine"], 0.5)
                    notes.append("comparison context weak in engines/answer")
            else:
                sub["engine"] = min(sub["engine"], 0.5)
                notes.append("follow-up context inheritance not visible in engine snapshot")

    if "**" in answer:
        sub["relevance"] = min(sub["relevance"], 0.5)
        notes.append("markdown bold still present")

    avg = (sub["relevance"] + sub["engine"] + sub["visual"]) / 3.0
    if avg >= 0.85 and not notes:
        grade = "PASS"
    elif avg >= 0.55:
        grade = "PARTIAL"
    else:
        grade = "FAIL"

    return grade, notes, sub


def run_simulation(*, json_out: Optional[Path], top_k: int, log_level: str) -> int:
    # Avoid OOM on hosts without the BGE reranker model cached.
    if not (os.getenv("RAG_RERANK_ENABLED") or "").strip():
        os.environ.setdefault("RAG_RERANK_ENABLED", "0")
    setup_logging(log_level=log_level)
    config = Config.from_env()
    if not config.postgres_connection_string:
        logger.error("Missing POSTGRES_CONNECTION_STRING")
        return 2

    embedding_service = EmbeddingService(
        api_key=config.openai_api_key or "",
        model=config.openai_embedding_model,
        dimension=config.openai_embedding_dimension,
    )
    pinecone = PineconeClient(
        api_key=config.pinecone_api_key or "",
        index_name=config.pinecone_index_name,
        dimension=config.pinecone_dimension,
        metric=config.pinecone_metric,
        host=config.pinecone_host,
    )
    if (config.pinecone_api_key or "").strip():
        pinecone.connect()

    service = RAGQueryService(
        embedding_service=embedding_service,
        pinecone_client=pinecone,
        postgres_client=PostgresClient(config.postgres_connection_string),
        openai_api_key=config.openai_api_key or "",
        chat_model=(os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip(),
    )

    history: List[Dict[str, str]] = []
    conversation_state: Optional[Dict[str, Any]] = None
    report_turns: List[Dict[str, Any]] = []
    grades: Dict[str, int] = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}

    print("\n=== Elite Advisor — 10-turn continuous simulation ===\n", flush=True)

    for spec in TURNS:
        ti = int(spec["turn"])
        q = str(spec["q"])
        t0 = time.perf_counter()
        print(f"Turn {ti}/10: {_safe(q)}", flush=True)

        out = service.answer(
            q,
            top_k=top_k,
            history=history or None,
            conversation_state=conversation_state,
        )
        elapsed = round(time.perf_counter() - t0, 2)
        answer = (out.get("answer") or "").strip()
        du = out.get("data_used") if isinstance(out.get("data_used"), dict) else {}
        eng = _engine_snapshot(du)
        grade, notes, sub = _score_turn(spec, out)
        grades[grade] = grades.get(grade, 0) + 1

        report_turns.append(
            {
                "turn": ti,
                "query": q,
                "kind": spec.get("kind"),
                "grade": grade,
                "elapsed_sec": elapsed,
                "word_count": _word_count(answer),
                "answer": answer,
                "notes": notes,
                "subscores": sub,
                "engine": eng,
                "error": out.get("error"),
            }
        )

        print(f"  Grade: {grade}  ({elapsed}s, { _word_count(answer)} words, {eng['image_count']} imgs)", flush=True)
        print(f"  Mode: {eng.get('response_mode')}  |  Intent routing: {eng.get('intent_routing')}", flush=True)
        print(f"  Memory aircraft: {eng.get('memory_active_aircraft')}  |  Budget: {eng.get('memory_budget')}", flush=True)
        if notes:
            print(f"  Notes: {_safe('; '.join(notes))}", flush=True)
        prev = answer[:300].replace("\n", " ")
        print(f"  Answer: {_safe(prev)}{'...' if len(answer) > 300 else ''}\n", flush=True)

        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})
        conversation_state = _client_state_from_response(du)

    summary = {
        "title": "Elite Advisor 10-turn continuous simulation",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "turns": len(TURNS),
        "grades": grades,
        "pass_rate": round(grades.get("PASS", 0) / len(TURNS), 2),
        "report": report_turns,
    }

    print("=== Summary ===", flush=True)
    print(f"PASS: {grades.get('PASS', 0)}  PARTIAL: {grades.get('PARTIAL', 0)}  FAIL: {grades.get('FAIL', 0)}", flush=True)
    print(f"Pass rate (strict PASS only): {summary['pass_rate']:.0%}\n", flush=True)

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON report: {json_out}", flush=True)

    return 0 if grades.get("FAIL", 0) == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--log-level", default="WARNING")
    parser.add_argument(
        "--json-out",
        default="runners/results/elite_advisor_10_turn.json",
        help="Path for JSON report (relative to backend/)",
    )
    args = parser.parse_args()
    backend_root = Path(__file__).parent.parent
    out_path = Path(args.json_out) if Path(args.json_out).is_absolute() else backend_root / args.json_out
    return run_simulation(json_out=out_path, top_k=args.top_k, log_level=args.log_level)


if __name__ == "__main__":
    raise SystemExit(main())
