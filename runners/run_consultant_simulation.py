"""
Run multi-turn consultant simulations (full conversations) and score each step.

Usage:
  cd backend
  python runners/run_consultant_simulation.py --log-level INFO
  python runners/run_consultant_simulation.py --scenarios 1 2 3

This runner executes scenarios as *full chats* (history passed each turn), then applies a simple
deterministic scoring rubric:
  - Decision Quality (0–2)
  - Image Accuracy (0–2)
  - Conversation Behavior (0–2)

It also runs "red flag" checks:
  - house/hotel images
  - Eclipse/EA500 suggested for large/ULR intent
  - repeated long text
  - "I cannot find images" style refusals
  - no images when user asked to see/show
"""

from __future__ import annotations

import argparse
import re
import sys
import os
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


_HOUSE_NEEDLES = (
    "airbnb",
    "vrbo",
    "zillow",
    "realtor",
    "hotel",
    "resort",
    "motel",
    "cabinrental",
    "cabinsforyou",
    "greatsmokyvacations",
    "logcabins.co.uk",
    "brokenbow",
    "gatlinburg",
)

_REFUSAL_NEEDLES = (
    "i cannot find reliable interior images",
    "i cannot find reliable images",
    "i can't show images",
    "i cannot show images",
    "i don't have photos",
    "no verified images found",
)


def _low(s: Any) -> str:
    return str(s or "").lower()


def _safe_console(s: Any) -> str:
    """
    Windows PowerShell often uses cp1252; model outputs may include Unicode (≈, …, etc.).
    Replace characters that cannot be encoded so the simulation runner never crashes mid-suite.
    """
    raw = str(s or "")
    try:
        return raw.encode("cp1252", errors="replace").decode("cp1252", errors="replace")
    except Exception:
        # Fallback: strip non-ascii
        return "".join(ch if ord(ch) < 128 else "?" for ch in raw)


def _gallery_blob(out: Dict[str, Any]) -> str:
    ims = out.get("aircraft_images") or []
    parts: List[str] = []
    for im in ims:
        if not isinstance(im, dict):
            continue
        parts.append(str(im.get("url") or ""))
        parts.append(str(im.get("page_url") or ""))
        parts.append(str(im.get("description") or ""))
    return " ".join(parts).lower()


def _answer_blob(out: Dict[str, Any]) -> str:
    return _low(out.get("answer") or "")


def _has_red_flag(out: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    g = _gallery_blob(out)
    a = _answer_blob(out)
    if any(n in g for n in _HOUSE_NEEDLES):
        flags.append("house/hotel imagery detected")
    if any(n in a for n in _REFUSAL_NEEDLES):
        flags.append("image refusal / no-images phrasing")
    if "eclipse" in (a + " " + g) and any(x in (a + " " + g) for x in ("g650", "global 7500", "g700", "falcon 10x")):
        flags.append("eclipse shown in large/ULR context")
    return flags


def _asked_for_visual(q: str) -> bool:
    low = (q or "").lower()
    return bool(re.search(r"\b(show|see)\b", low)) or any(
        x in low for x in ("inside", "interior", "cabin", "cockpit", "again")
    )


def _score_step(
    *,
    user_query: str,
    out: Dict[str, Any],
    expected_any: Tuple[str, ...] = (),
    expected_none: Tuple[str, ...] = (),
    expected_class_keywords_any: Tuple[str, ...] = (),
    allow_no_images: bool = False,
) -> Tuple[int, int, int, List[str]]:
    """
    Return (decision,image,behavior,notes).
    Uses only text/url heuristics (no vision).
    """
    notes: List[str] = []
    a = _answer_blob(out)
    g = _gallery_blob(out)
    blob = f"{a} {g}"
    imgs = out.get("aircraft_images") or []
    img_n = len(imgs) if isinstance(imgs, list) else 0

    # Decision quality
    dq = 2
    if expected_any:
        if not any(t.lower() in blob for t in expected_any):
            dq = 0
            notes.append("missing expected aircraft/type cues")
        elif expected_class_keywords_any and not any(t.lower() in blob for t in expected_class_keywords_any):
            dq = 1
            notes.append("class alignment partially missing")
    if expected_none and any(t.lower() in blob for t in expected_none):
        dq = 0
        notes.append("contains forbidden aircraft/type cues")

    # Image accuracy
    ia = 2
    if _asked_for_visual(user_query):
        if img_n <= 0 and not allow_no_images:
            ia = 0
            notes.append("no images returned on visual request")
        else:
            # crude: require at least one expected cue in gallery blob
            if expected_any and not any(t.lower() in g for t in expected_any):
                ia = 1
                notes.append("gallery does not strongly match expected cues")
            if any(n in g for n in _HOUSE_NEEDLES):
                ia = 0
                notes.append("gallery contains house/hotel signals")
    else:
        ia = 2 if img_n == 0 else 1

    # Conversation behavior
    cb = 2
    if any(n in a for n in _REFUSAL_NEEDLES):
        cb = min(cb, 1)
        notes.append("refusal phrasing in answer")
    if len(a) > 2200 and _asked_for_visual(user_query):
        cb = 1
        notes.append("too verbose for a visual step")
    return dq, ia, cb, notes


def _scenario_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "id": 1,
            "title": "Clean -> Context Switch -> Reset",
            "turns": [
                {"q": "show me Challenger 300 cabin", "exp_any": ("Challenger 300",), "exp_none": ("airbnb", "vrbo")},
                {"q": "looks good", "allow_no_images": True},
                {
                    "q": "show me best cabin under $12M",
                    "exp_any": ("Challenger 300", "Latitude", "Falcon 2000"),
                    "exp_none": ("airbnb", "vrbo", "cabinsforyou", "greatsmokyvacations"),
                },
            ],
        },
        {
            "id": 2,
            "title": "Tail -> Visual -> Follow-up",
            "turns": [
                {"q": "Have N878BW?", "allow_no_images": True},
                {"q": "show me that", "exp_any": ("Eclipse", "EA500", "ECLIPSE 500"), "exp_none": ("airbnb",)},
                {"q": "interior?", "exp_any": ("Eclipse", "EA500", "ECLIPSE 500"), "exp_none": ("airbnb",)},
            ],
        },
        {
            "id": 3,
            "title": "Ambiguous Luxury Intent",
            "turns": [
                {"q": "I want something like G650 but cheaper", "exp_any": ("G500", "Falcon 7X", "Challenger 650")},
                {"q": "show me inside", "exp_any": ("G500", "Falcon 7X", "Challenger 650"), "exp_none": ("Eclipse", "Phenom")},
            ],
        },
        {
            "id": 4,
            "title": "Invalid Model Handling",
            "turns": [
                {"q": "Falcon 9000 interior", "exp_any": ("Falcon 900", "Falcon 8X"), "exp_none": ("Falcon 9000",)},
            ],
        },
        {
            "id": 5,
            "title": "Hard Reset Mid Conversation",
            "turns": [
                {"q": "show me CJ3 cockpit", "exp_any": ("CJ3",), "exp_none": ("airbnb",)},
                {"q": "nice", "allow_no_images": True},
                {"q": "what about best jet interior", "exp_any": ("Global 7500", "G700", "Falcon 10X"), "exp_none": ("CJ3",)},
            ],
        },
        {
            "id": 6,
            "title": "Comparison + Visual",
            "turns": [
                {"q": "compare Challenger 300 vs Latitude", "exp_any": ("Challenger 300", "Latitude")},
                {"q": "show both cabins", "exp_any": ("Challenger 300", "Latitude")},
            ],
        },
        {
            "id": 7,
            "title": "Real Buyer Flow",
            "turns": [
                {"q": "8 people, LA to Miami, no fuel stop, budget $10M", "exp_any": ("Challenger 300", "Falcon 2000")},
                {"q": "show me inside best option", "exp_any": ("Challenger 300", "Challenger 350", "Falcon 2000")},
            ],
        },
        {
            "id": 8,
            "title": "Repetition Stress",
            "turns": [
                {"q": "show me N878BW", "exp_any": ("Eclipse", "EA500", "ECLIPSE 500", "N878BW")},
                {"q": "show me again", "exp_any": ("Eclipse", "EA500", "ECLIPSE 500", "N878BW")},
                {"q": "again", "exp_any": ("Eclipse", "EA500", "ECLIPSE 500", "N878BW")},
            ],
        },
    ]


def _print_step(i: int, q: str, out: Dict[str, Any], score: Tuple[int, int, int], notes: List[str], flags: List[str]) -> None:
    dq, ia, cb = score
    total = dq + ia + cb
    ans = (out.get("answer") or "").strip()
    ims = out.get("aircraft_images") or []
    print(f"\n  Step {i}: {_safe_console(q)!r}", flush=True)
    print(f"    Score: decision={dq}/2 images={ia}/2 behavior={cb}/2  =>  {total}/6", flush=True)
    if notes:
        print(f"    Notes: {_safe_console(', '.join(notes[:6]))}", flush=True)
    if flags:
        print(f"    RED FLAGS: {_safe_console(', '.join(flags))}", flush=True)
    # Short preview
    prev = ans[:240].replace("\n", " ")
    print(f"    Answer preview: {_safe_console(prev)}{'...' if len(ans) > 240 else ''}", flush=True)
    if isinstance(ims, list) and ims:
        print(f"    Images: {len(ims)}", flush=True)
        for j, im in enumerate(ims[:3], 1):
            if not isinstance(im, dict):
                continue
            u = str(im.get('url') or '')
            p = str(im.get('page_url') or '')
            d = str(im.get('description') or '')
            print(f"      [{j}] {_safe_console(d[:70])} | {_safe_console(p or u)}", flush=True)
    else:
        print("    Images: 0", flush=True)
    # Always print key visual diagnostics (even when images exist) so we can debug drift/mismatch.
    if _asked_for_visual(q):
        du = out.get("data_used") if isinstance(out.get("data_used"), dict) else {}
        if isinstance(du, dict) and du:
            keys = (
                "consultant_show_image_ui_context",
                "consultant_user_asked_photos",
                "consultant_image_intent_src",
                "consultant_image_llm_intent",
                "consultant_searchapi_images_enabled",
                "consultant_searchapi_image_engine",
                "consultant_strict_tail_no_confirmed_searchapi_images",
                "consultant_strict_tail_no_confirmed_tavily_images",
                "consultant_tail_led_fallback_to_model_images",
                "consultant_strict_tail_fallback_to_model_images",
                "consultant_gallery_marketing_anchor",
                "consultant_multi_gallery_models",
                "aircraft_query_seed",
            )
            diag = {k: du.get(k) for k in keys if du.get(k) not in (None, "", False, {}, [])}
            if diag:
                print(f"    Debug: {_safe_console(diag)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-turn consultant simulations + scoring")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-context-chars", type=int, default=14000)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--scenarios", nargs="*", type=int, default=[], help="Scenario IDs to run (default all)")
    args = parser.parse_args()

    # Avoid buffered stdout when redirected to a file (needed for long-running suites).
    try:
        sys.stdout.reconfigure(line_buffering=True, errors="replace")
        sys.stderr.reconfigure(line_buffering=True, errors="replace")
    except Exception:
        pass

    setup_logging(log_level=args.log_level)

    config = Config.from_env()
    if not config.postgres_connection_string:
        logger.error("Missing POSTGRES_CONNECTION_STRING in .env")
        return 2

    # Pinecone + OpenAI are optional for *running*, but recommended for realism.
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
        if not pinecone.connect():
            logger.warning("Pinecone connect failed; proceeding (some RAG context may be thin).")
    db = PostgresClient(config.postgres_connection_string)

    service = RAGQueryService(
        embedding_service=embedding_service,
        pinecone_client=pinecone,
        postgres_client=db,
        openai_api_key=config.openai_api_key or "",
        chat_model=(os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip(),
    )

    scen_defs = _scenario_definitions()
    pick = set(args.scenarios or [])
    if pick:
        scen_defs = [s for s in scen_defs if int(s["id"]) in pick]

    grand_total = 0
    grand_max = 0
    for s in scen_defs:
        sid = int(s["id"])
        title = str(s.get("title") or "")
        print(f"\n=== Scenario {sid}: {_safe_console(title)} ===", flush=True)
        history: List[Dict[str, str]] = []
        scen_score = 0
        scen_max = 0
        prev_answer = ""
        for ti, t in enumerate(s["turns"], 1):
            q = str(t.get("q") or "")
            out = service.answer(
                q,
                top_k=int(args.top_k),
                max_context_chars=int(args.max_context_chars),
                score_threshold=args.score_threshold,
                history=history or None,
            )
            dq, ia, cb, notes = _score_step(
                user_query=q,
                out=out,
                expected_any=tuple(t.get("exp_any") or ()),
                expected_none=tuple(t.get("exp_none") or ()),
                allow_no_images=bool(t.get("allow_no_images")),
            )
            flags = _has_red_flag(out)

            # repetition stress: compare preview text similarity
            ans = (out.get("answer") or "").strip()
            if prev_answer and len(ans) > 120 and ans[:220] == prev_answer[:220]:
                cb = max(0, cb - 1)
                notes.append("likely repeated answer text")

            score = (dq, ia, cb)
            scen_score += sum(score)
            scen_max += 6
            _print_step(ti, q, out, score, notes, flags)

            # Advance chat history
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": ans})
            prev_answer = ans

        print(f"\nScenario {sid} total: {scen_score}/{scen_max}", flush=True)
        grand_total += scen_score
        grand_max += scen_max

    print(f"\n=== Grand total: {grand_total}/{grand_max} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

