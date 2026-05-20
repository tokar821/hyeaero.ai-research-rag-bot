"""
3-turn cabin shopping + refinement chain (manual / CI smoke).

  cd backend
  $env:RAG_RERANK_ENABLED='0'
  python runners/run_cabin_refinement_chain.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from runners.run_elite_advisor_10_turn import TURNS, _client_state_from_response, _engine_snapshot, _score_turn, run_simulation

CHAIN = [t for t in TURNS if t["turn"] in (5, 6, 7)]


def main() -> int:
    if not (os.getenv("RAG_RERANK_ENABLED") or "").strip():
        os.environ.setdefault("RAG_RERANK_ENABLED", "0")

    import runners.run_elite_advisor_10_turn as elite

    elite.TURNS = CHAIN
    out = Path(__file__).parent / "results" / "cabin_refinement_chain.json"
    return run_simulation(json_out=out, top_k=20, log_level="WARNING")


if __name__ == "__main__":
    raise SystemExit(main())
