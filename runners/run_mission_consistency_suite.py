"""
Mission consistency regression suite — multi-turn decomposition stability.

Usage:
  cd backend && python runners/run_mission_consistency_suite.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

env_path = _ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

from evals.mission_consistency_suite import (  # noqa: E402
    CONSISTENCY_SCENARIOS,
    score_consistency,
)
from services.orchestration.pipeline_orchestrator import run_consultant_orchestration  # noqa: E402


def main() -> int:
    results = []
    passed = 0
    for sc in CONSISTENCY_SCENARIOS:
        data_used: dict = {"consultant_response_mode": "mission_advisory"}
        history: list = []
        orch1 = run_consultant_orchestration(
            sc.query,
            conversation_state={"history": history},
            data_used=data_used,
            query_intent="mission_feasibility",
        )
        # Persist state for follow-up turn and packet access (runner owns the data_used dict).
        if isinstance(orch1.data_used_patch, dict):
            data_used.update(orch1.data_used_patch)
        answer1 = orch1.answer or ""
        pkt1 = data_used.get("mission_understanding_packet")
        history.append({"role": "user", "content": sc.query})
        history.append({"role": "assistant", "content": answer1})

        answer2 = None
        pkt2 = None
        if sc.follow_up:
            orch2 = run_consultant_orchestration(
                sc.follow_up,
                conversation_state={"history": history},
                data_used=data_used,
                query_intent="mission_feasibility",
            )
            answer2 = orch2.answer or ""
            if isinstance(orch2.data_used_patch, dict):
                data_used.update(orch2.data_used_patch)
            pkt2 = data_used.get("mission_understanding_packet")

        scored = score_consistency(
            scenario=sc,
            answer=answer1,
            follow_up_answer=answer2,
            packet=pkt1 if isinstance(pkt1, dict) else None,
            turn2_packet=pkt2 if isinstance(pkt2, dict) else None,
        )
        if scored.passed:
            passed += 1
        results.append({"title": sc.title, **scored.to_dict()})
        status = "PASS" if scored.passed else "FAIL"
        print(f"[{status}] {sc.id}: score={scored.score:.2f} issues={scored.issues}")

    out_path = _ROOT / "evals" / "mission_consistency_results.json"
    out_path.write_text(json.dumps({"passed": passed, "total": len(CONSISTENCY_SCENARIOS), "results": results}, indent=2))
    print(f"\n{passed}/{len(CONSISTENCY_SCENARIOS)} passed — wrote {out_path}")
    return 0 if passed == len(CONSISTENCY_SCENARIOS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
