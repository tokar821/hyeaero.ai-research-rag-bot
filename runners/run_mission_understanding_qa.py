"""
Mission Understanding QA runner (broker-realistic scenarios).

Prints mission-understanding-driven output characteristics without focusing on scoring accuracy.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure backend is importable when running as a script.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.consultant.intelligence_engine import run_consultant_intelligence_layer
from services.state.mission_state import sync_persistent_mission_state


def _call_intelligence(
    *,
    query: str,
    history: List[Dict[str, str]] | None = None,
    data_used: Dict[str, Any] | None = None,
    conversation_state: Dict[str, Any] | None = None,
) -> str:
    data_used = data_used or {"consultant_response_mode": "mission_advisory"}
    history = history or []
    result = run_consultant_intelligence_layer(
        answer="Draft (ignore) — running mission understanding QA.",
        query=query,
        history=history,
        data_used=data_used,
        conversation_state=conversation_state,
    )
    # Keep output compact but concrete.
    txt = result.answer or ""
    # Avoid Windows console Unicode encode issues.
    txt = (
        txt.replace("→", "->")
        .replace("–", "-")
        .replace("\u2011", "-")
        .replace("\u202f", " ")
    )
    try:
        return txt.encode("ascii", "ignore").decode("ascii", "ignore")
    except Exception:
        return txt


def main() -> int:
    os.environ.setdefault("CONSULTANT_INTELLIGENCE_LAYER", "1")
    os.environ.setdefault("CONSULTANT_ORCHESTRATION", "1")

    scenarios: List[Dict[str, Any]] = [
        {
            "name": "Enterprise Europe shuttle (hidden priorities)",
            "query": (
                "We have 1000 employees, 100+ million revenue, and about 50 Europe trips per year "
                "with 12 people. Executive transport is the priority; nonstop reliability matters "
                "because delays harm schedule. Recommend an aircraft class for this posture."
            ),
            "history": [],
        },
        {
            "name": "Incomplete Europe twice monthly (no city pair)",
            "query": (
                "Need something for Europe twice monthly for 12 executives. We care more about "
                "nonstop reliability and airport access than cabin luxury. Recommend what to consider."
            ),
            "history": [],
        },
        {
            "name": "Multi-role SFO→Tokyo+London nonstop westbound winter",
            "query": (
                "SFO to Tokyo and London nonstop westbound winter for 8 executives. "
                "Recommend an aircraft class that can handle both legs reliably."
            ),
            "history": [],
        },
        {
            "name": "Conversational continuity: Europe → add Aspen mountain access",
            "turn1": (
                "We run an enterprise shuttle to Europe: 1200 employees, 12 executives, "
                "nonstop preferred and dispatch reliability matters. Budget around $8M. "
                "Recommend an acquisition approach."
            ),
            "turn2": (
                "Also sometimes need Aspen mountain access for the team. Keep the same "
                "nonstop reliability posture. Recommend again."
            ),
        },
        {
            "name": "Island operations & runway over luxury (hidden constraints)",
            "query": (
                "Miami Caribbean South America for 8 passengers. We want runway flexibility "
                "and simple dispatch; luxury cabin is secondary. Recommend an aircraft class."
            ),
            "history": [],
        },
    ]

    data_used: Dict[str, Any] = {"consultant_response_mode": "mission_advisory"}

    for s in scenarios:
        name = s["name"]
        name = name.replace("→", "->")
        print("\n" + "=" * 90)
        # Ensure console encoding compatibility.
        print(name.encode("ascii", "ignore").decode("ascii", "ignore"))
        print("-" * 90)
        if "turn1" in s:
            history = [{"role": "user", "content": s["turn1"]}]
            sync_persistent_mission_state(s["turn1"], data_used=data_used)
            out1 = _call_intelligence(
                query="Recommend an aircraft class for enterprise Europe shuttle.",
                history=history,
                data_used=data_used,
            )
            print("\n[Turn 1 output]\n" + out1)

            history.append({"role": "assistant", "content": out1})
            history.append({"role": "user", "content": s["turn2"]})
            out2 = _call_intelligence(
                query=s["turn2"],
                history=history,
                data_used=data_used,
            )
            print("\n[Turn 2 output]\n" + out2)
        else:
            query = s["query"]
            out = _call_intelligence(query=query, history=s.get("history") or [], data_used=data_used)
            print("\n[Output]\n" + out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

