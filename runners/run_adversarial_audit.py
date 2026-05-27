"""Adversarial mission-understanding audit runner."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from services.orchestration.pipeline_orchestrator import run_consultant_orchestration  # noqa: E402


def _ascii(s: str) -> str:
    return (s or "").encode("ascii", "ignore").decode("ascii", "ignore")


SCENARIOS = [
    ("SFO_Tokyo_London_winter", "SFO to Tokyo and London nonstop westbound winter 8 passengers", "mission_feasibility"),
    ("Miami_Caribbean_SA", "Miami Caribbean South America 8 passengers runway flexibility over luxury", "mission_feasibility"),
    ("Ownership_250hr", "We currently charter around 250 hours annually. At what point does ownership actually become rational?", "ownership_economics"),
    ("PE_group", "We are a private equity group. Teams move between New York, Dallas, London, and occasionally Dubai. Senior partners hate fuel stops, but we also visit smaller industrial airports domestically. We currently charter around 300 hours annually and are debating ownership.", "operational_tradeoff_analysis"),
]


def _run_context_contamination_audit() -> dict:
    """Turn 1 Miami/Caribbean then Turn 2 NYC/Tokyo/London — must not bleed."""
    du: dict = {"consultant_response_mode": "mission_advisory"}
    history: list = []
    r1 = run_consultant_orchestration(
        "Miami Caribbean 8 passengers runway flexibility over luxury",
        conversation_state={"history": history},
        data_used=du,
        query_intent="mission_feasibility",
    )
    if isinstance(r1.data_used_patch, dict):
        du.update(r1.data_used_patch)
    history.extend([
        {"role": "user", "content": "Miami Caribbean 8 passengers runway flexibility over luxury"},
        {"role": "assistant", "content": "prior"},
    ])
    q2 = "NYC to Tokyo and London, 8 passengers, nonstop westbound winter"
    r2 = run_consultant_orchestration(
        q2,
        conversation_state={"history": history},
        data_used=du,
        query_intent="mission_feasibility",
    )
    patch = r2.data_used_patch or {}
    pkt = (
        patch.get("mission_understanding_packet")
        or du.get("mission_understanding_packet")
        or {}
    )
    cont = (
        patch.get("mission_continuity_assessment")
        or du.get("mission_continuity_assessment")
        or {}
    )
    answer = (r2.answer or "").lower()
    bands = " ".join(pkt.get("fallback_operational_band") or []).lower()
    return {
        "query": q2,
        "mission_pivot": cont.get("mission_pivot"),
        "corridor_type": pkt.get("corridor_type"),
        "bands": pkt.get("fallback_operational_band"),
        "caribbean_in_bands": "caribbean" in bands,
        "miami_in_answer": "miami" in answer and "caribbean" in answer,
        "passed": cont.get("mission_pivot") is True
        and pkt.get("corridor_type") != "caribbean_regional"
        and "caribbean" not in bands,
    }


def main() -> None:
    out = []
    for name, q, intent in SCENARIOS:
        du = {"consultant_response_mode": "mission_advisory"}
        r = run_consultant_orchestration(q, data_used=du, query_intent=intent, max_results=5)
        patch = r.data_used_patch or {}
        pkt = du.get("mission_understanding_packet") or patch.get("mission_understanding_packet") or {}
        recs = [x.get("model", "") for x in (patch.get("consultant_recommendations") or [])]
        pipeline = patch.get("deterministic_recommendation_pipeline") or {}
        out.append({
            "name": name,
            "query": q,
            "recommendations": recs,
            "feasible_models": pipeline.get("feasible_models", []),
            "eliminated_models": pipeline.get("eliminated_models", [])[:10],
            "mission_category": pipeline.get("mission_category"),
            "packet": {
                "corridor_type": pkt.get("corridor_type"),
                "bands": pkt.get("fallback_operational_band"),
                "recommend_aircraft": pkt.get("recommend_aircraft"),
                "confidence": pkt.get("overall_confidence"),
                "inferred": pkt.get("inferred_constraints"),
                "synthesis": pkt.get("operational_synthesis"),
            },
            "answer": _ascii(r.answer or ""),
        })
        print(f"=== {name} ===")
        print(f"recs={recs} category={pipeline.get('mission_category')} gate={pkt.get('recommend_aircraft')}")
        print(_ascii(r.answer or "")[:900])
        print()

    contamination = _run_context_contamination_audit()
    out.append({"name": "context_contamination", **contamination})
    print(f"=== context_contamination === pivot={contamination.get('mission_pivot')} passed={contamination.get('passed')}")
    print(f"bands={contamination.get('bands')} corridor={contamination.get('corridor_type')}")
    print()

    path = _ROOT / "evals" / "adversarial_audit_results.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
