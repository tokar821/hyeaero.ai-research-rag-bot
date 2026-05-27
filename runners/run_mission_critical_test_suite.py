"""
Mission Understanding Critical Test Suite — live consultant runner.

Usage:
  cd backend && python runners/run_mission_critical_test_suite.py
  cd backend && python runners/run_mission_critical_test_suite.py --ids 1,11,21,41,66
  cd backend && python runners/run_mission_critical_test_suite.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

env_path = _ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

from evals.mission_critical_scoring import score_response  # noqa: E402
from services.orchestration.pipeline_orchestrator import run_consultant_orchestration  # noqa: E402
from services.state.mission_state import sync_persistent_mission_state  # noqa: E402


def _sanitize(text: str) -> str:
    txt = (text or "").replace("→", "->").replace("–", "-").replace("—", "-")
    try:
        return txt.encode("ascii", "ignore").decode("ascii", "ignore")
    except Exception:
        return txt


def _intent_for_category(cat: str) -> str:
    if cat == "ownership":
        return "ownership_economics"
    if cat in ("broker_realism", "failure_handling", "synthesis"):
        return "operational_tradeoff_analysis"
    return "mission_feasibility"


def _call(
    *,
    query: str,
    history: List[Dict[str, str]],
    data_used: Dict[str, Any],
    category: str,
) -> tuple[str, Dict[str, Any]]:
    data_used.setdefault("consultant_response_mode", "mission_advisory")
    orch = run_consultant_orchestration(
        query,
        llm_draft="",
        history=history,
        data_used=data_used,
        query_intent=_intent_for_category(category),
        max_results=3,
    )
    du = dict(data_used)
    if isinstance(orch.data_used_patch, dict):
        du.update(orch.data_used_patch)
    pkt = du.get("mission_understanding_packet") if isinstance(du.get("mission_understanding_packet"), dict) else {}
    return _sanitize(orch.answer or ""), pkt


SCENARIOS: List[Dict[str, Any]] = [
    # CATEGORY 1 — Hidden Operational Constraints
    {"id": 1, "cat": "hidden_constraints", "q": "We have 900 employees and fly leadership teams between New York, Dallas, and London several times a month. Sometimes the CEO continues onward to the Middle East. What kind of aircraft strategy would actually make sense?"},
    {"id": 2, "cat": "hidden_constraints", "q": "We move executives around the Caribbean constantly. Runway access matters more than cabin luxury."},
    {"id": 3, "cat": "hidden_constraints", "q": "Our board hates fuel stops. Especially westbound in winter."},
    {"id": 4, "cat": "hidden_constraints", "q": "We fly athletes and staff domestically, but ownership costs are getting out of control."},
    {"id": 5, "cat": "hidden_constraints", "q": "We need something that can still get into mountain airports without embarrassing the executives on longer flights."},
    {"id": 6, "cat": "hidden_constraints", "q": "We usually travel with 10–12 people, but occasionally only 2 or 3. I don't want to operate something inefficient."},
    {"id": 7, "cat": "hidden_constraints", "q": "We're trying to reduce total travel time, not just flight time."},
    {"id": 8, "cat": "hidden_constraints", "q": "We have manufacturing sites across Latin America and short-notice dispatch reliability matters more than luxury."},
    {"id": 9, "cat": "hidden_constraints", "q": "Our executives insist on nonstop Europe flights, but we also do a lot of Florida and Caribbean hops."},
    {"id": 10, "cat": "hidden_constraints", "q": "We care more about operating economics and airport flexibility than impressing people."},
    # CATEGORY 2 — Incomplete / Ambiguous
    {"id": 11, "cat": "incomplete", "q": "We do Europe twice monthly.", "exp": ["no_collapse", "class_band_or_followup"]},
    {"id": 12, "cat": "incomplete", "q": "We need something for heavy international use.", "exp": ["no_collapse", "class_band_or_followup"]},
    {"id": 13, "cat": "incomplete", "q": "We're mostly domestic but occasionally Asia.", "exp": ["no_collapse"]},
    {"id": 14, "cat": "incomplete", "q": "We fly teams around South America.", "exp": ["no_collapse"]},
    {"id": 15, "cat": "incomplete", "q": "We want reliable winter westbound capability.", "exp": ["no_collapse"]},
    {"id": 16, "cat": "incomplete", "q": "We need better dispatch reliability than charter.", "exp": ["no_collapse"]},
    {"id": 17, "cat": "incomplete", "q": "We've outgrown our current setup.", "exp": ["no_collapse", "ownership_economics"]},
    {"id": 18, "cat": "incomplete", "q": "Our mission profile changed after opening a London office.", "exp": ["no_collapse"]},
    {"id": 19, "cat": "incomplete", "q": "We spend too much time repositioning.", "exp": ["no_collapse"]},
    {"id": 20, "cat": "incomplete", "q": "We're doing more long-haul than we used to.", "exp": ["no_collapse"]},
    # CATEGORY 3 — Multi-Role Synthesis
    {"id": 21, "cat": "multi_role", "q": "We need London nonstop capability, but we also spend a lot of time in Aspen.", "exp": ["multi_aircraft"]},
    {"id": 22, "cat": "multi_role", "q": "We fly Europe frequently but also need short Caribbean runway access."},
    {"id": 23, "cat": "multi_role", "q": "Our family office alternates between New York, Jackson Hole, and the south of France."},
    {"id": 24, "cat": "multi_role", "q": "We need something for both board travel and factory-site access."},
    {"id": 25, "cat": "multi_role", "q": "We want one aircraft that can do Tokyo and mountain airports.", "exp": ["resist_single_aircraft"]},
    {"id": 26, "cat": "multi_role", "q": "We shuttle executives domestically during the week and take family trips to Europe."},
    {"id": 27, "cat": "multi_role", "q": "We need a corporate shuttle but occasionally fly ultra-long-haul."},
    {"id": 28, "cat": "multi_role", "q": "We're trying to consolidate two aircraft into one."},
    {"id": 29, "cat": "multi_role", "q": "We want nonstop transatlantic performance without giving up regional flexibility."},
    {"id": 30, "cat": "multi_role", "q": "We need something efficient for short legs but credible for Europe."},
    # CATEGORY 4 — Ownership / Economics
    {"id": 31, "cat": "ownership", "q": "We currently charter around 200–250 hours annually. At what point does ownership actually become rational?", "exp": ["ownership_economics"]},
    {"id": 32, "cat": "ownership", "q": "We're debating fractional versus full ownership.", "exp": ["ownership_economics"]},
    {"id": 33, "cat": "ownership", "q": "We don't want capital tied up in a depreciating asset.", "exp": ["ownership_economics"]},
    {"id": 34, "cat": "ownership", "q": "We need guaranteed availability more than prestige.", "exp": ["ownership_economics"]},
    {"id": 35, "cat": "ownership", "q": "We already operate a turboprop but are considering adding a jet."},
    {"id": 36, "cat": "ownership", "q": "We've been using NetJets but are questioning whether it still makes sense.", "exp": ["ownership_economics"]},
    {"id": 37, "cat": "ownership", "q": "We care a lot about exit liquidity."},
    {"id": 38, "cat": "ownership", "q": "Our travel patterns fluctuate heavily quarter to quarter."},
    {"id": 39, "cat": "ownership", "q": "We're trying to justify ownership internally.", "exp": ["ownership_economics"]},
    {"id": 40, "cat": "ownership", "q": "We don't fly enough to waste money, but charter has become operationally painful.", "exp": ["ownership_economics"]},
    # CATEGORY 5 — Conversational Continuity (multi-turn)
    {"id": 41, "cat": "continuity", "turn1": "We fly from San Francisco to Tokyo regularly.", "turn2": "What about winter westbound?", "exp": ["continuity"]},
    {"id": 42, "cat": "continuity", "turn1": "We operate mostly around the Caribbean.", "turn2": "Runways are sometimes an issue.", "exp": ["continuity"]},
    {"id": 43, "cat": "continuity", "turn1": "We travel with around 14 passengers.", "turn2": "Sometimes London.", "exp": ["continuity"]},
    {"id": 44, "cat": "continuity", "turn1": "We prioritize low operating cost.", "turn2": "But dispatch reliability has become a problem.", "exp": ["continuity"]},
    {"id": 45, "cat": "continuity", "turn1": "We're based in Miami.", "turn2": "We also ski frequently.", "exp": ["continuity"]},
    {"id": 46, "cat": "continuity", "turn1": "We charter constantly.", "turn2": "It's becoming chaotic operationally.", "exp": ["continuity", "ownership_economics"]},
    {"id": 47, "cat": "continuity", "turn1": "We care about airport access.", "turn2": "Especially in Europe.", "exp": ["continuity"]},
    {"id": 48, "cat": "continuity", "turn1": "We fly executives.", "turn2": "Sometimes families too.", "exp": ["continuity"]},
    {"id": 49, "cat": "continuity", "turn1": "We're expanding internationally.", "turn2": "Asia is becoming important.", "exp": ["continuity"]},
    {"id": 50, "cat": "continuity", "turn1": "We currently use a midsize jet.", "turn2": "We're hitting range limitations.", "exp": ["continuity"]},
    # CATEGORY 6 — Broker Realism
    {"id": 51, "cat": "broker_realism", "q": "What would actually annoy you operationally about this mission? We're a 900-employee company flying NY-Dallas-London with occasional Middle East continuation."},
    {"id": 52, "cat": "broker_realism", "q": "Where do companies usually make the wrong aircraft decision when flying Europe and Caribbean hops?"},
    {"id": 53, "cat": "broker_realism", "q": "What sounds good on paper but becomes painful in reality for mixed domestic and international executive travel?"},
    {"id": 54, "cat": "broker_realism", "q": "What would you avoid buying for a mission that mixes London nonstop and Aspen mountain access?"},
    {"id": 55, "cat": "broker_realism", "q": "What compromises am I underestimating if I want one jet for Tokyo and mountain airports?"},
    {"id": 56, "cat": "broker_realism", "q": "What would make you recommend against ownership for a company chartering 200 hours a year?"},
    {"id": 57, "cat": "broker_realism", "q": "Where would dispatch reliability become ugly on winter westbound transpac missions?"},
    {"id": 58, "cat": "broker_realism", "q": "What mission detail matters more than most buyers realize when buying for board travel?"},
    {"id": 59, "cat": "broker_realism", "q": "What kind of company usually regrets buying too much airplane?"},
    {"id": 60, "cat": "broker_realism", "q": "If this were your company flying executives domestically with rising ownership costs, what operational mistake would you try hardest to avoid?"},
    # CATEGORY 7 — Failure & Confidence
    {"id": 61, "cat": "failure_handling", "q": "Can a Citation CJ3+ fly nonstop from New York to Tokyo?", "exp": ["honest_refusal_guidance"]},
    {"id": 62, "cat": "failure_handling", "q": "We want one aircraft for London, Aspen, Caribbean islands, and Tokyo.", "exp": ["multi_aircraft", "honest_refusal_guidance"]},
    {"id": 63, "cat": "failure_handling", "q": "We need nonstop Europe capability from Aspen in winter.", "exp": ["honest_refusal_guidance"]},
    {"id": 64, "cat": "failure_handling", "q": "What aircraft can realistically do westbound Tokyo from San Diego year-round with 12 passengers?", "exp": ["honest_refusal_guidance"]},
    {"id": 65, "cat": "failure_handling", "q": "We need ultra-long-range capability but also want turboprop economics.", "exp": ["honest_refusal_guidance"]},
    # CATEGORY 8 — Real Broker Synthesis
    {"id": 66, "cat": "synthesis", "q": "We're a private equity group. Teams move between New York, Dallas, London, and occasionally Dubai. Senior partners hate fuel stops, but we also visit smaller industrial airports domestically. We currently charter around 300 hours annually and are debating ownership.", "exp": ["broker_synthesis", "ownership_economics", "multi_aircraft"]},
    {"id": 67, "cat": "synthesis", "q": "Our family office flies between Miami, Aspen, New York, and Europe. We care about comfort, but airport flexibility matters too. We've considered buying one large jet to simplify things.", "exp": ["broker_synthesis", "multi_aircraft"]},
    {"id": 68, "cat": "synthesis", "q": "We run a tech company with aggressive international growth. Asia travel increased dramatically this year, but we still do constant West Coast shuttles.", "exp": ["broker_synthesis"]},
    {"id": 69, "cat": "synthesis", "q": "We need something operationally credible for executives, but we're trying hard not to overspend on capability we rarely use.", "exp": ["broker_synthesis", "ownership_economics"]},
    {"id": 70, "cat": "synthesis", "q": "We're reaching the point where charter friction is hurting operations, but I'm not convinced full ownership is financially rational yet.", "exp": ["broker_synthesis", "ownership_economics"]},
]


def run_scenario(s: Dict[str, Any], data_used: Dict[str, Any]) -> Dict[str, Any]:
    sid = int(s["id"])
    cat = str(s["cat"])
    exp = list(s.get("exp") or [])

    if "turn1" in s:
        t1 = str(s["turn1"])
        t2 = str(s["turn2"])
        history: List[Dict[str, str]] = [{"role": "user", "content": t1}]
        sync_persistent_mission_state(t1, data_used=data_used)
        out1, pkt1 = _call(query=t1, history=[], data_used=data_used, category=cat)
        history.append({"role": "assistant", "content": out1})
        history.append({"role": "user", "content": t2})
        sync_persistent_mission_state(t2, data_used=data_used)
        out2, pkt2 = _call(query=t2, history=history, data_used=data_used, category=cat)
        scored = score_response(
            scenario_id=sid,
            category=cat,
            answer=out1,
            turn2_answer=out2,
            turn1_query=t1,
            expectations=exp,
            packet=pkt2,
        )
        return {
            "id": sid,
            "category": cat,
            "turn1_query": t1,
            "turn1_answer": out1[:1200],
            "turn2_query": t2,
            "turn2_answer": out2[:1200],
            "packet": pkt2,
            "score": scored.to_dict(),
        }

    q = str(s["q"])
    sync_persistent_mission_state(q, data_used=data_used)
    out, pkt = _call(query=q, history=[], data_used=data_used, category=cat)
    scored = score_response(
        scenario_id=sid,
        category=cat,
        answer=out,
        expectations=exp,
        packet=pkt,
    )
    return {
        "id": sid,
        "category": cat,
        "query": q,
        "answer": out[:1500],
        "packet_summary": {
            "confidence": pkt.get("overall_confidence"),
            "corridor_type": pkt.get("corridor_type"),
            "travel_pattern": pkt.get("travel_pattern"),
            "recommend_aircraft": pkt.get("recommend_aircraft"),
            "synthesis": (pkt.get("operational_synthesis") or "")[:300],
            "inferred": list((pkt.get("inferred_constraints") or {}).keys())[:8],
        },
        "score": scored.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Mission Understanding Critical Test Suite")
    parser.add_argument("--ids", type=str, default="", help="Comma-separated scenario IDs")
    parser.add_argument("--limit", type=int, default=0, help="Max scenarios to run")
    parser.add_argument("--out", type=str, default="evals/mission_critical_results.json")
    args = parser.parse_args()

    os.environ.setdefault("CONSULTANT_INTELLIGENCE_LAYER", "1")
    os.environ.setdefault("CONSULTANT_ORCHESTRATION", "1")

    selected = SCENARIOS
    if args.ids.strip():
        want = {int(x.strip()) for x in args.ids.split(",") if x.strip()}
        selected = [s for s in SCENARIOS if int(s["id"]) in want]
    if args.limit > 0:
        selected = selected[: args.limit]

    results: List[Dict[str, Any]] = []
    passed = 0
    failed = 0
    t0 = time.time()

    for s in selected:
        data_used: Dict[str, Any] = {"consultant_response_mode": "mission_advisory"}
        print(f"\n--- Scenario {s['id']} ({s['cat']}) ---")
        try:
            row = run_scenario(s, data_used)
            ok = bool(row["score"]["passed"])
            passed += int(ok)
            failed += int(not ok)
            print(f"{'PASS' if ok else 'FAIL'} score={row['score']['score']:.2f} issues={row['score']['issues']}")
            if "turn2_answer" in row:
                print(row["turn2_answer"][:400])
            else:
                print(row["answer"][:400])
            results.append(row)
        except Exception as exc:
            failed += 1
            print(f"ERROR: {exc}")
            results.append({"id": s["id"], "category": s["cat"], "error": str(exc), "score": {"passed": False}})

    report = {
        "total": len(selected),
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / max(1, len(selected)), 3),
        "elapsed_s": round(time.time() - t0, 1),
        "results": results,
    }
    out_path = _ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {out_path} ({passed}/{len(selected)} passed, {report['elapsed_s']}s)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
