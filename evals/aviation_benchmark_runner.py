"""
Run aviation mission benchmark suite against consultant intelligence pipeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from evals.aviation_benchmark_report import generate_benchmark_report
from evals.aviation_benchmark_scoring import BenchmarkCaseResult, score_benchmark_case

_SUITE_PATH = Path(__file__).resolve().parent / "aviation_mission_suite.json"

ResponseFn = Callable[[str, Optional[Dict[str, Any]]], str]


def load_aviation_mission_suite(path: str | Path | None = None) -> Dict[str, Any]:
    p = Path(path) if path else _SUITE_PATH
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _run_intelligence_pipeline(
    query: str,
    *,
    conversation_state: Optional[Dict[str, Any]] = None,
    draft_answer: str = "",
) -> Dict[str, Any]:
    from services.consultant.intelligence_engine import run_consultant_intelligence_layer
    from services.mission.memory_bridge import extract_mission_with_memory

    turn_profile, merged_profile, next_memory = extract_mission_with_memory(
        query,
        conversation_state=conversation_state,
    )
    result = run_consultant_intelligence_layer(
        answer=draft_answer or f"Mission advisory for: {query[:120]}",
        query=query,
        history=None,
        data_used={},
        conversation_state=conversation_state,
    )
    patch = result.data_used_patch or {}
    return {
        "query": query,
        "turn_profile": turn_profile.to_dict(),
        "merged_profile": patch.get("consultant_mission_profile") or merged_profile.to_dict(),
        "mission_state": patch.get("consultant_mission_state") or {},
        "recommendations": patch.get("consultant_recommendations") or [],
        "mission_category": patch.get("consultant_mission_category"),
        "answer": result.answer,
        "next_memory": patch.get("consultant_mission_memory"),
    }


def _format_structured_answer(pipeline: Dict[str, Any]) -> str:
    """Build answer text from formatter output when no LLM is wired."""
    from services.consultant.mission_state import MissionState
    from services.consultant.response_formatter import format_consultant_response
    from services.consultant.recommendation_engine import AircraftRecommendation

    ms = MissionState.from_dict(pipeline.get("mission_state"))
    recs_raw = pipeline.get("recommendations") or []
    recs: List[AircraftRecommendation] = []
    for r in recs_raw:
        if not isinstance(r, dict):
            continue
        expl = r.get("explanation") or {}
        from services.consultant.recommendation_engine import RecommendationExplanation

        from services.recommendation.fit_policy import normalize_fit_label, score_to_fit_label

        fit = normalize_fit_label(
            str(r.get("fit") or "")
            or score_to_fit_label(float(r.get("total_score") or 0), avoid=bool(r.get("avoid")))
        )
        rec = AircraftRecommendation(
            model=str(r.get("model") or ""),
            category=str(r.get("category") or ""),
            total_score=float(r.get("total_score") or 0),
            confidence=float(r.get("confidence") or 0),
            rank=int(r.get("rank") or 0),
            avoid=bool(r.get("avoid")),
            fit=fit,
            explanation=RecommendationExplanation(
                summary=str(expl.get("summary") or ""),
                strengths=list(expl.get("strengths") or []),
                penalties=list(expl.get("penalties") or []),
                operational_caveats=list(expl.get("operational_caveats") or []),
                why_it_fits=list(expl.get("why_it_fits") or []),
                operational_compromises=list(expl.get("operational_compromises") or []),
                why_alternatives_lost=list(expl.get("why_alternatives_lost") or []),
            ),
        )
        recs.append(rec)
    query = str(pipeline.get("query") or "")
    return format_consultant_response(
        mission=ms,
        recommendations=recs,
        route_assessments=[],
        draft_answer="",
        query=query,
        turn_seed=query,
    )


def run_single_case(
    case: Dict[str, Any],
    *,
    get_response: Optional[ResponseFn] = None,
    mode: str = "intelligence",
) -> BenchmarkCaseResult:
    query = str(case.get("input") or "")
    conv = case.get("conversation_state") if isinstance(case.get("conversation_state"), dict) else None

    pipeline = _run_intelligence_pipeline(query, conversation_state=conv)

    if mode == "full" and get_response:
        answer = get_response(query, conv)
        pipeline["answer"] = answer
    elif mode in ("intelligence", "structured"):
        pipeline["answer"] = _format_structured_answer(pipeline)
    else:
        pipeline["answer"] = pipeline.get("answer") or ""

    return score_benchmark_case(
        case=case,
        turn_profile=pipeline["turn_profile"],
        merged_profile=pipeline["merged_profile"],
        mission_state=pipeline["mission_state"],
        recommendations=pipeline["recommendations"],
        mission_category=pipeline.get("mission_category"),
        answer=str(pipeline.get("answer") or ""),
    )


def run_aviation_mission_benchmark(
    *,
    suite_path: str | Path | None = None,
    case_ids: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    mode: str = "intelligence",
    get_response: Optional[ResponseFn] = None,
    write_json_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Execute benchmark suite and return full report.

    Modes:
    - ``intelligence`` — mission extract + ranker + structured formatter (default, no API)
    - ``structured`` — same as intelligence
    - ``full`` — requires ``get_response`` callable for live LLM answers
    """
    suite = load_aviation_mission_suite(suite_path)
    cases = list(suite.get("cases") or [])

    if case_ids:
        ids = set(case_ids)
        cases = [c for c in cases if c.get("id") in ids]
    if categories:
        cats = set(categories)
        cases = [c for c in cases if c.get("category") in cats]

    results: List[BenchmarkCaseResult] = []
    for case in cases:
        results.append(
            run_single_case(case, get_response=get_response, mode=mode)
        )

    report = generate_benchmark_report(
        results,
        suite_meta={
            "path": str(suite_path or _SUITE_PATH),
            "schema_version": suite.get("schema_version"),
            "categories": suite.get("categories"),
            "mode": mode,
        },
    )
    report["cases"] = [
        {
            "id": r.case_id,
            "category": r.category,
            "passed": r.passed,
            "critical": r.critical,
            "scores": r.scores,
            "automated_failures": r.automated_failures,
            "issues": r.issues,
            "metadata": r.metadata,
        }
        for r in results
    ]

    if write_json_path:
        out = Path(write_json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report
