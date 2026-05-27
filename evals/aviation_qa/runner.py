"""
Aviation QA suite runner — scenario → answer → evaluate → improvement plan.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from evals.aviation_benchmark_runner import load_aviation_mission_suite, run_single_case
from evals.aviation_benchmark_report import generate_benchmark_report
from evals.aviation_qa.evaluator_agent import evaluate_advisor_response
from evals.aviation_qa.improvement_loop import build_improvement_plan
from evals.aviation_qa.repetition_detection import BatchRepetitionTracker

_SUITE_PATH = Path(__file__).resolve().parent.parent / "aviation_mission_suite.json"


def run_aviation_qa_suite(
    *,
    suite_path: str | Path | None = None,
    case_ids: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    mode: str = "intelligence",
    get_response: Optional[Callable[[str, Optional[Dict[str, Any]]], str]] = None,
    write_json_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Full QA loop:

    1. Run each scenario through the advisory pipeline
    2. Evaluate answer (evaluator agent — structured JSON)
    3. Track cross-suite repetition
    4. Emit improvement plan with targeted fix suggestions
    """
    suite = load_aviation_mission_suite(suite_path)
    cases = list(suite.get("cases") or [])
    qa_defaults = suite.get("qa_defaults") if isinstance(suite.get("qa_defaults"), dict) else {}

    if case_ids:
        ids = set(case_ids)
        cases = [c for c in cases if c.get("id") in ids]
    if categories:
        cats = set(categories)
        cases = [c for c in cases if c.get("category") in cats]

    tracker = BatchRepetitionTracker()
    case_rows: List[Dict[str, Any]] = []
    bench_results = []

    for case in cases:
        bench = run_single_case(case, get_response=get_response, mode=mode)
        bench_results.append(bench)

        # Re-run pipeline artifacts for evaluator (run_single_case doesn't return full pipeline)
        from evals.aviation_benchmark_runner import _run_intelligence_pipeline, _format_structured_answer

        pipeline = _run_intelligence_pipeline(
            str(case.get("input") or ""),
            conversation_state=case.get("conversation_state")
            if isinstance(case.get("conversation_state"), dict)
            else None,
        )
        answer = str(pipeline.get("answer") or "")
        if mode in ("intelligence", "structured"):
            answer = _format_structured_answer(pipeline)

        qa = case.get("qa") if isinstance(case.get("qa"), dict) else {}
        forbidden = list(qa.get("forbidden_phrases") or qa_defaults.get("forbidden_phrases") or [])
        rep = tracker.observe(answer, forbidden_extra=forbidden)

        verdict = evaluate_advisor_response(
            case=case,
            answer=answer,
            turn_profile=pipeline["turn_profile"],
            merged_profile=pipeline["merged_profile"],
            mission_state=pipeline["mission_state"],
            recommendations=pipeline["recommendations"],
            mission_category=pipeline.get("mission_category"),
            suite_qa_defaults=qa_defaults,
        )
        # Blend per-answer repetition into verdict
        if rep.repetition_score > verdict.repetition_score:
            verdict.repetition_score = rep.repetition_score
        if rep.repetition_score >= 0.55 and verdict.passed:
            verdict.passed = False
            if not verdict.main_failure:
                verdict.main_failure = "Robotic / repetitive templated phrasing"

        case_rows.append(
            {
                "id": case.get("id"),
                "category": case.get("category"),
                "input": case.get("input"),
                "answer_preview": (answer or "")[:400],
                "benchmark_passed": bench.passed,
                "evaluator": verdict.to_dict(),
                "repetition": rep.to_dict(),
                "top_recommendations": (bench.metadata or {}).get("top_recommendations"),
            }
        )

    bench_report = generate_benchmark_report(
        bench_results,
        suite_meta={
            "path": str(suite_path or _SUITE_PATH),
            "mode": mode,
            "qa_enabled": True,
        },
    )

    improvement = build_improvement_plan(
        case_rows,
        suite_repetition_score=tracker.suite_repetition_score(),
    )

    passed = sum(1 for r in case_rows if (r.get("evaluator") or {}).get("passed"))
    trust_avg = sum((r.get("evaluator") or {}).get("trust_score", 0) for r in case_rows) / max(
        len(case_rows), 1
    )

    report = {
        "schema_version": 2,
        "suite": {
            "path": str(suite_path or _SUITE_PATH),
            "case_count": len(case_rows),
        },
        "summary": {
            "total_cases": len(case_rows),
            "evaluator_passed": passed,
            "evaluator_pass_rate": round(passed / max(len(case_rows), 1), 4),
            "mean_trust_score": round(trust_avg, 4),
            "benchmark_pass_rate": bench_report.get("summary", {}).get("pass_rate"),
        },
        "repetition": tracker.to_dict(),
        "benchmark": bench_report,
        "cases": case_rows,
        "improvement_plan": improvement,
    }

    if write_json_path:
        out = Path(write_json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report
