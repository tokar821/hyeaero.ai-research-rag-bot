"""
Batch qualitative evaluation runner for the aviation consultant.

This is evaluation-only. It does NOT modify production logic.

Usage (stub mode):
  python scripts/simulate_consultant_evaluation.py

Usage (integration):
  Provide a callable that returns the assistant response text for a query.
  See `get_response_stub` for the expected signature.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(_HERE)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from eval.consultant_eval_queries import EvalCategory, build_default_eval_cases, group_cases_by_category
from eval.consultant_eval_scoring import evaluate_response
from eval.consultant_eval_gold import gold_index


ResponseFn = Callable[[str], str]


def get_response_stub(query: str) -> str:
    """
    Default stub response generator.

    Replace this with a real call to your consultant generation endpoint/pipeline when running
    integrated evaluations. Keeping it as a stub prevents accidental live API usage.
    """
    ql = (query or "").lower()
    if "falcon 9000" in ql:
        return (
            "There isn’t a Dassault “Falcon 9000” production model. Closest real families are Falcon 900, Falcon 2000, "
            "and Falcon 7X/8X/6X/10X.\n\nConsultant Insight: When a model name is off, photos and listings get mixed fast—lock the exact variant first."
        )
    if "show me n000zzz" in ql or "photos of n000zzz" in ql:
        return "No verified images found for this aircraft."
    if " vs " in ql or "versus" in ql:
        return (
            "Verdict: It depends on mission and cabin priorities.\n\n"
            "Key differences:\n- Range\n- Cabin\n- Cost\n\nWhen to choose each:\n- If you value cabin: pick the bigger cabin.\n\nConsultant Insight: Buyers often overestimate range needs."
        )
    return "I’d need your typical passengers and longest leg to give a precise recommendation. Assuming 6–8 pax, here are 3 options and why they fit…\n\nConsultant Insight: First-time buyers overbuy range."


def _parse_score_total(s: str) -> int:
    try:
        return int(str(s).split("/", 1)[0])
    except Exception:
        return 0


def simulate_consultant_evaluation(
    *,
    get_response: ResponseFn,
    limit_per_category: int | None = None,
    write_json_path: str | None = None,
) -> Dict[str, Any]:
    cases = build_default_eval_cases()
    if limit_per_category:
        grouped = group_cases_by_category(cases)
        trimmed: List[Any] = []
        for cat, items in grouped.items():
            trimmed.extend(items[: max(1, int(limit_per_category))])
        cases = trimmed

    gold = gold_index()

    per_cat_scores: Dict[str, List[int]] = defaultdict(list)
    all_rows: List[Dict[str, Any]] = []
    critical_failures = 0
    hallucination_failures = 0
    tail_cases = 0
    tail_critical = 0
    advisory_scores: List[int] = []

    for c in cases:
        resp = get_response(c.query)
        ev = evaluate_response(c.query, resp)
        st = _parse_score_total(ev["score_total"])
        per_cat_scores[c.category.value].append(st)

        if ev.get("severity") == "critical":
            critical_failures += 1
        if any("hallucinated" in x for x in (ev.get("critical_failures") or [])):
            hallucination_failures += 1
        if c.category in (EvalCategory.TAIL_NUMBER_PRECISION,):
            tail_cases += 1
            if ev.get("severity") == "critical" and any("tail_mismatch" in x for x in (ev.get("critical_failures") or [])):
                tail_critical += 1
        if c.category in (EvalCategory.ADVISORY_QUALITY, EvalCategory.CLIENT_DECISION_SCENARIOS):
            advisory_scores.append(st)

        g = gold.get(c.query)
        row = {
            "id": c.id,
            "category": c.category.value,
            "query": c.query,
            "notes": c.notes,
            "response": resp,
            "eval": ev,
            "gold_tags": (g.tags if g else None),
            "gold_present": bool(g),
        }
        all_rows.append(row)

    # Category breakdown
    cat_avg = {
        cat: (sum(vals) / max(1, len(vals)))
        for cat, vals in sorted(per_cat_scores.items(), key=lambda kv: kv[0])
    }

    # Worst 10 by score, tie-break critical
    all_rows_sorted = sorted(
        all_rows,
        key=lambda r: (
            _parse_score_total(r["eval"]["score_total"]),
            0 if r["eval"].get("severity") == "critical" else 1,
        ),
    )
    worst_10 = all_rows_sorted[:10]

    total = len(all_rows)
    report = {
        "overall": {
            "n_cases": total,
            "overall_avg_score": (sum(_parse_score_total(r["eval"]["score_total"]) for r in all_rows) / max(1, total)),
            "critical_failure_rate": (critical_failures / max(1, total)),
            "hallucination_failure_rate": (hallucination_failures / max(1, total)),
            "tail_accuracy_rate": (1.0 - (tail_critical / max(1, tail_cases))) if tail_cases else None,
            "advisory_avg_score": (sum(advisory_scores) / max(1, len(advisory_scores))) if advisory_scores else None,
        },
        "category_breakdown": cat_avg,
        "worst_10": [
            {
                "id": r["id"],
                "category": r["category"],
                "query": r["query"],
                "score_total": r["eval"]["score_total"],
                "severity": r["eval"]["severity"],
                "critical_failures": r["eval"].get("critical_failures"),
                "issues": r["eval"].get("issues"),
            }
            for r in worst_10
        ],
        "top_improvement_areas": _derive_improvement_areas(all_rows),
    }

    if write_json_path:
        payload = {"report": report, "cases": all_rows}
        with open(write_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    return report


def _derive_improvement_areas(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = defaultdict(int)
    sev_counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        ev = r.get("eval") or {}
        for issue in (ev.get("issues") or []):
            counts[str(issue)] += 1
        if ev.get("severity") == "critical":
            sev_counts["critical"] += 1
        elif ev.get("severity") == "medium":
            sev_counts["medium"] += 1
    worst = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:12]
    return [{"issue": k, "count": v} for k, v in worst] + [
        {"issue": "severity_critical", "count": int(sev_counts.get("critical", 0))},
        {"issue": "severity_medium", "count": int(sev_counts.get("medium", 0))},
    ]


def main() -> None:
    # Default: stub mode. Write a JSON artifact if desired.
    out_path = os.getenv("CONSULTANT_EVAL_OUT") or ""
    report = simulate_consultant_evaluation(
        get_response=get_response_stub,
        limit_per_category=None,
        write_json_path=(out_path.strip() or None),
    )
    print("=== FINAL REPORT ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

