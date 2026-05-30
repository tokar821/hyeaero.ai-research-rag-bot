"""
Path accuracy report — pass rates by golden case category.

Evaluation-only; does not modify production behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from evaluation.golden_dataset import GoldenTestCase, iter_cases_by_category
from evaluation.unified_evaluator import EvaluationResult


@dataclass(frozen=True)
class CategoryAccuracy:
    category: str
    total: int
    route_pass: int
    model_pass: int
    behavior_pass: int
    full_pass: int

    @property
    def route_pass_rate(self) -> float:
        return self.route_pass / self.total if self.total else 0.0

    @property
    def model_pass_rate(self) -> float:
        return self.model_pass / self.total if self.total else 0.0

    @property
    def behavior_pass_rate(self) -> float:
        return self.behavior_pass / self.total if self.total else 0.0

    @property
    def pass_rate(self) -> float:
        return self.full_pass / self.total if self.total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "total": self.total,
            "pass_rate": round(self.pass_rate, 4),
            "route_pass_rate": round(self.route_pass_rate, 4),
            "model_pass_rate": round(self.model_pass_rate, 4),
            "behavior_pass_rate": round(self.behavior_pass_rate, 4),
            "route_pass": self.route_pass,
            "model_pass": self.model_pass,
            "behavior_pass": self.behavior_pass,
            "full_pass": self.full_pass,
        }


def build_path_accuracy_report(
    cases: List[GoldenTestCase],
    results: List[EvaluationResult],
) -> Dict[str, Any]:
    by_id = {r.case_id: r for r in results}
    categories: Dict[str, CategoryAccuracy] = {}

    for cat, cat_cases in iter_cases_by_category(cases):
        rows = [by_id[c.id] for c in cat_cases if c.id in by_id]
        total = len(rows)
        categories[cat] = CategoryAccuracy(
            category=cat,
            total=total,
            route_pass=sum(1 for r in rows if r.route_correct),
            model_pass=sum(1 for r in rows if r.model_correct),
            behavior_pass=sum(1 for r in rows if r.behavior_correct),
            full_pass=sum(
                1 for r in rows if r.route_correct and r.model_correct and r.behavior_correct
            ),
        )

    overall_total = len(results)
    overall_full = sum(
        1 for r in results if r.route_correct and r.model_correct and r.behavior_correct
    )

    return {
        "overall": {
            "total": overall_total,
            "pass_rate": round(overall_full / overall_total, 4) if overall_total else 0.0,
            "full_pass": overall_full,
        },
        "by_category": {cat: acc.to_dict() for cat, acc in sorted(categories.items())},
    }


def format_path_accuracy_console(report: Dict[str, Any]) -> str:
    lines = ["=== Path Accuracy Report ===", ""]
    overall = report.get("overall") or {}
    lines.append(
        f"Overall: {overall.get('full_pass', 0)}/{overall.get('total', 0)} "
        f"({overall.get('pass_rate', 0):.1%})"
    )
    lines.append("")
    for cat, row in (report.get("by_category") or {}).items():
        lines.append(
            f"{cat:14} total={row['total']:3d}  pass_rate={row['pass_rate']:.1%}  "
            f"route={row['route_pass_rate']:.1%}  model={row['model_pass_rate']:.1%}  "
            f"behavior={row['behavior_pass_rate']:.1%}"
        )
    return "\n".join(lines)


def write_path_accuracy_json(report: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


__all__ = [
    "CategoryAccuracy",
    "build_path_accuracy_report",
    "format_path_accuracy_console",
    "write_path_accuracy_json",
]
