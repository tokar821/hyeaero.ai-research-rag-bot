"""
Golden dataset schema and loader for Phase 8 automated evaluation.

Read-only — does not modify routing or responders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden_cases"

VALID_CATEGORIES = frozenset(
    {
        "FACT",
        "MARKET",
        "CAPABILITY",
        "COMPARISON",
        "ALTERNATIVE",
        "MISSION",
        "BUY_DECISION",
    }
)

VALID_BEHAVIOR_TAGS = frozenset(
    {
        "factual_only",
        "no_mission_synthesis",
        "comparison_only",
        "alternative_only",
        "capability_yes_no",
        "market_price_band",
        "broker_style",
    }
)


@dataclass(frozen=True)
class GoldenTestCase:
    id: str
    category: str
    query: str
    expected_execution_path: Optional[str]
    expected_models: List[str] = field(default_factory=list)
    expected_behavior_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "query": self.query,
            "expected_execution_path": self.expected_execution_path,
            "expected_models": list(self.expected_models),
            "expected_behavior_tags": list(self.expected_behavior_tags),
        }


def _validate_case(case: GoldenTestCase) -> None:
    if case.category not in VALID_CATEGORIES:
        raise ValueError(f"Invalid category {case.category!r} in case {case.id}")
    for tag in case.expected_behavior_tags:
        if tag not in VALID_BEHAVIOR_TAGS:
            raise ValueError(f"Invalid behavior tag {tag!r} in case {case.id}")


def load_golden_cases(*, category: Optional[str] = None) -> List[GoldenTestCase]:
    """Load all golden cases, optionally filtered by category."""
    from evaluation.golden_cases.catalog_cases import build_golden_cases

    cases = build_golden_cases()
    for c in cases:
        _validate_case(c)
    if category:
        cat = category.upper()
        cases = [c for c in cases if c.category == cat]
    return cases


def load_golden_cases_from_json(path: Path) -> List[GoldenTestCase]:
    """Load cases from a JSON array file (optional external dataset)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: List[GoldenTestCase] = []
    for item in raw:
        cases.append(
            GoldenTestCase(
                id=str(item["id"]),
                category=str(item["category"]),
                query=str(item["query"]),
                expected_execution_path=item.get("expected_execution_path"),
                expected_models=list(item.get("expected_models") or []),
                expected_behavior_tags=list(item.get("expected_behavior_tags") or []),
            )
        )
    for c in cases:
        _validate_case(c)
    return cases


def iter_cases_by_category(cases: List[GoldenTestCase]) -> Iterator[tuple[str, List[GoldenTestCase]]]:
    """Yield (category, cases) groups in stable order."""
    seen: Dict[str, List[GoldenTestCase]] = {}
    for c in cases:
        seen.setdefault(c.category, []).append(c)
    for cat in sorted(seen.keys()):
        yield cat, seen[cat]


def dataset_summary(cases: List[GoldenTestCase]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for c in cases:
        summary[c.category] = summary.get(c.category, 0) + 1
    summary["total"] = len(cases)
    return summary


__all__ = [
    "GoldenTestCase",
    "VALID_BEHAVIOR_TAGS",
    "VALID_CATEGORIES",
    "dataset_summary",
    "iter_cases_by_category",
    "load_golden_cases",
    "load_golden_cases_from_json",
]
