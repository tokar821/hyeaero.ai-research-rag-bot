#!/usr/bin/env python3
"""
Phase 54 — audit e2e benchmark tests for semantic assertions.

A certification test must assert output correctness (not only execution path).
Exits 1 when violations found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
E2E = BACKEND / "tests" / "e2e"

# Tests allowed path-only (documented non-certification).
PATH_ONLY_ALLOWLIST = {
    "test_execution_path_parity.py",
    "test_broker_certification_suite.py",
    "test_broker_certification_v2.py",
    "test_consultant_retrieval_intent_lock_e2e.py",
    "test_conversation_guard_leaks.py",
}

# Must contain at least one semantic assert pattern.
SEMANTIC_ASSERT_SNIPPETS = (
    "assert passed",
    "assert correct",
    "assert ok",
    "assert bias",
    "assert not rec.authority",
    "assert not rec.drift",
    "assert rec.semantic",
    "assert metrics[",
    "assert primary_acc",
    "_compatible(",
    "assert_listing_observability",
    "assert_observability_contract",
    "mission_primary_present",
    "mission_semantic_ok",
)


def _audit_file(path: Path) -> list[str]:
    issues: list[str] = []
    if path.name in PATH_ONLY_ALLOWLIST:
        return issues
    if not path.name.startswith("test_") and path.name.endswith("_suite.py"):
        pass
    elif not (path.name.endswith("_suite.py") or path.name.endswith("_benchmark.py")):
        return issues

    text = path.read_text(encoding="utf-8")
    if "def test_" not in text:
        return issues

    has_semantic = any(s in text for s in SEMANTIC_ASSERT_SNIPPETS)
    only_path = "assert path in" in text and not has_semantic

    if only_path or not has_semantic:
        issues.append(f"{path.relative_to(BACKEND)}: no semantic correctness assert")

    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        issues.append(f"{path}: syntax error {e}")
        return issues

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            body = ast.unparse(node) if hasattr(ast, "unparse") else ""
            if "assert path in" in body and not any(s in body for s in SEMANTIC_ASSERT_SNIPPETS):
                issues.append(f"{path.name}::{node.name}: path-only test function")

    return issues


def main() -> int:
    violations: list[str] = []
    for py in sorted(E2E.rglob("*.py")):
        if py.name.startswith("__"):
            continue
        violations.extend(_audit_file(py))

    if violations:
        print("Benchmark assertion audit FAILED:")
        for v in violations:
            print(f"  - {v}")
        return 1

    print("Benchmark assertion audit OK — certification suites include semantic asserts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
