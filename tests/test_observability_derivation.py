"""Observability must not import or run recommendation/inference logic."""

from __future__ import annotations

import ast
from pathlib import Path


def test_pipeline_observability_has_no_inference_imports():
    path = Path(__file__).resolve().parents[1] / "tests" / "e2e" / "pipeline_observability.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = (
        "recommendation_selector",
        "infer_listing",
        "select_executive",
        "category_resolver",
        "market_intelligence_engine",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            for bad in forbidden:
                assert bad not in mod, f"observability must not import {mod}"
        if isinstance(node, ast.Import):
            for alias in node.names:
                for bad in forbidden:
                    assert bad not in alias.name, f"observability must not import {alias.name}"


def test_pipeline_observability_only_reads_data_used():
    path = Path(__file__).resolve().parents[1] / "tests" / "e2e" / "pipeline_observability.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    banned_calls = {"broker_certify", "run_retrieval", "select_executive_recommendation", "infer_listing_verdict"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or (getattr(func, "attr", None) if isinstance(func, ast.Attribute) else None)
            if name in banned_calls:
                raise AssertionError(f"observability must not call {name}")
