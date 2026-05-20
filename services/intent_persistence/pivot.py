"""Detect when the user starts a new shopping/visual thread (do not inherit prior aircraft)."""

from __future__ import annotations

import re


def is_visual_budget_shopping_pivot(query: str) -> bool:
    """
    e.g. "Show me modern cabin under $10M" after an unrelated mission thread.

    Requires visual/cabin language + budget; must not name a specific model to compare.
    """
    q = (query or "").strip()
    if not q:
        return False
    ql = q.lower()
    has_visual = bool(
        re.search(
            r"\b(show\s+me|see\s+the|photos?|pictures?|gallery|cabin|interior|cockpit)\b",
            ql,
        )
    )
    has_budget = bool(
        re.search(
            r"(?:under|around|about|<=?)\s*\$?\s*\d+(?:\.\d+)?\s*(?:m|mm|million|mil)\b",
            ql,
        )
        or re.search(r"\bbudget\b", ql)
    )
    if not (has_visual and has_budget):
        return False
    if re.search(r"\b(compare|versus|vs\.?)\b", ql):
        return False
    try:
        from rag.consultant_query_expand import _detect_models

        if _detect_models(q):
            return False
    except Exception:
        pass
    return True


def _parse_budget_millions(query: str) -> Optional[float]:
    m = re.search(
        r"(?:under|around|about|<=?)\s*\$?\s*(\d+(?:\.\d+)?)\s*(m|mm|million|mil)\b",
        (query or ""),
        re.I,
    )
    if not m:
        return None
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return None


def modern_cabin_under_10m_models() -> list[str]:
    """Best modern-feeling cabins under ~$10M (super-midsize band)."""
    return ["Challenger 350", "Praetor 500", "Citation Latitude", "Legacy 450"]


def less_corporate_interior_models() -> list[str]:
    """Softer / lifestyle interiors after 'less corporate' refinement."""
    return ["Praetor 600", "Falcon 8X", "Global 6500"]


def bigger_modern_cabin_models() -> list[str]:
    """Upscale cabin volume while keeping modern aesthetic."""
    return ["Global 6000", "Falcon 8X", "G500"]


def shopping_gallery_models(query: str) -> list[str]:
    """Default type-representative models for budget + cabin browse (no ULR bleed)."""
    budget_m = _parse_budget_millions(query)
    if budget_m is not None and budget_m <= 12:
        return modern_cabin_under_10m_models()
    if budget_m is not None and budget_m <= 6:
        return ["Citation Latitude", "Phenom 300", "Learjet 75"]
    if budget_m is not None and budget_m <= 22:
        return ["Challenger 650", "Falcon 2000LXS", "G280"]
    return modern_cabin_under_10m_models()


def refinement_gallery_models(refinement_type: str, query: str) -> list[str]:
    """Gallery anchors for short aesthetic / size refinements."""
    ref = (refinement_type or "").strip().lower()
    ql = (query or "").lower()
    if ref == "style_shift" or re.search(r"\bless\s+corporate\b", ql):
        return less_corporate_interior_models()
    if ref == "size_upgrade" or re.match(r"^\s*bigger\s*[\.\!]?\s*$", (query or "").strip(), re.I):
        return bigger_modern_cabin_models()
    return []


def shopping_search_query(query: str) -> str:
    """Retrieval/gallery anchor text for a budget cabin browse pivot."""
    models = shopping_gallery_models(query)
    budget_m = _parse_budget_millions(query)
    band = f"under ${int(budget_m)}M" if budget_m else "mid budget"
    primary = models[0] if models else "Challenger 350"
    return (
        f"{primary} modern cabin interior {band} super-midsize "
        f"alternatives {', '.join(models[1:3])}"
    ).strip()


def refinement_search_query(refinement_type: str, query: str) -> str:
    """SearchAPI / retrieval text for refinement turns."""
    models = refinement_gallery_models(refinement_type, query)
    if not models:
        return ""
    primary = models[0]
    ql = (query or "").lower()
    if re.search(r"\bless\s+corporate\b", ql) or (refinement_type or "").lower() == "style_shift":
        return (
            f"{primary} white interior ambient lighting minimalist cabin "
            f"alternatives {' '.join(models[1:])}"
        ).strip()
    if re.match(r"^\s*bigger\s*[\.\!]?\s*$", (query or "").strip(), re.I):
        return (
            f"{primary} large cabin modern interior lounge "
            f"alternatives {' '.join(models[1:])}"
        ).strip()
    return f"{primary} interior cabin"
