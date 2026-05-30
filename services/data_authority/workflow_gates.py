"""
Workflow isolation — prevent image/tail/recommendation contamination across modes.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from services.data_authority.source_hierarchy import (
    WorkflowType,
    postgres_specs_required,
    tavily_allowed,
)


def resolve_workflow_type(
    query: str,
    *,
    v2_renderer: str = "",
    v2_query_type: str = "",
    data_used: Optional[Dict[str, Any]] = None,
) -> WorkflowType:
    ql = (query or "").lower()
    renderer = (v2_renderer or "").lower()
    qtype = (v2_query_type or "").lower()

    try:
        from services.orchestration.query_archetype import (
            is_image_request_query,
            is_ownership_structure_query,
        )
        from services.recommendation.survival_filter_shortlist import is_survival_filter_query
    except Exception:
        is_image_request_query = lambda _q: False  # type: ignore
        is_ownership_structure_query = lambda _q: False  # type: ignore
        is_survival_filter_query = lambda _q: False  # type: ignore

    if is_image_request_query(query):
        return WorkflowType.IMAGE_VERIFICATION
    if re.search(r"\b(?:tail\s+number|n\d{1,5}[a-z]{0,2}|vp-[a-z]{3})\b", ql):
        return WorkflowType.TAIL_LOOKUP
    if is_ownership_structure_query(query) or renderer == "ownership_economics":
        return WorkflowType.OWNERSHIP_ECONOMICS
    if renderer == "named_aircraft_capability" or qtype == "named_aircraft_capability":
        return WorkflowType.NAMED_CAPABILITY
    if renderer in ("explicit_comparison_table", "strategic_comparison") or qtype == "explicit_comparison":
        return WorkflowType.EXPLICIT_COMPARISON
    if is_survival_filter_query(query) or renderer == "recommendation_broker" or qtype == "recommendation_request":
        return WorkflowType.RECOMMENDATION
    if renderer in ("strategic_analysis", "network_topology") or qtype in (
        "strategic_fleet_analysis",
        "network_structure",
    ):
        return WorkflowType.STRATEGIC_ANALYSIS
    if any(k in ql for k in ("for sale", "listing", "asking price", "market comp")):
        return WorkflowType.MARKET_LISTINGS
    return WorkflowType.STRATEGIC_ANALYSIS


def enforce_workflow_isolation(
    workflow: WorkflowType,
    *,
    query: str = "",
) -> Dict[str, Any]:
    """Flags consumed by pipeline / retrieval to suppress cross-mode behavior."""
    ql = (query or "").lower()
    return {
        "data_authority_workflow": workflow.value,
        "suppress_tavily_for_specs": postgres_specs_required(workflow),
        "suppress_rag_performance_specs": True,
        "suppress_image_pipeline": workflow
        not in (WorkflowType.IMAGE_VERIFICATION, WorkflowType.TAIL_LOOKUP),
        "suppress_recommendation_shortlist": workflow
        in (
            WorkflowType.NAMED_CAPABILITY,
            WorkflowType.EXPLICIT_COMPARISON,
            WorkflowType.IMAGE_VERIFICATION,
            WorkflowType.TAIL_LOOKUP,
        ),
        "suppress_tail_enrichment": workflow
        in (WorkflowType.EXPLICIT_COMPARISON, WorkflowType.NAMED_CAPABILITY, WorkflowType.RECOMMENDATION),
        "tavily_market_only": tavily_allowed(workflow),
        "block_aircraft_substitution": True,
    }


def attach_data_authority_metadata(
    data_used: Dict[str, Any],
    query: str,
    *,
    v2_renderer: str = "",
    v2_query_type: str = "",
) -> WorkflowType:
    wf = resolve_workflow_type(
        query,
        v2_renderer=v2_renderer,
        v2_query_type=v2_query_type,
        data_used=data_used,
    )
    gates = enforce_workflow_isolation(wf, query=query)
    data_used["data_authority_workflow"] = wf.value
    data_used.update(gates)
    return wf


__all__ = [
    "resolve_workflow_type",
    "enforce_workflow_isolation",
    "attach_data_authority_metadata",
]
