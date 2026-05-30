"""
Source-of-truth hierarchy — which layers may supply data for each workflow.
"""

from __future__ import annotations

from enum import Enum
from typing import FrozenSet


class DataSourceLevel(str, Enum):
    POSTGRES_AIRCRAFT = "postgres_aircraft"
    MISSION_ENGINE = "mission_engine"
    LLM_EXPLANATION = "llm_explanation"
    RAG_DOCUMENTS = "rag_documents"
    TAVILY_MARKET = "tavily_market"
    SEARCHAPI_IMAGES = "searchapi_images"
    CURATED_FALLBACK = "curated_fallback"  # dev-only when strict mode off


class WorkflowType(str, Enum):
    NAMED_CAPABILITY = "named_capability"
    RECOMMENDATION = "recommendation"
    EXPLICIT_COMPARISON = "explicit_comparison"
    STRATEGIC_ANALYSIS = "strategic_analysis"
    OWNERSHIP_ECONOMICS = "ownership_economics"
    TAIL_LOOKUP = "tail_lookup"
    IMAGE_VERIFICATION = "image_verification"
    MARKET_LISTINGS = "market_listings"


_WORKFLOW_ALLOWED: dict[WorkflowType, FrozenSet[DataSourceLevel]] = {
    WorkflowType.NAMED_CAPABILITY: frozenset(
        {
            DataSourceLevel.POSTGRES_AIRCRAFT,
            DataSourceLevel.MISSION_ENGINE,
            DataSourceLevel.LLM_EXPLANATION,
        }
    ),
    WorkflowType.RECOMMENDATION: frozenset(
        {
            DataSourceLevel.POSTGRES_AIRCRAFT,
            DataSourceLevel.MISSION_ENGINE,
            DataSourceLevel.LLM_EXPLANATION,
            DataSourceLevel.RAG_DOCUMENTS,
        }
    ),
    WorkflowType.EXPLICIT_COMPARISON: frozenset(
        {
            DataSourceLevel.POSTGRES_AIRCRAFT,
            DataSourceLevel.MISSION_ENGINE,
            DataSourceLevel.LLM_EXPLANATION,
        }
    ),
    WorkflowType.STRATEGIC_ANALYSIS: frozenset(
        {
            DataSourceLevel.POSTGRES_AIRCRAFT,
            DataSourceLevel.MISSION_ENGINE,
            DataSourceLevel.LLM_EXPLANATION,
        }
    ),
    WorkflowType.OWNERSHIP_ECONOMICS: frozenset(
        {
            DataSourceLevel.POSTGRES_AIRCRAFT,
            DataSourceLevel.MISSION_ENGINE,
            DataSourceLevel.LLM_EXPLANATION,
            DataSourceLevel.TAVILY_MARKET,
        }
    ),
    WorkflowType.TAIL_LOOKUP: frozenset(
        {
            DataSourceLevel.POSTGRES_AIRCRAFT,
            DataSourceLevel.TAVILY_MARKET,
            DataSourceLevel.LLM_EXPLANATION,
            DataSourceLevel.SEARCHAPI_IMAGES,
        }
    ),
    WorkflowType.IMAGE_VERIFICATION: frozenset(
        {
            DataSourceLevel.SEARCHAPI_IMAGES,
            DataSourceLevel.LLM_EXPLANATION,
        }
    ),
    WorkflowType.MARKET_LISTINGS: frozenset(
        {
            DataSourceLevel.POSTGRES_AIRCRAFT,
            DataSourceLevel.TAVILY_MARKET,
            DataSourceLevel.RAG_DOCUMENTS,
            DataSourceLevel.LLM_EXPLANATION,
        }
    ),
}


def allowed_sources_for_workflow(workflow: WorkflowType) -> FrozenSet[DataSourceLevel]:
    return _WORKFLOW_ALLOWED.get(workflow, frozenset({DataSourceLevel.LLM_EXPLANATION}))


def postgres_specs_required(workflow: WorkflowType) -> bool:
    return workflow in (
        WorkflowType.NAMED_CAPABILITY,
        WorkflowType.EXPLICIT_COMPARISON,
        WorkflowType.RECOMMENDATION,
    )


def tavily_allowed(workflow: WorkflowType) -> bool:
    return DataSourceLevel.TAVILY_MARKET in allowed_sources_for_workflow(workflow)


def rag_specs_allowed(workflow: WorkflowType) -> bool:
    """RAG must not supply aircraft performance specs — only documents/listings."""
    return False


__all__ = [
    "DataSourceLevel",
    "WorkflowType",
    "allowed_sources_for_workflow",
    "postgres_specs_required",
    "tavily_allowed",
    "rag_specs_allowed",
]
