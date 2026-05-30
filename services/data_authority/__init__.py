"""
Data authority — strict source-of-truth hierarchy for the acquisition consultant.

PostgreSQL (aviacost + aircraft master) → mission engine → LLM explanation.
RAG/Tavily gated to documents and dynamic market data only.
"""

from services.data_authority.aircraft_spec_repository import (
    INSUFFICIENT_VERIFIED_AIRCRAFT_DATA,
    VerifiedAircraftSpec,
    get_verified_spec,
    get_verified_spec_profile,
    list_verified_model_keys,
    require_verified_specs,
)
from services.data_authority.source_hierarchy import (
    DataSourceLevel,
    WorkflowType,
    allowed_sources_for_workflow,
)
from services.data_authority.workflow_gates import (
    attach_data_authority_metadata,
    enforce_workflow_isolation,
    resolve_workflow_type,
)

__all__ = [
    "INSUFFICIENT_VERIFIED_AIRCRAFT_DATA",
    "VerifiedAircraftSpec",
    "DataSourceLevel",
    "WorkflowType",
    "allowed_sources_for_workflow",
    "attach_data_authority_metadata",
    "enforce_workflow_isolation",
    "get_verified_spec",
    "get_verified_spec_profile",
    "list_verified_model_keys",
    "require_verified_specs",
    "resolve_workflow_type",
]
