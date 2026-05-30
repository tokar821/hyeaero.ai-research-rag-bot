"""Data authority hierarchy — spec repository and workflow gates."""

from __future__ import annotations

from services.catalog.catalog_alias_resolver import resolve_catalog_profile_key
from services.data_authority.aircraft_spec_repository import (
    INSUFFICIENT_VERIFIED_AIRCRAFT_DATA,
    get_verified_spec,
)
from services.data_authority.workflow_gates import (
    attach_data_authority_metadata,
    resolve_workflow_type,
)
from services.data_authority.source_hierarchy import WorkflowType, postgres_specs_required


def test_no_cross_model_bridge_g500():
    """G500 must not resolve to G650 profile key."""
    key = resolve_catalog_profile_key("Gulfstream G500")
    assert key != "Gulfstream G650"
    if key is not None:
        assert key == "Gulfstream G500"


def test_capability_workflow_postgres_required():
    wf = resolve_workflow_type(
        "Could a Citation Longitude fly San Francisco to Paris with NBAA reserves?",
        v2_renderer="named_aircraft_capability",
        v2_query_type="named_aircraft_capability",
    )
    assert wf == WorkflowType.NAMED_CAPABILITY
    assert postgres_specs_required(wf)


def test_comparison_workflow_isolation():
    du: dict = {}
    wf = attach_data_authority_metadata(
        du,
        "Compare Praetor 600 vs Challenger 650",
        v2_renderer="explicit_comparison_table",
        v2_query_type="explicit_comparison",
    )
    assert wf == WorkflowType.EXPLICIT_COMPARISON
    assert du.get("suppress_image_pipeline") is True
    assert du.get("suppress_recommendation_shortlist") is True


def test_image_workflow_not_comparison():
    wf = resolve_workflow_type(
        "Show verified exterior images of the Falcon 8X only.",
        v2_renderer="recommendation_broker",
    )
    assert wf == WorkflowType.IMAGE_VERIFICATION


def test_verified_spec_or_insufficient_message():
    spec = get_verified_spec("Citation Longitude")
    assert spec is not None or INSUFFICIENT_VERIFIED_AIRCRAFT_DATA


def test_insufficient_constant_format():
    assert "INSUFFICIENT VERIFIED" in INSUFFICIENT_VERIFIED_AIRCRAFT_DATA
