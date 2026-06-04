"""Phase 14 — API contract versioning and stability tests."""

from __future__ import annotations

import pytest

from services.response.api_contract_versioning import (
    apply_api_contract_versioning,
    downgrade_contract_if_needed,
    resolve_api_contract_version,
    validate_contract_compatibility,
)
from services.response.contract_validator import ContractValidationError, validate_contract_envelope
from services.response.response_normalizer import apply_consultant_response_normalization


def _structured_payload() -> dict:
    return {
        "answer": (
            "Verified catalog comparison:\n"
            "- Gulfstream G650: large class; practical range 7000 nm.\n"
            "- Falcon 8X: large class; practical range 6450 nm."
        ),
        "sources": [],
        "data_used": {
            "authority_dispatch_kind": "comparison",
            "comparison_v2": {"status": "OK", "models": ["Gulfstream G650", "Falcon 8X"]},
        },
        "aircraft_images": [],
        "error": None,
    }


def _full_v3_payload() -> dict:
    payload = apply_consultant_response_normalization(
        _structured_payload(),
        context={"query": "G650 vs Falcon 8X"},
    )
    return downgrade_contract_if_needed(payload, "v3")


def test_resolve_api_contract_version_header_priority():
    assert resolve_api_contract_version({"X-HyeAero-Version": "v2"}) == "v2"
    assert resolve_api_contract_version({"headers": {"X-HyeAero-Version": "v1"}}) == "v1"
    assert resolve_api_contract_version({"X-HyeAero-Version": "v3", "client_version": "v1"}) == "v3"


def test_resolve_api_contract_version_client_capability_fallback():
    assert resolve_api_contract_version({"client_version": "v2"}) == "v2"
    assert resolve_api_contract_version({}) == "v3"


def test_v3_returns_full_structured_and_ui_contract():
    payload = _full_v3_payload()
    out = downgrade_contract_if_needed(payload, "v3")
    assert out["data_used"]["api_contract_version"] == "v3"
    assert isinstance(out.get("normalized_response"), dict)
    assert isinstance(out.get("ui_render_contract"), dict)
    assert validate_contract_compatibility(out, "v3") is True


def test_v2_returns_normalized_only():
    payload = _full_v3_payload()
    out = downgrade_contract_if_needed(payload, "v2")
    assert out["data_used"]["api_contract_version"] == "v2"
    assert isinstance(out.get("normalized_response"), dict)
    assert "ui_render_contract" not in out
    assert "ui_render_contract" not in out["data_used"]
    assert validate_contract_compatibility(out, "v2") is True


def test_v1_strips_ui_contract_completely():
    payload = _full_v3_payload()
    out = downgrade_contract_if_needed(payload, "v1")
    assert out["data_used"]["api_contract_version"] == "v1"
    assert "normalized_response" not in out
    assert "ui_render_contract" not in out
    assert "normalized_response" not in out["data_used"]
    assert "ui_render_contract" not in out["data_used"]
    assert "structured_sections" not in out["data_used"]
    assert out["data_used"].get("legacy_intent_type") == "comparison"
    assert validate_contract_compatibility(out, "v1") is True


def test_downgrade_is_deterministic():
    payload = _full_v3_payload()
    first = downgrade_contract_if_needed(payload, "v2")
    second = downgrade_contract_if_needed(payload, "v2")
    assert first == second


def test_apply_api_contract_versioning_uses_request_context():
    payload = _full_v3_payload()
    v2 = apply_api_contract_versioning(payload, {"client_version": "v2"})
    assert v2["data_used"]["api_contract_version"] == "v2"
    assert "ui_render_contract" not in v2


def test_invalid_schema_fields_rejected():
    payload = _full_v3_payload()
    payload["normalized_response"]["unexpected_field"] = True
    with pytest.raises(ContractValidationError):
        validate_contract_envelope(payload, "v3")


def test_ui_contract_only_allowed_in_v3():
    payload = _full_v3_payload()
    v2 = downgrade_contract_if_needed(payload, "v2")
    with pytest.raises(ContractValidationError):
        validate_contract_envelope(
            {
                **v2,
                "ui_render_contract": {"ui_intent": "comparison", "layout_type": "side_by_side"},
            },
            "v2",
        )


def test_no_kernel_leakage_in_any_version():
    payload = _full_v3_payload()
    payload["answer"] = "OPERATIONAL SYNTHESIS (AUTHORITATIVE) comparison text"
    with pytest.raises(ContractValidationError):
        downgrade_contract_if_needed(payload, "v3")


def test_invalid_intent_type_rejected():
    payload = _full_v3_payload()
    payload["normalized_response"]["intent_type"] = "invalid_intent"
    with pytest.raises(ContractValidationError):
        validate_contract_envelope(payload, "v3")


def test_invalid_verdict_rejected():
    payload = _full_v3_payload()
    payload["normalized_response"]["verdict"] = "MAYBE OK"
    with pytest.raises(ContractValidationError):
        validate_contract_envelope(payload, "v3")
