"""
Versioned API contract resolution and backward-compatible response downgrades.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from services.response.contract_validator import (
    SUPPORTED_API_VERSIONS,
    ContractValidationError,
    validate_contract_envelope,
)

DEFAULT_API_CONTRACT_VERSION = "v3"

_V1_STRIP_DATA_USED_KEYS = frozenset(
    {
        "normalized_response",
        "ui_render_contract",
        "ui_render_contract_applied",
        "response_normalization_applied",
        "structured_sections",
    }
)


def resolve_api_contract_version(request_context: Optional[Dict[str, Any]] = None) -> str:
    """
    Resolve API contract version from request context.

    Priority:
    1. Header ``X-HyeAero-Version``
    2. ``client_version`` capability flag
    3. Default ``v3``
    """
    ctx = request_context if isinstance(request_context, dict) else {}

    headers = ctx.get("headers")
    header_val = ctx.get("X-HyeAero-Version")
    if not header_val and isinstance(headers, dict):
        header_val = headers.get("X-HyeAero-Version") or headers.get("x-hyeaero-version")

    version = _normalize_version_token(header_val)
    if version:
        return version

    version = _normalize_version_token(ctx.get("client_version"))
    if version:
        return version

    return DEFAULT_API_CONTRACT_VERSION


def validate_contract_compatibility(response: Dict[str, Any], version: str) -> bool:
    """Return True when *response* satisfies the schema for *version*."""
    try:
        validate_contract_envelope(response, version)
        return True
    except ContractValidationError:
        return False


def downgrade_contract_if_needed(
    response: Dict[str, Any],
    target_version: str,
) -> Dict[str, Any]:
    """Downgrade a full v3-shaped response envelope to *target_version*."""
    version = _normalize_version_token(target_version) or DEFAULT_API_CONTRACT_VERSION
    if version not in SUPPORTED_API_VERSIONS:
        version = DEFAULT_API_CONTRACT_VERSION

    out = _promote_contract_fields(dict(response))
    du = dict(out.get("data_used") or {})

    normalized = out.get("normalized_response")
    ui_contract = out.get("ui_render_contract")

    if version == "v3":
        if isinstance(normalized, dict):
            cleaned = _clean_normalized(normalized)
            out["normalized_response"] = cleaned
            du["normalized_response"] = cleaned
        if isinstance(ui_contract, dict):
            out["ui_render_contract"] = dict(ui_contract)
            du["ui_render_contract"] = dict(ui_contract)
            du["ui_render_contract_applied"] = 1
        du["api_contract_version"] = "v3"
        out["data_used"] = du
        validate_contract_envelope(out, "v3")
        return out

    if version == "v2":
        out.pop("ui_render_contract", None)
        for key in ("ui_render_contract", "ui_render_contract_applied"):
            du.pop(key, None)
        if isinstance(normalized, dict):
            cleaned = _clean_normalized(normalized)
            out["normalized_response"] = cleaned
            du["normalized_response"] = cleaned
        else:
            out.pop("normalized_response", None)
            du.pop("normalized_response", None)
        du["api_contract_version"] = "v2"
        du["response_normalization_applied"] = du.get("response_normalization_applied", 1)
        out["data_used"] = du
        validate_contract_envelope(out, "v2")
        return out

    # v1 legacy-compatible flat envelope
    out.pop("normalized_response", None)
    out.pop("ui_render_contract", None)
    for key in _V1_STRIP_DATA_USED_KEYS:
        du.pop(key, None)
    if isinstance(normalized, dict):
        _apply_v1_legacy_flatten(du, normalized)
    du["api_contract_version"] = "v1"
    out["data_used"] = du
    validate_contract_envelope(out, "v1")
    return out


def apply_api_contract_versioning(
    response: Dict[str, Any],
    request_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve version and apply deterministic downgrade at the response boundary."""
    if not isinstance(response, dict):
        return response
    version = resolve_api_contract_version(request_context)
    return downgrade_contract_if_needed(response, version)


def _promote_contract_fields(response: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure normalized/UI contract fields are available at top level when nested in data_used."""
    out = copy.deepcopy(response)
    du = dict(out.get("data_used") or {})
    if not isinstance(out.get("normalized_response"), dict) and isinstance(
        du.get("normalized_response"), dict
    ):
        out["normalized_response"] = dict(du["normalized_response"])
    if not isinstance(out.get("ui_render_contract"), dict) and isinstance(
        du.get("ui_render_contract"), dict
    ):
        out["ui_render_contract"] = dict(du["ui_render_contract"])
    out["data_used"] = du
    return out


def _clean_normalized(normalized: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(normalized)
    for key in list(cleaned.keys()):
        if key not in {
            "intent_type",
            "aircraft",
            "primary_recommendation",
            "alternatives",
            "financial_summary",
            "mission_fit",
            "verdict",
            "confidence",
            "notes",
            "structured_sections",
            "data_sources",
        }:
            cleaned.pop(key, None)
    return cleaned


def _apply_v1_legacy_flatten(data_used: Dict[str, Any], normalized: Dict[str, Any]) -> None:
    """Expose minimal legacy metadata without structured contract fields."""
    intent = str(normalized.get("intent_type") or "").strip()
    verdict = str(normalized.get("verdict") or "").strip()
    aircraft = normalized.get("aircraft")
    if intent:
        data_used["legacy_intent_type"] = intent
    if verdict:
        data_used["legacy_verdict"] = verdict
    if isinstance(aircraft, list) and aircraft:
        data_used["legacy_aircraft"] = list(aircraft)[:4]


def _normalize_version_token(raw: Any) -> str:
    token = str(raw or "").strip().lower()
    if token in SUPPORTED_API_VERSIONS:
        return token
    if token in ("1", "2", "3"):
        return f"v{token}"
    if token.startswith("v") and len(token) == 2 and token[1] in "123":
        return token
    return ""


__all__ = [
    "DEFAULT_API_CONTRACT_VERSION",
    "ContractValidationError",
    "apply_api_contract_versioning",
    "downgrade_contract_if_needed",
    "resolve_api_contract_version",
    "validate_contract_compatibility",
]
