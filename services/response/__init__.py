"""Consultant response normalization and broker-grade output schemas."""

from services.response.api_contract_versioning import (
    apply_api_contract_versioning,
    downgrade_contract_if_needed,
    resolve_api_contract_version,
)
from services.response.contract_validator import validate_contract_compatibility
from services.response.response_normalizer import (
    NormalizedConsultantResponse,
    apply_consultant_response_normalization,
    normalize_consultant_response,
)
from services.response.ui_render_contract import (
    UIRenderContract,
    apply_ui_render_contract_to_response,
    build_ui_render_contract,
)

__all__ = [
    "NormalizedConsultantResponse",
    "UIRenderContract",
    "apply_api_contract_versioning",
    "apply_consultant_response_normalization",
    "apply_ui_render_contract_to_response",
    "build_ui_render_contract",
    "downgrade_contract_if_needed",
    "normalize_consultant_response",
    "resolve_api_contract_version",
    "validate_contract_compatibility",
]
