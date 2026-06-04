"""
Strict schema validation for versioned API contract envelopes.
"""

from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet, List, Optional, Set

SUPPORTED_API_VERSIONS: FrozenSet[str] = frozenset({"v1", "v2", "v3"})

_INTENT_TYPES: FrozenSet[str] = frozenset(
    {"comparison", "alternative", "buy_decision", "mission", "fact", "other"}
)
_VERDICTS: FrozenSet[str] = frozenset(
    {
        "",
        "GOOD FIT",
        "CONDITIONAL FIT",
        "NOT A FIT",
        "GOOD DEAL",
        "OVERPRICED",
        "RISKY",
        "VIABLE WITH COMPROMISES",
    }
)
_UI_INTENTS: FrozenSet[str] = _INTENT_TYPES
_LAYOUT_TYPES: FrozenSet[str] = frozenset(
    {"side_by_side", "ranked_list", "deal_card", "mission_brief", "info_card"}
)
_SECTION_TYPES: FrozenSet[str] = frozenset({"overview", "analysis", "recommendation", "risks"})
_RENDER_MODES: FrozenSet[str] = frozenset({"text", "bullet", "table"})

_RESPONSE_TOP_LEVEL_V1: FrozenSet[str] = frozenset(
    {"answer", "sources", "data_used", "aircraft_images", "error"}
)
_RESPONSE_TOP_LEVEL_V2: FrozenSet[str] = _RESPONSE_TOP_LEVEL_V1 | frozenset({"normalized_response"})
_RESPONSE_TOP_LEVEL_V3: FrozenSet[str] = _RESPONSE_TOP_LEVEL_V2 | frozenset({"ui_render_contract"})

_NORMALIZED_KEYS: FrozenSet[str] = frozenset(
    {
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
    }
)
_STRUCTURED_SECTION_KEYS: FrozenSet[str] = frozenset({"overview", "analysis", "recommendation", "risks"})
_DATA_SOURCES_KEYS: FrozenSet[str] = frozenset({"phly_used", "tavily_used", "market_used"})

_UI_CONTRACT_KEYS: FrozenSet[str] = frozenset(
    {
        "ui_intent",
        "layout_type",
        "primary_cards",
        "secondary_cards",
        "risk_cards",
        "financial_cards",
        "headline",
        "subheadline",
        "sections",
        "ui_flags",
        "render_hints",
    }
)
_UI_FLAGS_KEYS: FrozenSet[str] = frozenset(
    {
        "show_verdict_badge",
        "show_price_comparison",
        "show_risk_panel",
        "show_mission_fit_meter",
    }
)
_RENDER_HINTS_KEYS: FrozenSet[str] = frozenset(
    {
        "comparison_mode",
        "alternative_mode",
        "buy_mode",
        "mission_mode",
        "single_authority",
    }
)

_V1_DATA_USED_CONTRACT_KEYS: FrozenSet[str] = frozenset(
    {
        "normalized_response",
        "ui_render_contract",
        "ui_render_contract_applied",
        "response_normalization_applied",
        "structured_sections",
    }
)

_LEAKAGE_RE = re.compile(
    r"\b(?:operational\s+synthesis|query_recommendation_intent|unified_intent|"
    r"mission_authority_kernel|pre_llm_recommendation)\b",
    re.I,
)


class ContractValidationError(ValueError):
    """Raised when a response envelope violates the versioned contract schema."""


def validate_contract_envelope(response: Dict[str, Any], version: str) -> None:
    """Validate response envelope for the given API contract version."""
    v = _normalize_version_token(version)
    if v not in SUPPORTED_API_VERSIONS:
        raise ContractValidationError(f"unsupported api contract version: {version}")

    if not isinstance(response, dict):
        raise ContractValidationError("response must be a dict")

    allowed_top = {
        "v1": _RESPONSE_TOP_LEVEL_V1,
        "v2": _RESPONSE_TOP_LEVEL_V2,
        "v3": _RESPONSE_TOP_LEVEL_V3,
    }[v]
    unknown_top = set(response.keys()) - allowed_top
    if unknown_top:
        raise ContractValidationError(f"v{v} unknown top-level keys: {sorted(unknown_top)}")

    du = response.get("data_used")
    if du is not None and not isinstance(du, dict):
        raise ContractValidationError("data_used must be a dict when present")

    if v in ("v2", "v3"):
        norm = response.get("normalized_response")
        if norm is not None:
            if not isinstance(norm, dict):
                raise ContractValidationError(f"v{v} normalized_response must be a dict")
            _validate_normalized_response(norm, version=v)

    if v == "v3":
        ui = response.get("ui_render_contract")
        if ui is not None:
            if not isinstance(ui, dict):
                raise ContractValidationError("ui_render_contract must be a dict")
            _validate_ui_render_contract(ui)
        if isinstance(du, dict) and du.get("ui_render_contract") is not None:
            if not isinstance(du.get("ui_render_contract"), dict):
                raise ContractValidationError("data_used.ui_render_contract must be a dict")
            _validate_ui_render_contract(du["ui_render_contract"])

    if v == "v2":
        if response.get("ui_render_contract") is not None:
            raise ContractValidationError("v2 must not expose ui_render_contract at top level")
        if isinstance(du, dict) and du.get("ui_render_contract") is not None:
            raise ContractValidationError("v2 must not expose ui_render_contract in data_used")

    if v == "v1":
        if response.get("normalized_response") is not None:
            raise ContractValidationError("v1 must not expose normalized_response at top level")
        if response.get("ui_render_contract") is not None:
            raise ContractValidationError("v1 must not expose ui_render_contract at top level")
        if isinstance(du, dict):
            leaked = _V1_DATA_USED_CONTRACT_KEYS.intersection(du.keys())
            if leaked:
                raise ContractValidationError(
                    f"v1 must not expose contract keys in data_used: {sorted(leaked)}"
                )

    _assert_no_leakage(response)


def validate_contract_compatibility(response: Dict[str, Any], version: str) -> bool:
    """Return True when response matches the versioned contract schema."""
    try:
        validate_contract_envelope(response, version)
        return True
    except ContractValidationError:
        return False


def _validate_normalized_response(norm: Dict[str, Any], *, version: str) -> None:
    unknown = set(norm.keys()) - _NORMALIZED_KEYS
    if unknown:
        raise ContractValidationError(f"normalized_response unknown keys: {sorted(unknown)}")

    intent = str(norm.get("intent_type") or "")
    if intent not in _INTENT_TYPES:
        raise ContractValidationError(f"invalid intent_type: {intent}")

    verdict = str(norm.get("verdict") or "").upper()
    if verdict and verdict not in _VERDICTS:
        raise ContractValidationError(f"invalid verdict: {verdict}")

    sections = norm.get("structured_sections")
    if sections is not None:
        if not isinstance(sections, dict):
            raise ContractValidationError("structured_sections must be a dict")
        unknown_sec = set(sections.keys()) - _STRUCTURED_SECTION_KEYS
        if unknown_sec:
            raise ContractValidationError(
                f"structured_sections unknown keys: {sorted(unknown_sec)}"
            )

    sources = norm.get("data_sources")
    if sources is not None:
        if not isinstance(sources, dict):
            raise ContractValidationError("data_sources must be a dict")
        unknown_src = set(sources.keys()) - _DATA_SOURCES_KEYS
        if unknown_src:
            raise ContractValidationError(f"data_sources unknown keys: {sorted(unknown_src)}")

    if version == "v1" and sections:
        raise ContractValidationError("v1 must not include structured_sections")


def _validate_ui_render_contract(ui: Dict[str, Any]) -> None:
    unknown = set(ui.keys()) - _UI_CONTRACT_KEYS
    if unknown:
        raise ContractValidationError(f"ui_render_contract unknown keys: {sorted(unknown)}")

    ui_intent = str(ui.get("ui_intent") or "")
    if ui_intent not in _UI_INTENTS:
        raise ContractValidationError(f"invalid ui_intent: {ui_intent}")

    layout = str(ui.get("layout_type") or "")
    if layout not in _LAYOUT_TYPES:
        raise ContractValidationError(f"invalid layout_type: {layout}")

    flags = ui.get("ui_flags")
    if flags is not None:
        if not isinstance(flags, dict):
            raise ContractValidationError("ui_flags must be a dict")
        unknown_flags = set(flags.keys()) - _UI_FLAGS_KEYS
        if unknown_flags:
            raise ContractValidationError(f"ui_flags unknown keys: {sorted(unknown_flags)}")

    hints = ui.get("render_hints")
    if hints is not None:
        if not isinstance(hints, dict):
            raise ContractValidationError("render_hints must be a dict")
        unknown_hints = set(hints.keys()) - _RENDER_HINTS_KEYS
        if unknown_hints:
            raise ContractValidationError(f"render_hints unknown keys: {sorted(unknown_hints)}")

    sections = ui.get("sections")
    if sections is not None:
        if not isinstance(sections, list):
            raise ContractValidationError("sections must be a list")
        for idx, section in enumerate(sections):
            if not isinstance(section, dict):
                raise ContractValidationError(f"sections[{idx}] must be a dict")
            stype = str(section.get("type") or "")
            if stype not in _SECTION_TYPES:
                raise ContractValidationError(f"invalid section type: {stype}")
            mode = str(section.get("render_mode") or "")
            if mode and mode not in _RENDER_MODES:
                raise ContractValidationError(f"invalid render_mode: {mode}")


def _assert_no_leakage(response: Dict[str, Any]) -> None:
    answer = str(response.get("answer") or "")
    if _LEAKAGE_RE.search(answer):
        raise ContractValidationError("answer contains forbidden internal routing/kernel tokens")

    norm = response.get("normalized_response")
    if isinstance(norm, dict):
        blob = str(norm)
        if _LEAKAGE_RE.search(blob):
            raise ContractValidationError("normalized_response contains forbidden internal tokens")

    ui = response.get("ui_render_contract")
    if isinstance(ui, dict):
        blob = str(ui)
        if _LEAKAGE_RE.search(blob):
            raise ContractValidationError("ui_render_contract contains forbidden internal tokens")


def _normalize_version_token(raw: str) -> str:
    token = str(raw or "").strip().lower()
    if token in SUPPORTED_API_VERSIONS:
        return token
    if token in ("1", "2", "3"):
        return f"v{token}"
    if token.startswith("v") and token[1:] in ("1", "2", "3"):
        return token
    return ""


__all__ = [
    "ContractValidationError",
    "SUPPORTED_API_VERSIONS",
    "validate_contract_compatibility",
    "validate_contract_envelope",
]
