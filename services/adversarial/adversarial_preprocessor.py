"""Adversarial pre-processor — runs before IntentLock and downstream pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.adversarial.budget_conflict_normalizer import BudgetConflictState, normalize_budget_conflicts
from services.adversarial.intent_sanitizer import sanitize_intents
from services.adversarial.model_adversary_resolver import (
    AdversaryResolvedModel,
    AmbiguityType,
    resolve_adversary_models,
)
from services.adversarial.query_conflict_detector import (
    ConflictSeverity,
    ConflictType,
    QueryConflictReport,
    detect_query_conflicts,
)

COMPARISON_CONFIDENCE_THRESHOLD = 70


@dataclass
class CleanNormalizedQuery:
    normalized_query: str
    conflict_report: QueryConflictReport
    budget_state: BudgetConflictState
    resolved_models: List[AdversaryResolvedModel] = field(default_factory=list)
    resolved_intent: Optional[str] = None
    canonical_aircraft_tokens: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "normalized_query": self.normalized_query,
            "resolved_intent": self.resolved_intent,
            "canonical_aircraft_tokens": list(self.canonical_aircraft_tokens),
            "conflict_report": {
                "conflict_type": [c.value for c in self.conflict_report.conflict_type],
                "severity": self.conflict_report.severity.value,
                "resolved_intent_override": self.conflict_report.resolved_intent_override,
                "normalized_query_tokens": list(self.conflict_report.normalized_query_tokens),
                "details": list(self.conflict_report.details),
            },
            "budget_state": {
                "feasibility": self.budget_state.feasibility.value,
                "budget_caps_musd": list(self.budget_state.budget_caps_musd),
                "primary_cap_musd": self.budget_state.primary_cap_musd,
                "acquisition_cap_musd": self.budget_state.acquisition_cap_musd,
                "listing_ask_musd": self.budget_state.listing_ask_musd,
                "reason": self.budget_state.reason,
            },
            "resolved_models": [
                {
                    "canonical_model": m.canonical_model,
                    "alias_chain": list(m.alias_chain),
                    "resolution_confidence": m.resolution_confidence,
                    "ambiguity_type": m.ambiguity_type.value,
                }
                for m in self.resolved_models
            ],
        }


def to_unified_adversarial_metadata(clean: CleanNormalizedQuery) -> Dict[str, Any]:
    """Structured ``UnifiedBrokerState.adversarial`` payload (metadata only)."""
    return {
        "normalized_query": clean.normalized_query,
        "conflict_report": clean.to_dict()["conflict_report"],
        "resolved_models": clean.to_dict()["resolved_models"],
        "resolved_intent": clean.resolved_intent,
        "budget_feasibility": clean.budget_state.feasibility.value,
    }


def _append_canonical_tokens(q: str, models: List[AdversaryResolvedModel]) -> str:
    out = q
    for m in models:
        if m.canonical_model and m.canonical_model.lower() not in out.lower():
            if m.ambiguity_type != AmbiguityType.NONE:
                out = f"{out} [{m.canonical_model}]".strip()
    return out


def preprocess_adversarial_query(
    query: str,
    *,
    data_used: Optional[Dict[str, Any]] = None,
) -> CleanNormalizedQuery:
    """
    Normalize adversarial input before IntentLock / dispatch / market layers.

    Pipeline order: model resolver → budget normalizer → conflict detector → intent sanitizer.
    Does not change pricing math — only query text and metadata stamps.
    """
    raw = (query or "").strip()

    resolved = resolve_adversary_models(raw)
    canon_tokens = [m.canonical_model for m in resolved if m.canonical_model]

    budget = normalize_budget_conflicts(raw, resolved_models=canon_tokens)
    conflict = detect_query_conflicts(raw, budget_state=budget, resolved_models=canon_tokens)

    intent_override = sanitize_intents(raw, existing_override=conflict.resolved_intent_override)
    if intent_override:
        conflict.resolved_intent_override = intent_override

    normalized = _append_canonical_tokens(raw, resolved)

    clean = CleanNormalizedQuery(
        normalized_query=normalized,
        conflict_report=conflict,
        budget_state=budget,
        resolved_models=resolved,
        resolved_intent=intent_override,
        canonical_aircraft_tokens=canon_tokens,
    )

    if isinstance(data_used, dict):
        data_used["clean_normalized_query"] = clean.to_dict()
        data_used["adversarial"] = to_unified_adversarial_metadata(clean)
        data_used["adversarial_preprocess"] = {
            "severity": conflict.severity.value,
            "budget_feasibility": budget.feasibility.value,
        }

    return clean


def get_pipeline_query(query: str, data_used: Optional[Dict[str, Any]] = None) -> str:
    """Return normalized query from data_used when present."""
    if isinstance(data_used, dict):
        raw = data_used.get("clean_normalized_query")
        if isinstance(raw, dict) and raw.get("normalized_query"):
            return str(raw["normalized_query"]).strip()
        adv = data_used.get("adversarial")
        if isinstance(adv, dict) and adv.get("normalized_query"):
            return str(adv["normalized_query"]).strip()
    return (query or "").strip()


def _adversarial_from_data_used(data_used: Optional[Dict[str, Any]]) -> Optional[CleanNormalizedQuery]:
    if not isinstance(data_used, dict):
        return None
    raw = data_used.get("clean_normalized_query")
    if not isinstance(raw, dict):
        return None
    cr = raw.get("conflict_report") or {}
    bs = raw.get("budget_state") or {}

    types = tuple(ConflictType(t) for t in cr.get("conflict_type") or [])
    report = QueryConflictReport(
        conflict_type=types,
        severity=ConflictSeverity(cr.get("severity") or "LOW"),
        resolved_intent_override=cr.get("resolved_intent_override"),
        normalized_query_tokens=tuple(cr.get("normalized_query_tokens") or []),
        details=tuple(cr.get("details") or []),
    )
    from services.adversarial.budget_conflict_normalizer import BudgetFeasibility

    budget = BudgetConflictState(
        feasibility=BudgetFeasibility(bs.get("feasibility") or "FEASIBLE"),
        budget_caps_musd=tuple(bs.get("budget_caps_musd") or []),
        primary_cap_musd=bs.get("primary_cap_musd"),
        acquisition_cap_musd=bs.get("acquisition_cap_musd"),
        listing_ask_musd=bs.get("listing_ask_musd"),
        reason=str(bs.get("reason") or ""),
    )
    models = [
        AdversaryResolvedModel(
            canonical_model=str(m.get("canonical_model") or ""),
            alias_chain=tuple(m.get("alias_chain") or []),
            resolution_confidence=int(m.get("resolution_confidence") or 0),
            ambiguity_type=AmbiguityType(m.get("ambiguity_type") or "NONE"),
        )
        for m in raw.get("resolved_models") or []
    ]
    return CleanNormalizedQuery(
        normalized_query=str(raw.get("normalized_query") or ""),
        conflict_report=report,
        budget_state=budget,
        resolved_models=models,
        resolved_intent=raw.get("resolved_intent"),
        canonical_aircraft_tokens=list(raw.get("canonical_aircraft_tokens") or []),
    )


def try_adversarial_buy_block(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Return a deterministic block response for adversarial buy queries, or None.

    HIGH severity conflicts → CLARIFICATION_REQUIRED
    INFEASIBLE budget → INFEASIBLE_BUDGET_CONSTRAINT
    """
    clean = _adversarial_from_data_used(data_used)
    if clean is None:
        clean = preprocess_adversarial_query(query, data_used=data_used)

    cr = clean.conflict_report
    bs = clean.budget_state

    if bs.feasibility.value == "INFEASIBLE":
        return (
            "Aircraft: (constraint review)\n\n"
            "Market Reality:\n"
            f"- {bs.reason}\n\n"
            "Verdict:\n"
            "INFEASIBLE_BUDGET_CONSTRAINT"
        )

    if cr.severity == ConflictSeverity.HIGH:
        details = "; ".join(cr.details[:3]) if cr.details else "conflicting constraints detected"
        return (
            "Aircraft: (clarification required)\n\n"
            "Market Reality:\n"
            f"- Adversarial input normalized — {details}.\n"
            "- Resolve budget vs model class before a buy verdict.\n\n"
            "Verdict:\n"
            "CLARIFICATION_REQUIRED"
        )

    if ConflictType.BUDGET_MODEL_INFEASIBLE in cr.conflict_type:
        return (
            "Aircraft: (constraint review)\n\n"
            "Market Reality:\n"
            "- Budget and aircraft class appear incompatible in this query.\n\n"
            "Verdict:\n"
            "INFEASIBLE_BUDGET_CONSTRAINT"
        )

    return None


def check_comparison_safety(
    query: str,
    data_used: Optional[Dict[str, Any]] = None,
    *,
    compare_models: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Block comparison when model resolution confidence is below threshold.

    Returns deterministic clarification text or None to proceed.
    """
    clean = _adversarial_from_data_used(data_used)
    if clean is None:
        clean = preprocess_adversarial_query(query, data_used=data_used)

    if compare_models and len(compare_models) < 2:
        return (
            "Insufficient verified aircraft data to produce a comparison.\n\n"
            "CLARIFICATION_REQUIRED: Resolve both aircraft models explicitly "
            "(catalog names required)."
        )

    low_conf: List[str] = []
    for m in clean.resolved_models:
        if m.resolution_confidence < COMPARISON_CONFIDENCE_THRESHOLD and m.ambiguity_type != AmbiguityType.NONE:
            low_conf.append(m.canonical_model)

    if low_conf and (not compare_models or len(compare_models) < 2):
        names = ", ".join(low_conf[:2])
        return (
            f"Insufficient verified aircraft data to produce a comparison for {names}.\n\n"
            "CLARIFICATION_REQUIRED: Ambiguous model reference — specify catalog names "
            "(e.g. Citation Longitude, Gulfstream G650)."
        )

    if ConflictType.MODEL_AMBIGUOUS in clean.conflict_report.conflict_type:
        if not compare_models or len(compare_models) < 2:
            return (
                "Insufficient verified aircraft data to produce a comparison.\n\n"
                "CLARIFICATION_REQUIRED: Query contains ambiguous aircraft naming."
            )

    return None
