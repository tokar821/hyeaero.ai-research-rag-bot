"""Entity scope validation for Phly rows and memory consistency."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .scope import EntityScope, aircraft_identities_conflict, normalize_aircraft_label


def phly_row_marketing_label(row: Dict[str, Any]) -> str:
    mfr = str(row.get("manufacturer") or "").strip()
    mdl = str(row.get("model") or "").strip()
    return normalize_aircraft_label(" ".join(x for x in (mfr, mdl) if x))


def phly_row_registration(row: Dict[str, Any]) -> str:
    return str(row.get("registration_number") or "").strip().upper().replace(" ", "")


def phly_row_matches_scope(row: Dict[str, Any], scope: EntityScope) -> bool:
    if not isinstance(row, dict):
        return False
    if scope.scope_type == "tail":
        reg = phly_row_registration(row)
        target = str(scope.scope_value or "").strip().upper().replace(" ", "")
        return bool(reg and target and reg == target)
    if scope.scope_type == "aircraft_model":
        label = phly_row_marketing_label(row)
        target = normalize_aircraft_label(scope.scope_value)
        if not label or not target:
            return False
        if label == target:
            return True
        if target in label or label in target:
            return True
        return bool(set(target.split()) & set(label.split()))
    if scope.scope_type == "comparison":
        return True
    if scope.scope_type == "deictic":
        return True
    return True


def filter_phly_rows_by_entity_scope(
    rows: List[Dict[str, Any]],
    scope: EntityScope,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Reject Phly rows that conflict with the current-turn entity scope."""
    incoming = list(rows or [])
    if not incoming:
        return [], {
            "accepted": 0,
            "rejected": 0,
            "scope_type": scope.scope_type,
            "scope_value": scope.scope_value,
            "rejected_identities": [],
        }

    if scope.scope_type in ("deictic", "none", "comparison"):
        identities = [phly_row_registration(r) or phly_row_marketing_label(r) for r in incoming[:4]]
        return incoming, {
            "accepted": len(incoming),
            "rejected": 0,
            "scope_type": scope.scope_type,
            "scope_value": scope.scope_value,
            "accepted_identities": identities,
        }

    kept: List[Dict[str, Any]] = []
    rejected: List[str] = []
    for row in incoming:
        if phly_row_matches_scope(row, scope):
            kept.append(row)
        else:
            ident = phly_row_registration(row) or phly_row_marketing_label(row)
            if ident:
                rejected.append(ident)

    return kept, {
        "accepted": len(kept),
        "rejected": len(incoming) - len(kept),
        "scope_type": scope.scope_type,
        "scope_value": scope.scope_value,
        "rejected_identities": rejected[:8],
        "accepted_identities": [
            phly_row_registration(r) or phly_row_marketing_label(r) for r in kept[:4]
        ],
    }


def tail_conflicts_with_aircraft(
    tail: Optional[str],
    aircraft: Optional[str],
    *,
    tail_aircraft: Optional[str] = None,
) -> bool:
    """
    True when an active tail should not coexist with the resolved aircraft anchor.

    When ``tail_aircraft`` is known (aircraft previously tied to the tail), conflict is
    evaluated against that label; otherwise any explicit new aircraft clears the tail.
    """
    t = str(tail or "").strip()
    air = str(aircraft or "").strip()
    if not t or not air:
        return False
    basis = (tail_aircraft or "").strip() or air
    return aircraft_identities_conflict(air, basis)


def attach_entity_scope_observability(
    data_used: Dict[str, Any],
    *,
    scope: EntityScope,
    tail_lock_source: str,
    effective_query: str,
    original_query: str,
    phly_lookup_tokens: Optional[List[str]] = None,
    phly_row_identity: Optional[List[str]] = None,
    entity_scope_validation: Optional[Dict[str, Any]] = None,
) -> None:
    """Production-debug metadata only — no prompt or routing changes."""
    aug = ""
    orig = (original_query or "").strip()
    eff = (effective_query or "").strip()
    if eff and orig and eff.lower() != orig.lower():
        if eff.lower().startswith(orig.lower()):
            aug = eff[len(orig) :].strip()
        else:
            aug = eff
    data_used["entity_scope"] = scope.to_dict()
    data_used["tail_lock_source"] = tail_lock_source
    data_used["effective_query_augmentation"] = aug or None
    if phly_lookup_tokens is not None:
        data_used["phly_lookup_tokens"] = list(phly_lookup_tokens)[:16]
    if phly_row_identity is not None:
        data_used["phly_row_identity"] = list(phly_row_identity)[:8]
    if entity_scope_validation is not None:
        data_used["entity_scope_validation"] = entity_scope_validation
