"""Cross-pipeline canonical aircraft identity resolution (deterministic)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_LAYER_INTENT = "intent_lock"
_LAYER_AKAL = "akal"
_LAYER_DISPATCH = "dispatch"
_LAYER_MARKET_INTEL = "market_intel"
_LAYER_RECOVERY = "recovery"
_LAYER_COMPARISON = "comparison"


@dataclass(frozen=True)
class CanonicalAircraftIdentity:
    canonical_model: str
    aliases_used: Tuple[str, ...]
    source_layers: Tuple[str, ...]
    confidence_score: int
    resolved_from_query_tokens: Tuple[str, ...]


def _norm_model(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip()).lower()


def _authority_canonical(raw: str) -> Optional[str]:
    from services.aircraft.aircraft_authority_service import (
        get_aircraft_authority_record,
        resolve_aircraft_alias,
    )
    from services.catalog.catalog_alias_resolver import resolve_canonical_display_name

    token = (raw or "").strip()
    if not token:
        return None
    alias = resolve_aircraft_alias(token)
    if alias:
        rec = get_aircraft_authority_record(aircraft_model=alias)
        if rec:
            return rec.canonical_name
    display = resolve_canonical_display_name(token)
    if display:
        rec = get_aircraft_authority_record(aircraft_model=display)
        if rec:
            return rec.canonical_name
        return display
    rec = get_aircraft_authority_record(aircraft_model=token)
    if rec:
        return rec.canonical_name
    return None


def _tokens_from_query(query: str) -> Tuple[str, ...]:
    from services.consultant.recommendation_engine import detect_models_from_text

    detected = detect_models_from_text(query or "")
    parts: List[str] = list(detected)
    for m in re.findall(r"\b([A-Za-z][\w\s+\-]{2,40})\b", query or ""):
        if any(k in m.lower() for k in ("citation", "gulfstream", "falcon", "global", "challenger", "phenom")):
            parts.append(m.strip())
    seen: Set[str] = set()
    out: List[str] = []
    for p in parts:
        k = _norm_model(p)
        if k and k not in seen:
            seen.add(k)
            out.append(p.strip())
    return tuple(out[:8])


def _models_from_data_used(data_used: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Map layer name -> model string observed in data_used."""
    found: Dict[str, str] = {}
    if not isinstance(data_used, dict):
        return found

    lock = data_used.get("intent_lock")
    if isinstance(lock, dict):
        models = lock.get("canonical_models") or []
        if isinstance(models, list) and models:
            found[_LAYER_INTENT] = str(models[0]).strip()

    bdd = data_used.get("buy_decision_dispatch")
    if isinstance(bdd, dict) and bdd.get("model"):
        found[_LAYER_DISPATCH] = str(bdd["model"]).strip()

    mi = data_used.get("market_intelligence")
    if isinstance(mi, dict):
        snap = mi.get("snapshot")
        if isinstance(snap, dict) and snap.get("model"):
            found[_LAYER_MARKET_INTEL] = str(snap["model"]).strip()

    rec = data_used.get("aircraft_authority_record")
    if isinstance(rec, dict) and rec.get("canonical_name"):
        found[_LAYER_AKAL] = str(rec["canonical_name"]).strip()

    cv2 = data_used.get("comparison_v2")
    if isinstance(cv2, dict):
        models = cv2.get("models")
        if isinstance(models, list) and models:
            found[_LAYER_COMPARISON] = str(models[0]).strip()

    verified = data_used.get("verified_recovery_models") or data_used.get("recovery_authority_models")
    if isinstance(verified, list) and verified:
        found[_LAYER_RECOVERY] = str(verified[0]).strip()

    ubs = data_used.get("unified_broker_state")
    if isinstance(ubs, dict) and ubs.get("canonical_model"):
        found["unified_broker_state"] = str(ubs["canonical_model"]).strip()

    return found


def _confidence_score(
    canonical: str,
    layer_models: Dict[str, str],
    aliases: Sequence[str],
) -> int:
    if not canonical:
        return 0
    score = 100
    canon_norm = _norm_model(canonical)
    distinct = {_norm_model(v) for v in layer_models.values() if v}
    distinct.discard(canon_norm)
    for other in distinct:
        other_canon = _authority_canonical(other) or other
        if _norm_model(other_canon) != canon_norm:
            score -= 15
    if len(aliases) > 1:
        score -= min(10, (len(aliases) - 1) * 3)
    return max(0, min(100, score))


def resolve_canonical_identity(
    *,
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
    explicit_model: Optional[str] = None,
    source_layer: str = _LAYER_DISPATCH,
) -> CanonicalAircraftIdentity:
    """
    Resolve a single canonical identity across intent, dispatch, market intel, recovery.
    """
    tokens = _tokens_from_query(query)
    layer_models = _models_from_data_used(data_used)
    aliases: List[str] = []

    if explicit_model:
        layer_models[source_layer] = explicit_model.strip()
        aliases.append(explicit_model.strip())

    for layer, model in layer_models.items():
        if model and model not in aliases:
            aliases.append(model)

    for tok in tokens:
        if tok not in aliases:
            aliases.append(tok)

    canonical: Optional[str] = None
    for candidate in (
        explicit_model,
        layer_models.get(_LAYER_AKAL),
        layer_models.get(_LAYER_DISPATCH),
        layer_models.get(_LAYER_MARKET_INTEL),
        layer_models.get(_LAYER_INTENT),
        layer_models.get(_LAYER_RECOVERY),
        layer_models.get(_LAYER_COMPARISON),
        tokens[0] if tokens else None,
    ):
        if not candidate:
            continue
        resolved = _authority_canonical(candidate)
        if resolved:
            canonical = resolved
            break

    if not canonical and aliases:
        canonical = _authority_canonical(aliases[0]) or aliases[0].strip()

    canonical = canonical or ""
    active_layers = tuple(sorted(layer_models.keys()))
    conf = _confidence_score(canonical, layer_models, aliases)

    return CanonicalAircraftIdentity(
        canonical_model=canonical,
        aliases_used=tuple(aliases),
        source_layers=active_layers,
        confidence_score=conf,
        resolved_from_query_tokens=tokens,
    )


def resolve_comparison_identities(
    query: str,
    compare_models: Sequence[str],
    data_used: Optional[Dict[str, Any]] = None,
) -> Tuple[CanonicalAircraftIdentity, CanonicalAircraftIdentity]:
    """Resolve both comparison aircraft through the same identity resolver."""
    from services.comparison.aircraft_registry_lock import lock_comparison_aircraft

    models = [m for m in compare_models if (m or "").strip()]
    lock = lock_comparison_aircraft(models)
    canonical_pair = list(lock.canonical)[:2]
    while len(canonical_pair) < 2:
        canonical_pair.append("")

    id_a = resolve_canonical_identity(
        query=query,
        data_used=data_used,
        explicit_model=canonical_pair[0],
        source_layer=_LAYER_COMPARISON,
    )
    id_b = resolve_canonical_identity(
        query=query,
        data_used=data_used,
        explicit_model=canonical_pair[1] if len(canonical_pair) > 1 else "",
        source_layer=_LAYER_COMPARISON,
    )
    return id_a, id_b
