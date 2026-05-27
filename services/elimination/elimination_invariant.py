"""
Elimination invariant — presented models must never include eliminated aircraft.

Enforced at formatter and LLM context boundaries:
  presented_models ∩ eliminated_models == ∅
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from services.consultant.recommendation_engine import AircraftRecommendation

logger = logging.getLogger(__name__)


def _norm(model: str) -> str:
    return (model or "").strip().lower()


def collect_eliminated_models(
    *,
    data_used: Optional[Dict[str, Any]] = None,
    elimination_log: Optional[Sequence[Dict[str, Any]]] = None,
    feasibility_map: Optional[Dict[str, Any]] = None,
    explicit_eliminated: Optional[Sequence[str]] = None,
) -> Set[str]:
    """Union of all eliminated model keys from pipeline artifacts."""
    out: Set[str] = set()

    if explicit_eliminated:
        out.update(_norm(m) for m in explicit_eliminated if m)

    if elimination_log:
        for entry in elimination_log:
            if not isinstance(entry, dict):
                continue
            name = entry.get("aircraft_name") or entry.get("model") or ""
            if name:
                out.add(_norm(name))

    if isinstance(data_used, dict):
        try:
            from services.telemetry.reasoning_packet import IMMUTABLE_PACKET_KEY

            packet = data_used.get(IMMUTABLE_PACKET_KEY)
            if isinstance(packet, dict):
                for m in packet.get("eliminated_models") or []:
                    out.add(_norm(str(m)))
                for entry in packet.get("eliminations") or []:
                    if isinstance(entry, dict):
                        stage = str(entry.get("stage") or "")
                        if stage.startswith("fleet_domain_"):
                            continue
                        name = entry.get("model") or ""
                        if name:
                            out.add(_norm(str(name)))
        except Exception:
            pass
        for key in (
            "corridor_hard_elimination",
            "mountain_field_elimination",
            "airport_constraint_elimination",
            "operational_band_elimination",
        ):
            block = data_used.get(key)
            if isinstance(block, dict):
                for m in block.get("eliminated") or []:
                    out.add(_norm(str(m)))
        pipe = data_used.get("deterministic_recommendation_pipeline")
        if isinstance(pipe, dict):
            for m in pipe.get("eliminated_models") or []:
                out.add(_norm(str(m)))

    if feasibility_map:
        for model, fr in feasibility_map.items():
            feasible = getattr(fr, "feasible", None)
            if feasible is None and isinstance(fr, dict):
                feasible = fr.get("feasible")
            risk = getattr(fr, "operational_risk_level", None)
            if risk is None and isinstance(fr, dict):
                risk = fr.get("operational_risk_level")
            if feasible is False or risk == "eliminated":
                out.add(_norm(model))

    return out


def merge_feasibility_maps(
    base: Dict[str, Any],
    overlay: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge rank-stage feasibility into the pipeline map.

    Overlay wins when it marks a model infeasible or eliminated (stricter result).
    """
    merged = dict(base or {})
    for model, fr in (overlay or {}).items():
        if model not in merged:
            merged[model] = fr
            continue
        existing = merged[model]
        ex_feasible = getattr(existing, "feasible", None)
        if ex_feasible is None and isinstance(existing, dict):
            ex_feasible = existing.get("feasible")
        ov_feasible = getattr(fr, "feasible", None)
        if ov_feasible is None and isinstance(fr, dict):
            ov_feasible = fr.get("feasible")
        ex_risk = getattr(existing, "operational_risk_level", None)
        if ex_risk is None and isinstance(existing, dict):
            ex_risk = existing.get("operational_risk_level")
        ov_risk = getattr(fr, "operational_risk_level", None)
        if ov_risk is None and isinstance(fr, dict):
            ov_risk = fr.get("operational_risk_level")
        if ov_feasible is False or ov_risk == "eliminated":
            merged[model] = fr
    return merged


def collect_hard_eliminated_models(
    *,
    data_used: Optional[Dict[str, Any]] = None,
    elimination_log: Optional[Sequence[Dict[str, Any]]] = None,
    explicit_eliminated: Optional[Sequence[str]] = None,
) -> Set[str]:
    """
    Mission-wide hard eliminations only — excludes feasibility_map range failures.

    Used for fleet composition: domain-local survivors must not be stripped because
    they failed a different domain's stage envelope.
    """
    out: Set[str] = set()

    if explicit_eliminated:
        out.update(_norm(m) for m in explicit_eliminated if m)

    if elimination_log:
        for entry in elimination_log:
            if not isinstance(entry, dict):
                continue
            stage = str(entry.get("stage") or entry.get("hard_elimination_rule") or "")
            if stage and "feasibility" in stage.lower() and "hard" not in stage.lower():
                continue
            name = entry.get("aircraft_name") or entry.get("model") or ""
            if name and (
                entry.get("hard_elimination_rule")
                or "corridor" in stage.lower()
                or "airport" in stage.lower()
                or "hard" in stage.lower()
            ):
                out.add(_norm(name))

    if isinstance(data_used, dict):
        for key in (
            "corridor_hard_elimination",
            "mountain_field_elimination",
            "airport_constraint_elimination",
            "operational_band_elimination",
        ):
            block = data_used.get(key)
            if isinstance(block, dict):
                for m in block.get("eliminated") or []:
                    out.add(_norm(str(m)))

    return out


def filter_eliminated_recommendations(
    recommendations: Sequence[AircraftRecommendation],
    eliminated: Set[str],
) -> List[AircraftRecommendation]:
    """Drop recommendations whose model was eliminated upstream."""
    if not eliminated:
        return list(recommendations)
    return [r for r in recommendations if _norm(r.model) not in eliminated]


def enforce_elimination_invariant(
    recommendations: Sequence[AircraftRecommendation],
    eliminated: Set[str],
    *,
    context: str = "formatter",
) -> List[AircraftRecommendation]:
    """
    Strip eliminated models from recommendations.

    Logs an error only if a model remains after filtering (true invariant breach).
    """
    filtered = filter_eliminated_recommendations(recommendations, eliminated)
    if not eliminated:
        return filtered
    leaked = {_norm(r.model) for r in filtered if _norm(r.model) in eliminated}
    if leaked:
        logger.error(
            "ELIMINATION_INVARIANT_VIOLATION context=%s leaked=%s",
            context,
            sorted(leaked),
        )
    stripped = [_norm(r.model) for r in recommendations if _norm(r.model) in eliminated]
    if stripped and not leaked:
        logger.info(
            "elimination_invariant_stripped context=%s models=%s",
            context,
            stripped,
        )
    return filtered


def assert_elimination_invariant(
    presented_models: Iterable[str],
    eliminated: Set[str],
) -> None:
    """Raise if invariant violated — for tests and adversarial evals."""
    presented = {_norm(m) for m in presented_models if m}
    overlap = presented & eliminated
    if overlap:
        raise AssertionError(
            f"presented_models and eliminated_models must not overlap; overlap={sorted(overlap)}"
        )
