"""
HACK v2 — Ranking + Verdict Unification Engine.

This layer MUST run after HACK v1 (feasible set established) and MUST be the
single source of truth for:
- ranking order
- composite scores
- eligibility status
- verdict labels

Downstream renderers must only display what this layer computed; they must not
reorder or reinterpret scores/verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


class RankingIntegrityError(RuntimeError):
    """Raised when verdict/score/ranking integrity is violated."""


_VERDICT_GOOD = "GOOD FIT"
_VERDICT_CONDITIONAL = "CONDITIONAL FIT"
_VERDICT_NOT_FIT = "NOT A FIT"

_ELIGIBLE = "ELIGIBLE"
_NOT_ELIGIBLE = "NOT_ELIGIBLE"

_SEVERITY_LOW = "LOW"
_SEVERITY_MED = "MEDIUM"
_SEVERITY_HIGH = "HIGH"


@dataclass(frozen=True)
class AircraftEvaluation:
    aircraft_name: str
    mission_fit_score: float
    flexibility_score: float
    economics_score: float
    composite_score: float
    hard_flags: Dict[str, str]
    eligibility_status: str
    verdict: str

    def to_contract_row(self) -> Dict[str, Any]:
        return {
            "aircraft_name": self.aircraft_name,
            "composite_score": round(float(self.composite_score), 4),
            "eligibility_status": self.eligibility_status,
            "verdict": self.verdict,
        }


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _score_dimension(rec: Any, dim: str) -> float:
    try:
        for s in rec.scores or []:
            if getattr(s, "dimension", None) == dim:
                return float(getattr(s, "score", 0.0) or 0.0)
    except Exception:
        pass
    return 0.0


def _severity_from_score(score_01: float) -> str:
    x = float(score_01 or 0.0)
    if x < 0.45:
        return _SEVERITY_HIGH
    if x < 0.60:
        return _SEVERITY_MED
    return _SEVERITY_LOW


def _compute_hard_flags(
    rec: Any,
    *,
    packet_constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Derive hard flags + severity per aircraft.

    These flags are used ONLY to determine CONDITIONAL vs GOOD; physical
    feasibility is already enforced by HACK v1.
    """
    ic: Dict[str, Any] = packet_constraints or {}

    # Range / payload realism are already available from rec.scores and are
    # strongly correlated with suitability_score and composite_score.
    range_realism = _score_dimension(rec, "range_realism")
    passenger_count_fit = _score_dimension(rec, "passenger_count_fit")

    marginal_range = _severity_from_score(range_realism)
    payload_risk = _severity_from_score(passenger_count_fit)

    # Winter penalty only matters when mission-level westbound pressure exists.
    winter_pressure = bool(ic.get("westbound_winter_pressure") or ic.get("westbound_winter"))
    winter_margin = _score_dimension(rec, "winter_westbound_margin")
    winter_penalty = (
        _severity_from_score(winter_margin) if winter_pressure else _SEVERITY_LOW
    )

    # Runway risk only matters when short-runway / industrial field constraints exist.
    runway_risk_enabled = bool(
        ic.get("short_runway_likely")
        or ic.get("runway_over_cabin")
        or ic.get("industrial_airport_access")
        or ic.get("mountain_ops")
    )
    runway_perf = _score_dimension(rec, "runway_performance")
    runway_flex = _score_dimension(rec, "runway_flexibility")
    runway_constraint_raw = max(runway_perf, runway_flex)
    runway_constraint_risk = (
        _severity_from_score(1.0 - runway_constraint_raw)
        if runway_risk_enabled
        else _SEVERITY_LOW
    )

    return {
        "marginal_range": marginal_range,
        "payload_risk": payload_risk,
        "winter_penalty": winter_penalty,
        "runway_constraint_risk": runway_constraint_risk,
    }


def _compute_verdict_from_rules(
    *,
    eligibility_status: str,
    hard_flags: Dict[str, str],
    composite_score: float,
) -> str:
    if eligibility_status == _NOT_ELIGIBLE:
        return _VERDICT_NOT_FIT

    if any(sev == _SEVERITY_HIGH for sev in hard_flags.values()):
        return _VERDICT_CONDITIONAL

    if float(composite_score) >= 0.80:
        return _VERDICT_GOOD

    return _VERDICT_CONDITIONAL


def _verdict_to_fit_labels(verdict: str) -> Tuple[str, str]:
    """
    Map unified verdict into existing recommendation fields.

    - fit_verdict is the broker verdict label (used by renderers)
    - fit is the qualitative tier label used by some legacy paths
    """
    v = (verdict or "").strip().upper()
    if v == _VERDICT_NOT_FIT:
        return ("Not Recommended", _VERDICT_NOT_FIT)
    if v == _VERDICT_GOOD:
        return ("Good Fit", _VERDICT_GOOD)
    return ("Partial Fit", _VERDICT_CONDITIONAL)


def hack_v2_unify_rank_and_verdict(
    *,
    mission: Any,
    recommendations: Sequence[Any],
    packet: Optional[Any],
    query: str = "",
    data_used: Optional[Dict[str, Any]] = None,
    max_results: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Unify scoring model and verdict labels.

    Returns a contract-only list:
    RankedAircraftList[{aircraft_name, composite_score, eligibility_status, verdict}]
    """
    from services.recommendation.hack_v1_constraint_kernel import (
        hack_v1_permanent_exclusions,
        load_hack_v1_result,
    )

    du = data_used if isinstance(data_used, dict) else {}
    hack_v1 = load_hack_v1_result(du)
    if hack_v1 is None:
        # Defensive: if HACK v1 metadata is missing, treat as invalid state.
        raise RankingIntegrityError("RANKING_INTEGRITY_ERROR: HACK v1 metadata missing")

    allowed = set(hack_v1.feasible_aircraft_list or [])
    permanent_exclusions = hack_v1_permanent_exclusions(du)

    # Economics / utilization missions may recover non-ULR aircraft after the HACK v1
    # post-gate list is ULR-only — never when V2 router blocks fallback.
    try:
        from services.orchestration.orchestration_router_v2 import (
            orchestration_v2_blocks_tier_fallback,
        )
        from services.recommendation.tier_downgrade_recovery import (
            _ULR_MODELS,
            _economics_exclude_ulr,
        )

        if not orchestration_v2_blocks_tier_fallback(du) and _economics_exclude_ulr(query, du):
            non_ulr_allowed = {m for m in allowed if m not in _ULR_MODELS}
            if not non_ulr_allowed:
                rec_models = {
                    str(getattr(r, "model", "") or "")
                    for r in recommendations
                    if getattr(r, "model", None)
                }
                allowed = {m for m in rec_models if m and m not in permanent_exclusions}
    except Exception:
        pass

    if not allowed:
        return []

    pkt_constraints = getattr(packet, "inferred_constraints", None) or {}

    # --- Compute exactly one unified evaluation per HACK v1 feasible aircraft ---
    eval_rows: List[AircraftEvaluation] = []
    for rec in recommendations:
        model = str(getattr(rec, "model", "") or "")
        if not model or model not in allowed:
            continue
        if model in permanent_exclusions:
            continue

        eligibility_status = _ELIGIBLE

        mission_fit_score = _clamp01(getattr(rec, "suitability_score", None) or 0.0)
        flexibility_score = _clamp01(
            getattr(rec, "operational_flexibility_score", None) or 0.0
        )
        economics_score = _clamp01(getattr(rec, "economics_score", None) or 0.0)

        # Preserve ordering intent from the existing pipeline: multi-factor
        # already derives a mission_conflict_penalty, but the v2 composite
        # formula is penalty-free. We therefore fold the penalty into the
        # flexibility component so composite_score remains the only sort key.
        conflict_penalty = float(getattr(rec, "mission_conflict_penalty", None) or 0.0)
        if conflict_penalty > 0:
            flexibility_score = _clamp01(
                flexibility_score - (conflict_penalty * 0.10 / 0.30)
            )

        composite_score = _clamp01(
            0.5 * mission_fit_score + 0.3 * flexibility_score + 0.2 * economics_score
        )

        hard_flags = _compute_hard_flags(rec, packet_constraints=pkt_constraints)
        verdict = _compute_verdict_from_rules(
            eligibility_status=eligibility_status,
            hard_flags=hard_flags,
            composite_score=composite_score,
        )

        eval_rows.append(
            AircraftEvaluation(
                aircraft_name=model,
                mission_fit_score=mission_fit_score,
                flexibility_score=flexibility_score,
                economics_score=economics_score,
                composite_score=composite_score,
                hard_flags=hard_flags,
                eligibility_status=eligibility_status,
                verdict=verdict,
            )
        )

    if not eval_rows:
        return []

    # --- Strict ranking: sort by composite_score DESC only ---
    eval_rows.sort(
        key=lambda e: (-float(e.composite_score), str(e.aircraft_name).lower())
    )
    if max_results is not None:
        eval_rows = eval_rows[: int(max_results)]

    contract = [e.to_contract_row() for e in eval_rows]

    # --- Integrity checks (MANDATORY) ---
    # 1) No aircraft outside HACK v1 allowed set exists in the ranking output.
    for row in contract:
        if row["aircraft_name"] not in allowed:
            raise RankingIntegrityError(
                f"RANKING_INTEGRITY_ERROR: aircraft not in HACK v1 feasible set: {row['aircraft_name']}"
            )

    # 2) Verdict matches eligibility rules.
    for e in eval_rows:
        expected = _compute_verdict_from_rules(
            eligibility_status=e.eligibility_status,
            hard_flags=e.hard_flags,
            composite_score=e.composite_score,
        )
        if expected != e.verdict:
            raise RankingIntegrityError(
                f"RANKING_INTEGRITY_ERROR: verdict mismatch for {e.aircraft_name}"
            )

    # 3) Ranking order is strictly descending composite_score.
    for i in range(len(eval_rows) - 1):
        if float(eval_rows[i].composite_score) + 1e-9 < float(
            eval_rows[i + 1].composite_score
        ):
            raise RankingIntegrityError(
                "RANKING_INTEGRITY_ERROR: composite_score sort invariant violated"
            )

    # 4) Forbidden states.
    # NOT A FIT + ranked #1
    if eval_rows and eval_rows[0].verdict == _VERDICT_NOT_FIT:
        raise RankingIntegrityError(
            "RANKING_INTEGRITY_ERROR: NOT A FIT aircraft ranked #1"
        )

    # Conditional higher score than Good Fit but ranked lower
    # (pairwise compare by composite ordering)
    for i in range(len(eval_rows)):
        for j in range(i + 1, len(eval_rows)):
            if (
                eval_rows[i].verdict == _VERDICT_GOOD
                and eval_rows[j].verdict == _VERDICT_CONDITIONAL
                and eval_rows[i].composite_score + 1e-9 < eval_rows[j].composite_score
            ):
                raise RankingIntegrityError(
                    "RANKING_INTEGRITY_ERROR: GOOD FIT score below CONDITIONAL FIT score"
                )

            if (
                eval_rows[i].verdict == _VERDICT_CONDITIONAL
                and eval_rows[j].verdict == _VERDICT_GOOD
                and eval_rows[i].composite_score > eval_rows[j].composite_score + 1e-9
            ):
                # Because sorting is by composite score, this should never be reachable.
                raise RankingIntegrityError(
                    "RANKING_INTEGRITY_ERROR: CONDITIONAL FIT ranked below higher score GOOD FIT"
                )

    # --- Apply unified results back onto recommendation objects ---
    eval_by_model = {e.aircraft_name: e for e in eval_rows}
    # Preserve only the eval set order.
    ordered_models = [e.aircraft_name for e in eval_rows]

    for rec in recommendations:
        model = str(getattr(rec, "model", "") or "")
        if model not in eval_by_model:
            continue
        e = eval_by_model[model]
        rec.total_score = float(e.composite_score)
        rec.fit_verdict = e.verdict
        fit_label, fit_verdict = _verdict_to_fit_labels(e.verdict)
        rec.fit = fit_label
        rec.suitability_score = float(e.mission_fit_score)
        rec.operational_flexibility_score = float(e.flexibility_score)
        rec.economics_score = float(e.economics_score)

    return contract


__all__ = [
    "AircraftEvaluation",
    "RankingIntegrityError",
    "hack_v2_unify_rank_and_verdict",
]

