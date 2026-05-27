"""
Recommendation diversity controls — reduce default shortlist bias and improve mission fit.

- Repetition tracking across unrelated missions (session / conversation state)
- Aircraft overuse penalties (e.g. Challenger 350 / G280 / Praetor 600 triad)
- Hard undersized-platform rejection before ranking (ULR / long transatlantic)
- Route-specific dimension weighting
- Category diversity in final shortlist
- Audit logs with rejection reasons and internal scoring notes (not user-facing)
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation, RecommendationScore
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES
from services.mission.models import MissionProfile
from services.recommendation.mission_ranker import (
    MissionCategory,
    mission_max_leg_nm,
)

logger = logging.getLogger(__name__)

# Frequently over-recommended on missions where they are a poor primary fit
_DEFAULT_OVERUSED_MODELS = frozenset(
    {"Challenger 350", "Gulfstream G280", "Praetor 600"}
)

_SUPER_MIDSIZE_CATEGORIES = frozenset({"super-midsize", "midsize"})
_ULR_CATEGORIES = frozenset({"ultra-long", "large"})

_HISTORY_KEY = "recommendation_diversity_history"
_MAX_HISTORY_ENTRIES = 24
_REPETITION_PENALTY_PER_HIT = 0.04
_MAX_REPETITION_PENALTY = 0.14
_OVERUSE_BASE_PENALTY = 0.06

_CATEGORY_ORDER = ("light", "midsize", "super-midsize", "large", "ultra-long")


@dataclass
class RejectionRecord:
    model: str
    stage: str
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"model": self.model, "stage": self.stage, "reasons": list(self.reasons)}


@dataclass
class InternalScoringNote:
    """Internal-only ranking explanation — never shown verbatim to end users."""

    model: str
    base_total: float
    adjusted_total: float
    penalties: List[str] = field(default_factory=list)
    boosts: List[str] = field(default_factory=list)
    dimension_highlights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "base_total": round(self.base_total, 4),
            "adjusted_total": round(self.adjusted_total, 4),
            "penalties": list(self.penalties),
            "boosts": list(self.boosts),
            "dimension_highlights": dict(self.dimension_highlights),
        }


@dataclass
class RecommendationSelectionAudit:
    mission_fingerprint: str
    mission_category: str
    route_max_leg_nm: float
    feasible_before_rank: List[str] = field(default_factory=list)
    hard_rejected: List[RejectionRecord] = field(default_factory=list)
    ranked_models: List[str] = field(default_factory=list)
    scoring_notes: Dict[str, InternalScoringNote] = field(default_factory=dict)
    diversity_actions: List[str] = field(default_factory=list)
    ranking_transparency: List[Dict[str, Any]] = field(default_factory=list)
    repetition_justifications: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_fingerprint": self.mission_fingerprint,
            "mission_category": self.mission_category,
            "route_max_leg_nm": round(self.route_max_leg_nm, 1),
            "feasible_before_rank": list(self.feasible_before_rank),
            "hard_rejected": [r.to_dict() for r in self.hard_rejected],
            "ranked_models": list(self.ranked_models),
            "scoring_notes": {k: v.to_dict() for k, v in self.scoring_notes.items()},
            "diversity_actions": list(self.diversity_actions),
            "ranking_transparency": list(self.ranking_transparency),
            "repetition_justifications": dict(self.repetition_justifications),
        }


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _normalize_category(raw: str) -> str:
    return (raw or "").strip().lower().replace("_", "-")


def mission_fingerprint(mission: MissionState, *, mission_category: MissionCategory) -> str:
    """Stable key for unrelated-mission repetition detection."""
    routes = "|".join(sorted((r or "").strip().lower() for r in (mission.routes or [])))
    pax = mission.passenger_count or 0
    flags = (
        int(bool(mission.nonstop_requirement)),
        int(bool(mission.westbound)),
        int(bool(mission.mountain_airport_requirement)),
        (mission.seasonal_constraints or "")[:32].lower(),
        mission_category.value,
    )
    blob = f"{routes}::{pax}::{flags}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_recommendation_history(
    conversation_state: Optional[Dict[str, Any]],
    data_used: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    for src in (data_used, conversation_state):
        if isinstance(src, dict):
            hist = src.get(_HISTORY_KEY)
            if isinstance(hist, list):
                return list(hist)[-_MAX_HISTORY_ENTRIES:]
    return []


def record_recommendation_history(
    *,
    fingerprint: str,
    ranked_models: Sequence[str],
    conversation_state: Optional[Dict[str, Any]] = None,
    data_used: Optional[Dict[str, Any]] = None,
) -> None:
    entry = {
        "fingerprint": fingerprint,
        "models": list(ranked_models)[:5],
    }
    for target in (data_used, conversation_state):
        if not isinstance(target, dict):
            continue
        hist = target.get(_HISTORY_KEY)
        if not isinstance(hist, list):
            hist = []
        hist = [e for e in hist if isinstance(e, dict)][-_MAX_HISTORY_ENTRIES + 1 :]
        hist.append(entry)
        target[_HISTORY_KEY] = hist[-_MAX_HISTORY_ENTRIES:]


def repetition_penalty_for_model(
    model: str,
    fingerprint: str,
    history: Sequence[Dict[str, Any]],
) -> Tuple[float, str]:
    """Penalize repeating the same aircraft on a different mission fingerprint."""
    hits = 0
    for entry in history:
        if not isinstance(entry, dict):
            continue
        if entry.get("fingerprint") == fingerprint:
            continue
        if model in (entry.get("models") or []):
            hits += 1
    if hits == 0:
        return 0.0, ""
    penalty = min(_MAX_REPETITION_PENALTY, hits * _REPETITION_PENALTY_PER_HIT)
    return penalty, f"Recently recommended on {hits} unrelated mission(s) — diversity penalty applied."


def overuse_penalty_for_model(
    model: str,
    mission_category: MissionCategory,
    max_leg_nm: float,
) -> Tuple[float, str]:
    """Baseline penalty for catalog types that dominate generic shortlists."""
    if model not in _DEFAULT_OVERUSED_MODELS:
        return 0.0, ""

    cat = _normalize_category((AIRCRAFT_PROFILES.get(model) or {}).get("category", ""))

    if mission_category == MissionCategory.ULTRA_LONG_RANGE and cat in _SUPER_MIDSIZE_CATEGORIES:
        return 0.35, f"{model} is a super-midsize default — inappropriate primary fit for ULR missions."

    if mission_category == MissionCategory.TRANSATLANTIC_EXECUTIVE and max_leg_nm >= 3200:
        if cat in _SUPER_MIDSIZE_CATEGORIES:
            return 0.28, f"{model} lacks consistent transatlantic margin at stated stage length."

    if mission_category in (
        MissionCategory.REGIONAL_UTILITY,
        MissionCategory.MOUNTAIN_AIRPORT,
    ):
        if max_leg_nm < 1500 and cat in _SUPER_MIDSIZE_CATEGORIES:
            return 0.08, ""

    if mission_category == MissionCategory.COAST_TO_COAST and max_leg_nm < 2200:
        return _OVERUSE_BASE_PENALTY, f"{model} is a common default — verify it beats peers on economics for this leg."

    return _OVERUSE_BASE_PENALTY, f"{model} overuse guard — deprioritized vs mission-specific alternatives."


def undersized_platform_hard_reject(
    model: str,
    profile: Dict[str, Any],
    mission_category: MissionCategory,
    max_leg_nm: float,
    *,
    required_peak_nm: float = 0.0,
) -> Optional[str]:
    """
    Hard reject before scoring — impossible / unrealistic platform for mission class.

    Returns rejection reason or None if allowed to rank.
    """
    cat = _normalize_category(str(profile.get("category") or ""))
    practical = float(profile.get("practical_nm") or 0)

    if mission_category == MissionCategory.ULTRA_LONG_RANGE:
        if cat in _SUPER_MIDSIZE_CATEGORIES and max_leg_nm >= 4000:
            return (
                f"{model} (super-midsize) cannot satisfy ULR mission "
                f"(~{int(max_leg_nm)} nm leg) — excluded before ranking."
            )
        if cat == "light":
            return f"{model} (light jet) excluded from ULR mission shortlist."

    if mission_category == MissionCategory.TRANSATLANTIC_EXECUTIVE and max_leg_nm >= 3000:
        if cat in ("light", "midsize"):
            return f"{model} lacks transatlantic range class for ~{int(max_leg_nm)} nm leg."
        if cat == "super-midsize" and required_peak_nm > 0 and practical < required_peak_nm * 1.05:
            return (
                f"{model} practical range (~{int(practical)} nm) insufficient for "
                f"~{int(required_peak_nm)} nm required with reserves."
            )

    if mission_category == MissionCategory.MOUNTAIN_AIRPORT:
        if cat in ("ultra-long",) and max_leg_nm < 1800:
            return f"{model} (ULR) excluded — runway/field performance poor for mountain mission."

    if mission_category in (
        MissionCategory.REGIONAL_UTILITY,
        MissionCategory.MOUNTAIN_AIRPORT,
    ):
        if max_leg_nm < 1800 and cat in ("ultra-long", "large"):
            return (
                f"{model} ({cat}) excluded — oversized for short regional / runway-focused mission."
            )
        if max_leg_nm < 1200 and cat == "super-midsize" and required_peak_nm > 0:
            if practical < required_peak_nm * 1.15:
                return f"{model} — tight range margin on short leg with runway constraints."

    if max_leg_nm < 2500 and cat == "ultra-long":
        if mission_category not in (
            MissionCategory.ULTRA_LONG_RANGE,
            MissionCategory.TRANSATLANTIC_EXECUTIVE,
        ):
            return f"{model} (ULR) excluded — stage length does not justify ultra-long-range class."

    return None


def route_specific_score_multiplier(
    mission: MissionState,
    mission_category: MissionCategory,
    profile: Dict[str, Any],
    scores: Sequence[RecommendationScore],
) -> Tuple[float, List[str]]:
    """
  Apply route-aware boosts to total score via multiplier (ownership, runway, range).
    """
    mult = 1.0
    boosts: List[str] = []
    max_leg = mission_max_leg_nm(mission)
    cat = _normalize_category(str(profile.get("category") or ""))

    dim = {s.dimension: s.score for s in scores}

    if mission_category == MissionCategory.MOUNTAIN_AIRPORT or mission.mountain_airport_requirement:
        rw = dim.get("runway_performance", 0.5)
        if rw >= 0.75:
            mult += 0.06
            boosts.append("Runway/field performance weighted for mountain mission.")
        elif cat in ("ultra-long", "large") and max_leg < 2000:
            mult -= 0.08

    if mission_category in (
        MissionCategory.ULTRA_LONG_RANGE,
        MissionCategory.TRANSATLANTIC_EXECUTIVE,
    ):
        rr = dim.get("route_realism", 0.5)
        wb = dim.get("winter_westbound_margin", 0.5)
        if cat in _ULR_CATEGORIES and rr >= 0.7:
            mult += 0.08
            boosts.append("Route realism boost for long-range mission class.")
        if mission.westbound and wb >= 0.75 and cat in _ULR_CATEGORIES:
            mult += 0.05
            boosts.append("Westbound margin boost applied.")

    if mission.budget_usd or (mission.acquisition_strategy or "").lower() in (
        "fractional",
        "ownership",
        "purchase",
    ):
        own = dim.get("ownership_economics", 0.5)
        if own >= 0.72:
            mult += 0.05
            boosts.append("Ownership economics weighted for acquisition-focused query.")
        elif own < 0.5 and mission.budget_usd:
            mult -= 0.06

    if mission_category == MissionCategory.REGIONAL_UTILITY:
        op = dim.get("operating_economics", 0.5)
        ob = dim.get("overbuying_penalty", 0.5)
        if op >= 0.75:
            mult += 0.04
        if ob >= 0.85 and cat in _SUPER_MIDSIZE_CATEGORIES:
            mult -= 0.03
        if (mission.operating_cost_priority or "") == "high" and cat in ("large", "ultra-long"):
            mult -= 0.12

    if (mission.operating_cost_priority or "") == "high" and cat in ("large", "ultra-long"):
        if mission_category == MissionCategory.COAST_TO_COAST and max_leg < 2400:
            mult -= 0.1

    return _clamp01(mult), boosts


def apply_hard_feasibility_gate(
    models: Sequence[str],
    mission: MissionState,
    mission_profile: Optional[MissionProfile],
    mission_category: MissionCategory,
    *,
    required_peak_nm: float = 0.0,
    skip_feasibility_engine: bool = False,
) -> Tuple[List[str], List[RejectionRecord]]:
    """
    Second-pass hard gate before ranking (undersized platform + optional aircraft_feasibility engine).

    When ``skip_feasibility_engine`` is True (pipeline already filtered), only undersized-platform
    rejects run — avoids double-eliminating ULR types that passed the advisory hard gate.
    """
    survivors: List[str] = []
    rejected: List[RejectionRecord] = []

    try:
        from services.aircraft_feasibility import evaluate_aircraft_feasibility

        use_engine = True
    except ImportError:
        use_engine = False

    for model in models:
        spec = AIRCRAFT_PROFILES.get(model)
        if not spec:
            rejected.append(
                RejectionRecord(model, "hard_gate", ["Unknown aircraft — not in catalog."])
            )
            continue

        reason = undersized_platform_hard_reject(
            model,
            spec,
            mission_category,
            mission_max_leg_nm(mission),
            required_peak_nm=required_peak_nm,
        )
        if reason:
            rejected.append(RejectionRecord(model, "undersized_platform", [reason]))
            logger.info("DIVERSITY_HARD_REJECT: model=%s reason=%s", model, reason)
            continue

        if use_engine and mission_profile is not None and not skip_feasibility_engine:
            try:
                verdict = evaluate_aircraft_feasibility(mission_profile, model)
                if not verdict.feasible:
                    reasons = list(verdict.rejection_reasons or ["feasibility_engine_reject"])
                    rejected.append(
                        RejectionRecord(model, "aircraft_feasibility_engine", reasons[:4])
                    )
                    logger.info(
                        "FEASIBILITY_ENGINE_REJECT: model=%s reasons=%s",
                        model,
                        "; ".join(reasons[:2]),
                    )
                    continue
            except Exception:
                pass

        survivors.append(model)

    return survivors, rejected


def _category_diversity_reorder(
    scored: List[AircraftRecommendation],
    mission_category: MissionCategory,
    max_results: int,
) -> List[AircraftRecommendation]:
    """
    Prefer a shortlist that is not three identical cabin classes when scores are close.
    """
    if len(scored) <= 1 or max_results <= 1:
        return scored

    if mission_category not in (
        MissionCategory.ULTRA_LONG_RANGE,
        MissionCategory.TRANSATLANTIC_EXECUTIVE,
        MissionCategory.GENERAL_ADVISORY,
    ):
        return scored

    selected: List[AircraftRecommendation] = []
    used_categories: Set[str] = set()
    pool = list(scored)

    def _cat(rec: AircraftRecommendation) -> str:
        return _normalize_category(rec.category)

    # First pass: best score per distinct category
    for rec in pool:
        if len(selected) >= max_results:
            break
        c = _cat(rec)
        if c in used_categories and len(used_categories) >= 2:
            continue
        selected.append(rec)
        used_categories.add(c)

    # Fill remaining slots by score
    for rec in pool:
        if len(selected) >= max_results:
            break
        if rec in selected:
            continue
        selected.append(rec)

    return selected[:max_results]


def apply_diversity_controls(
    scored: List[AircraftRecommendation],
    mission: MissionState,
    *,
    mission_category: MissionCategory,
    mission_profile: Optional[MissionProfile] = None,
    max_results: int = 3,
    conversation_state: Optional[Dict[str, Any]] = None,
    data_used: Optional[Dict[str, Any]] = None,
    hard_rejected: Optional[List[RejectionRecord]] = None,
) -> Tuple[List[AircraftRecommendation], RecommendationSelectionAudit]:
    """
    Adjust scores, re-sort, apply category diversity, and build audit trail.
    """
    fingerprint = mission_fingerprint(mission, mission_category=mission_category)
    max_leg = mission_max_leg_nm(mission)
    history = load_recommendation_history(conversation_state, data_used)

    audit = RecommendationSelectionAudit(
        mission_fingerprint=fingerprint,
        mission_category=mission_category.value,
        route_max_leg_nm=max_leg,
        feasible_before_rank=[r.model for r in scored],
        hard_rejected=list(hard_rejected or []),
    )

    adjusted: List[AircraftRecommendation] = []

    for rec in scored:
        if rec.avoid:
            continue

        base = rec.total_score
        penalty_total = 0.0
        penalties: List[str] = []
        boosts: List[str] = []

        rep_pen, rep_note = repetition_penalty_for_model(rec.model, fingerprint, history)
        if rep_pen > 0 and rep_note:
            penalty_total += rep_pen
            penalties.append(rep_note)

        ou_pen, ou_note = overuse_penalty_for_model(rec.model, mission_category, max_leg)
        if ou_pen > 0:
            penalty_total += ou_pen
            if ou_note:
                penalties.append(ou_note)

        route_mult, route_boosts = route_specific_score_multiplier(
            mission, mission_category, AIRCRAFT_PROFILES.get(rec.model) or {}, rec.scores
        )
        boosts.extend(route_boosts)

        adjusted_total = _clamp01((base - penalty_total) * route_mult)

        dim_highlights = {
            s.dimension: round(s.score, 3)
            for s in rec.scores
            if s.dimension
            in (
                "route_realism",
                "runway_performance",
                "ownership_economics",
                "overbuying_penalty",
                "operating_economics",
            )
        }

        audit.scoring_notes[rec.model] = InternalScoringNote(
            model=rec.model,
            base_total=base,
            adjusted_total=adjusted_total,
            penalties=penalties,
            boosts=boosts,
            dimension_highlights=dim_highlights,
        )

        rec.total_score = round(adjusted_total, 4)
        if penalties and rec.explanation:
            rec.explanation.penalties = (rec.explanation.penalties or []) + penalties[:2]
        adjusted.append(rec)

    from services.recommendation.aircraft_archetype_weighting import modern_operational_fit_score

    def _sort_key(r: AircraftRecommendation) -> tuple:
        mod = modern_operational_fit_score(r.model)
        for s in r.scores:
            if s.dimension == "modern_operational_fit":
                mod = s.score
                break
        return (-r.total_score, -mod)

    adjusted.sort(key=_sort_key)

    if mission_category == MissionCategory.ULTRA_LONG_RANGE:
        ulr_survivors = [
            r
            for r in adjusted
            if _normalize_category(r.category) in _ULR_CATEGORIES
            or r.total_score >= 0.72
        ]
        if ulr_survivors:
            demoted = [r for r in adjusted if r not in ulr_survivors[: max_results * 2]]
            if demoted:
                audit.diversity_actions.append(
                    "ULR mission — super-midsize defaults deprioritized below large/ULR types."
                )
            adjusted = ulr_survivors + [r for r in adjusted if r not in ulr_survivors]

    diversified = _category_diversity_reorder(adjusted, mission_category, max_results)

    from services.recommendation.recommendation_diversity_guard import (
        apply_recommendation_diversity_guard,
    )

    guard_result = apply_recommendation_diversity_guard(
        diversified,
        mission,
        mission_category=mission_category,
        fingerprint=fingerprint,
        history=history,
        max_results=max_results,
        audit=audit,
    )
    final = guard_result.recommendations
    audit.ranking_transparency = [r.to_dict() for r in guard_result.transparency]
    audit.repetition_justifications = dict(guard_result.repetition_justifications)
    if guard_result.actions:
        audit.diversity_actions.extend(guard_result.actions)

    if diversified != final:
        audit.diversity_actions.append(
            "Diversity guard — default triad / repetition rules adjusted final shortlist."
        )

    audit.ranked_models = [r.model for r in final]

    record_recommendation_history(
        fingerprint=fingerprint,
        ranked_models=audit.ranked_models,
        conversation_state=conversation_state,
        data_used=data_used,
    )

    if audit.diversity_actions or audit.hard_rejected:
        logger.info(
            "RECOMMENDATION_AUDIT fingerprint=%s category=%s ranked=%s actions=%s rejected=%d",
            fingerprint,
            mission_category.value,
            audit.ranked_models,
            audit.diversity_actions,
            len(audit.hard_rejected),
        )

    return final, audit


def merge_elimination_log_with_audit(
    elimination_log: List[Dict[str, Any]],
    audit: Optional[RecommendationSelectionAudit],
) -> List[Dict[str, Any]]:
    """Append hard rejects from diversity gate to pipeline elimination log."""
    out = list(elimination_log or [])
    if audit is None:
        return out
    for rec in audit.hard_rejected:
        out.append(
            {
                "aircraft_name": rec.model,
                "reason": "; ".join(rec.reasons),
                "stage": rec.stage,
                "mission_constraint_failed": "platform_class_mismatch",
            }
        )
    return out
