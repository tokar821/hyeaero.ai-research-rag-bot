"""
Recommendation diversity guard — prevent default-triad bias and stale repeats.

Rules:
  - Challenger 350, Praetor 600, Gulfstream G280 only appear when they genuinely lead scoring.
  - Repeated recommendations on unrelated missions require mission justification and
    are compared against mission-specific alternatives.
  - Full ranking transparency is emitted on every shortlist (internal audit only).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import AircraftRecommendation
from services.mission.aircraft_profiles import AIRCRAFT_PROFILES
from services.recommendation.diversity_controls import (
    RecommendationSelectionAudit,
    load_recommendation_history,
    repetition_penalty_for_model,
)
from services.recommendation.mission_ranker import MissionCategory

logger = logging.getLogger(__name__)

DEFAULT_TRIAD_MODELS: frozenset[str] = frozenset(
    {"Challenger 350", "Gulfstream G280", "Praetor 600"}
)

# Must be the scored leader or within this margin of leader (after all penalties)
_GENUINE_LEAD_EPSILON = 0.008
# Extra penalty when a triad model is not the genuine top scorer
_TRIAD_NON_LEADER_PENALTY = 0.14
# Stronger penalty when triad would occupy all three slots without justification
_FULL_TRIAD_BLOCK_PENALTY = 0.22
_REPETITION_JUSTIFICATION_PENALTY = 0.10
_MAX_REPETITION_HITS_BEFORE_BLOCK = 2


@dataclass
class RankingTransparencyRow:
    """Internal ranking row — never shown verbatim to end users."""

    rank: int
    model: str
    base_score: float
    adjusted_score: float
    triad_member: bool
    triad_guard: str  # allowed | demoted | blocked | not_triad
    repetition_hits: int
    mission_justification_required: bool
    vs_leader_delta: float
    mission_specific_alternative: Optional[str] = None
    guard_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "model": self.model,
            "base_score": round(self.base_score, 4),
            "adjusted_score": round(self.adjusted_score, 4),
            "triad_member": self.triad_member,
            "triad_guard": self.triad_guard,
            "repetition_hits": self.repetition_hits,
            "mission_justification_required": self.mission_justification_required,
            "vs_leader_delta": round(self.vs_leader_delta, 4),
            "mission_specific_alternative": self.mission_specific_alternative,
            "guard_reason": self.guard_reason,
        }


@dataclass
class DiversityGuardResult:
    recommendations: List[AircraftRecommendation]
    transparency: List[RankingTransparencyRow] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    repetition_justifications: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": list(self.actions),
            "ranking_transparency": [r.to_dict() for r in self.transparency],
            "repetition_justifications": dict(self.repetition_justifications),
        }


def _repetition_hits(
    model: str,
    fingerprint: str,
    history: Sequence[Dict[str, Any]],
) -> int:
    hits = 0
    for entry in history:
        if not isinstance(entry, dict):
            continue
        if entry.get("fingerprint") == fingerprint:
            continue
        if model in (entry.get("models") or []):
            hits += 1
    return hits


def _best_mission_specific_alternative(
    model: str,
    pool: Sequence[AircraftRecommendation],
) -> Optional[AircraftRecommendation]:
    """Highest-scoring non-triad peer that beats ``model``."""
    rec = next((r for r in pool if r.model == model), None)
    if rec is None:
        return None
    candidates = [
        r
        for r in pool
        if r.model not in DEFAULT_TRIAD_MODELS
        and not r.avoid
        and r.total_score > rec.total_score + _GENUINE_LEAD_EPSILON
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.total_score)


def genuinely_scores_highest(
    model: str,
    pool: Sequence[AircraftRecommendation],
) -> Tuple[bool, str]:
    """
    True only when ``model`` is the adjusted-score leader or effectively tied.

    Default triad types must not appear in the shortlist on inertia alone.
    """
    if model not in DEFAULT_TRIAD_MODELS:
        return True, "not_default_triad"

    viable = [r for r in pool if not r.avoid]
    if not viable:
        return False, "empty_pool"

    leader = max(viable, key=lambda r: r.total_score)
    rec = next((r for r in viable if r.model == model), None)
    if rec is None:
        return False, "not_in_pool"

    delta = leader.total_score - rec.total_score
    if rec.model == leader.model:
        return True, "genuine_leader"

    if delta <= _GENUINE_LEAD_EPSILON:
        return True, "tied_with_leader"

    alt = _best_mission_specific_alternative(model, viable)
    if alt is None:
        return True, "no_higher_non_triad_peer"

    return False, f"outscored_by_{alt.model}_(Δ={delta:.3f})"


def apply_repetition_and_justification(
    rec: AircraftRecommendation,
    *,
    fingerprint: str,
    history: Sequence[Dict[str, Any]],
    pool: Sequence[AircraftRecommendation],
) -> Tuple[float, str, bool]:
    """
    Returns (extra_penalty, audit_note, mission_justification_required).
    """
    hits = _repetition_hits(rec.model, fingerprint, history)
    if hits == 0:
        return 0.0, "", False

    pen, note = repetition_penalty_for_model(rec.model, fingerprint, history)
    justification_required = hits >= 1

    alt = _best_mission_specific_alternative(rec.model, pool)
    if alt is not None:
        note = (
            f"{note} Compare against mission-specific alternative {alt.model} "
            f"(adjusted {alt.total_score:.3f} vs {rec.total_score:.3f})."
        ).strip()
    elif note:
        note = f"{note} Mission-specific alternatives should be weighed before repeating this type."

    extra = pen
    if hits >= _MAX_REPETITION_HITS_BEFORE_BLOCK:
        extra += _REPETITION_JUSTIFICATION_PENALTY
        justification_required = True

    return extra, note, justification_required


def _apply_triad_non_leader_penalties(
    pool: List[AircraftRecommendation],
) -> List[str]:
    """Demote triad models that do not genuinely lead the pool."""
    actions: List[str] = []
    viable = [r for r in pool if not r.avoid]
    if not viable:
        return actions

    leader_score = max(r.total_score for r in viable)

    for rec in pool:
        if rec.avoid or rec.model not in DEFAULT_TRIAD_MODELS:
            continue
        ok, reason = genuinely_scores_highest(rec.model, pool)
        if ok:
            continue
        alt = _best_mission_specific_alternative(rec.model, pool)
        rec.total_score = max(0.0, rec.total_score - _TRIAD_NON_LEADER_PENALTY)
        msg = (
            f"{rec.model} demoted — default triad type did not genuinely score highest ({reason})."
        )
        actions.append(msg)
        if rec.explanation:
            rec.explanation.penalties = (rec.explanation.penalties or []) + [
                msg,
            ]
            if alt:
                rec.explanation.operational_compromises = (
                    rec.explanation.operational_compromises or []
                ) + [
                    f"Mission-specific alternative {alt.model} outscores this default shortlist type."
                ]
        logger.info(
            "TRIAD_GUARD_DEMOTE: model=%s leader=%.3f self=%.3f reason=%s alt=%s",
            rec.model,
            leader_score,
            rec.total_score,
            reason,
            alt.model if alt else None,
        )

    return actions


def _rebuild_shortlist_without_unjustified_triad(
    pool: List[AircraftRecommendation],
    max_results: int,
) -> Tuple[List[AircraftRecommendation], List[str]]:
    """
    Build final shortlist: triad slots only if genuinely highest; never all three triad.
    """
    actions: List[str] = []
    sorted_pool = sorted(
        [r for r in pool if not r.avoid],
        key=lambda r: -r.total_score,
    )
    if not sorted_pool:
        return [], actions

    selected: List[AircraftRecommendation] = []
    triad_in_selected: List[str] = []

    for rec in sorted_pool:
        if len(selected) >= max_results:
            break
        if rec.model in DEFAULT_TRIAD_MODELS:
            ok, reason = genuinely_scores_highest(rec.model, sorted_pool)
            if not ok:
                actions.append(
                    f"Blocked {rec.model} from shortlist — not genuine top scorer ({reason})."
                )
                continue
        selected.append(rec)
        if rec.model in DEFAULT_TRIAD_MODELS:
            triad_in_selected.append(rec.model)

    if len(triad_in_selected) == 3 and len(selected) >= 3:
        weakest = min(selected, key=lambda r: r.total_score)
        if weakest.model in DEFAULT_TRIAD_MODELS:
            selected.remove(weakest)
            weakest.total_score = max(0.0, weakest.total_score - _FULL_TRIAD_BLOCK_PENALTY)
            actions.append(
                "Full default triad blocked — replaced lowest triad slot with mission-specific alternative."
            )
            for rec in sorted_pool:
                if rec in selected or rec.avoid:
                    continue
                if rec.model not in DEFAULT_TRIAD_MODELS:
                    selected.append(rec)
                    break

    while len(selected) < max_results:
        for rec in sorted_pool:
            if rec in selected or rec.avoid:
                continue
            if rec.model in DEFAULT_TRIAD_MODELS:
                ok, _ = genuinely_scores_highest(rec.model, sorted_pool)
                if not ok:
                    continue
            selected.append(rec)
            if len(selected) >= max_results:
                break
        else:
            break

    return selected[:max_results], actions


def build_ranking_transparency(
    pool: Sequence[AircraftRecommendation],
    final: Sequence[AircraftRecommendation],
    *,
    fingerprint: str,
    history: Sequence[Dict[str, Any]],
    guard_actions: Sequence[str],
    scoring_notes: Optional[Dict[str, Any]] = None,
) -> List[RankingTransparencyRow]:
    """Full internal ranking table for audit and pipeline transparency."""
    viable = sorted([r for r in pool if not r.avoid], key=lambda r: -r.total_score)
    if not viable:
        return []

    leader_score = viable[0].total_score
    final_models = {r.model for r in final}
    rows: List[RankingTransparencyRow] = []

    for idx, rec in enumerate(viable, start=1):
        hits = _repetition_hits(rec.model, fingerprint, history)
        is_triad = rec.model in DEFAULT_TRIAD_MODELS
        ok, reason = genuinely_scores_highest(rec.model, viable) if is_triad else (True, "not_triad")
        alt = _best_mission_specific_alternative(rec.model, viable) if is_triad else None

        if rec.model not in final_models and is_triad and not ok:
            triad_guard = "blocked"
        elif rec.model in final_models:
            triad_guard = "allowed" if (not is_triad or ok) else "demoted"
        elif is_triad:
            triad_guard = "demoted"
        else:
            triad_guard = "not_triad"

        note = reason if is_triad else ""
        if hits > 0 and rec.model in final_models:
            note = (note + f" repetition_hits={hits}").strip()

        note_obj = (scoring_notes or {}).get(rec.model) if scoring_notes else None
        if hasattr(note_obj, "base_total"):
            base = float(note_obj.base_total)
            adjusted = float(note_obj.adjusted_total)
        elif isinstance(note_obj, dict):
            base = float(note_obj.get("base_total") or rec.total_score)
            adjusted = float(note_obj.get("adjusted_total") or rec.total_score)
        else:
            base = rec.total_score
            adjusted = rec.total_score

        if rec.model in final_models:
            for action in guard_actions:
                if rec.model in action:
                    note = (note + " " + action).strip()
                    break

        rows.append(
            RankingTransparencyRow(
                rank=idx,
                model=rec.model,
                base_score=base,
                adjusted_score=adjusted,
                triad_member=is_triad,
                triad_guard=triad_guard,
                repetition_hits=hits,
                mission_justification_required=hits >= 1 and rec.model in final_models,
                vs_leader_delta=round(leader_score - rec.total_score, 4),
                mission_specific_alternative=alt.model if alt else None,
                guard_reason=note,
            )
        )

    return rows


def apply_recommendation_diversity_guard(
    scored: List[AircraftRecommendation],
    mission: MissionState,
    *,
    mission_category: MissionCategory,
    fingerprint: str,
    history: Optional[Sequence[Dict[str, Any]]] = None,
    max_results: int = 3,
    audit: Optional[RecommendationSelectionAudit] = None,
) -> DiversityGuardResult:
    """
    Final diversity guard pass after score adjustments.

    Enforces triad genuine-lead rule, repetition justification, and transparency.
    """
    del mission, mission_category  # reserved for future mission-shape rules
    hist = list(history or [])
    pool = list(scored)
    actions: List[str] = []
    repetition_justifications: Dict[str, str] = {}

    for rec in pool:
        if rec.avoid:
            continue
        extra, note, need_just = apply_repetition_and_justification(
            rec,
            fingerprint=fingerprint,
            history=hist,
            pool=pool,
        )
        if extra > 0:
            rec.total_score = max(0.0, rec.total_score - extra)
            actions.append(f"Repetition guard: {rec.model} (-{extra:.3f})")
            if note:
                repetition_justifications[rec.model] = note
            if need_just and rec.explanation:
                rec.explanation.penalties = (rec.explanation.penalties or []) + [
                    "Repeated on a prior unrelated mission — mission justification required "
                    "before recommending again; compare mission-specific alternatives.",
                ]

    actions.extend(_apply_triad_non_leader_penalties(pool))

    pool.sort(key=lambda r: (-r.total_score, r.model))
    final, rebuild_actions = _rebuild_shortlist_without_unjustified_triad(pool, max_results)
    actions.extend(rebuild_actions)

    scoring_notes = audit.scoring_notes if audit is not None else None
    transparency = build_ranking_transparency(
        pool,
        final,
        fingerprint=fingerprint,
        history=hist,
        guard_actions=actions,
        scoring_notes=scoring_notes,
    )

    logger.info(
        "DIVERSITY_GUARD final=%s triad_blocked=%d transparency_rows=%d",
        [r.model for r in final],
        sum(1 for a in actions if "Blocked" in a or "demoted" in a.lower()),
        len(transparency),
    )

    return DiversityGuardResult(
        recommendations=final,
        transparency=transparency,
        actions=actions,
        repetition_justifications=repetition_justifications,
    )


__all__ = [
    "DEFAULT_TRIAD_MODELS",
    "DiversityGuardResult",
    "RankingTransparencyRow",
    "apply_recommendation_diversity_guard",
    "build_ranking_transparency",
    "genuinely_scores_highest",
]
