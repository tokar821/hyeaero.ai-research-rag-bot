"""
Executive Intelligence Synthesis Engine — Phase 27.

Consumes prior-phase ``data_used`` outputs only. Does not alter routing or execution.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_SYNTH_ENV = "ENABLE_EXECUTIVE_SYNTHESIS"

_RANK_WEIGHTS = {
    "ownership": 0.25,
    "market": 0.20,
    "mission": 0.20,
    "optimization": 0.20,
    "confidence": 0.15,
}


@dataclass
class ExecutiveIntelligenceSynthesis:
    fleet_summary: Dict[str, Any]
    portfolio_ranking: List[Dict[str, Any]]
    strategic_actions: Dict[str, List[str]]
    insights: List[str]
    synthesis_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fleet_summary": dict(self.fleet_summary),
            "portfolio_ranking": [dict(r) for r in self.portfolio_ranking],
            "strategic_actions": {k: list(v) for k, v in self.strategic_actions.items()},
            "insights": list(self.insights),
            "synthesis_id": self.synthesis_id,
        }


def executive_synthesis_enabled() -> bool:
    return (os.getenv(_SYNTH_ENV) or "").strip().lower() in ("1", "true", "yes")


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(value)))


def _liquidity_label(score: float) -> str:
    if score >= 70:
        return "strong"
    if score >= 40:
        return "moderate"
    return "weak"


def _collect_aircraft(data_used: Dict[str, Any]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []

    def _add(name: str) -> None:
        n = str(name or "").strip()
        if n and n not in seen:
            seen.add(n)
            ordered.append(n)

    fleet = data_used.get("fleet_portfolio_strategy")
    if isinstance(fleet, dict):
        for ac in fleet.get("current_aircraft") or []:
            _add(str(ac))

    own = data_used.get("ownership_intelligence")
    if isinstance(own, dict):
        for row in own.get("ownership_reports") or []:
            if isinstance(row, dict):
                _add(str(row.get("aircraft") or ""))

    opt = data_used.get("optimization_result")
    if isinstance(opt, dict):
        for row in opt.get("ranked_candidates") or []:
            if isinstance(row, dict):
                _add(str(row.get("aircraft") or ""))

    conf = data_used.get("recommendation_confidence")
    if isinstance(conf, dict):
        for row in conf.get("aircraft_confidence") or []:
            if isinstance(row, dict):
                _add(str(row.get("aircraft") or ""))

    just = data_used.get("recommendation_justification")
    if isinstance(just, dict):
        for row in just.get("recommendations") or []:
            if isinstance(row, dict):
                _add(str(row.get("aircraft") or ""))

    market = data_used.get("market_intelligence")
    if isinstance(market, dict):
        _add(str(market.get("aircraft") or ""))

    return ordered


def _index_by_aircraft(rows: Sequence[Any], key: str = "aircraft") -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict):
            name = str(row.get(key) or "").strip()
            if name:
                out[name] = row
    return out


def _ownership_scores(data_used: Dict[str, Any]) -> Dict[str, float]:
    own = data_used.get("ownership_intelligence")
    if not isinstance(own, dict):
        return {}
    scores: Dict[str, float] = {}
    for row in own.get("ownership_reports") or []:
        if isinstance(row, dict) and row.get("aircraft"):
            scores[str(row["aircraft"])] = _clamp(float(row.get("lifecycle_score") or 50.0))
    return scores


def _market_scores(data_used: Dict[str, Any], aircraft: Sequence[str]) -> Dict[str, float]:
    market = data_used.get("market_intelligence")
    base = 50.0
    if isinstance(market, dict):
        base = _clamp(float(market.get("market_strength_score") or 50.0))
        liq = float(market.get("liquidity_score") or 50.0)
        base = _clamp(base * 0.65 + liq * 0.35)
    return {ac: base for ac in aircraft}


def _mission_scores(data_used: Dict[str, Any], aircraft: Sequence[str]) -> Dict[str, float]:
    scores: Dict[str, float] = {ac: 50.0 for ac in aircraft}

    just = data_used.get("recommendation_justification")
    if isinstance(just, dict):
        for row in just.get("recommendations") or []:
            if isinstance(row, dict) and row.get("aircraft"):
                scores[str(row["aircraft"])] = _clamp(
                    float(row.get("mission_alignment_score") or 50.0)
                )

    fleet = data_used.get("fleet_portfolio_strategy")
    if isinstance(fleet, dict):
        coverage = fleet.get("mission_coverage_map") or {}
        if isinstance(coverage, dict) and coverage:
            fleet_avg = sum(float(v) for v in coverage.values()) / len(coverage)
            for ac in aircraft:
                if scores.get(ac, 50.0) == 50.0:
                    scores[ac] = _clamp(fleet_avg)

    return scores


def _optimization_scores(data_used: Dict[str, Any]) -> Dict[str, float]:
    opt = data_used.get("optimization_result")
    if not isinstance(opt, dict):
        return {}
    scores: Dict[str, float] = {}
    for row in opt.get("ranked_candidates") or []:
        if isinstance(row, dict) and row.get("aircraft"):
            scores[str(row["aircraft"])] = _clamp(float(row.get("total_score") or 0.0))
    return scores


def _confidence_scores(data_used: Dict[str, Any]) -> Dict[str, float]:
    conf = data_used.get("recommendation_confidence")
    if not isinstance(conf, dict):
        return {}
    scores: Dict[str, float] = {}
    for row in conf.get("aircraft_confidence") or []:
        if isinstance(row, dict) and row.get("aircraft"):
            scores[str(row["aircraft"])] = _clamp(float(row.get("overall_confidence") or 0.0))
    return scores


def _build_portfolio_ranking(data_used: Dict[str, Any]) -> List[Dict[str, Any]]:
    aircraft = _collect_aircraft(data_used)
    if not aircraft:
        return []

    own = _ownership_scores(data_used)
    market = _market_scores(data_used, aircraft)
    mission = _mission_scores(data_used, aircraft)
    opt = _optimization_scores(data_used)
    conf = _confidence_scores(data_used)

    ranked: List[Dict[str, Any]] = []
    for ac in aircraft:
        drivers = {
            "ownership": own.get(ac, 50.0),
            "market": market.get(ac, 50.0),
            "mission": mission.get(ac, 50.0),
            "optimization": opt.get(ac, 50.0),
            "confidence": conf.get(ac, 50.0),
        }
        total = sum(drivers[k] * _RANK_WEIGHTS[k] for k in _RANK_WEIGHTS)
        ranked.append(
            {
                "aircraft": ac,
                "total_score": round(_clamp(total), 2),
                "drivers": {k: round(_clamp(v), 2) for k, v in drivers.items()},
            }
        )

    ranked.sort(key=lambda r: (-float(r["total_score"]), str(r["aircraft"])))
    return ranked


def _build_fleet_summary(
    data_used: Dict[str, Any],
    portfolio_ranking: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    fleet = data_used.get("fleet_portfolio_strategy")
    own = data_used.get("ownership_intelligence")
    market = data_used.get("market_intelligence")

    fleet_efficiency = 50.0
    redundancy = 0.0
    mission_efficiency = 50.0
    if isinstance(fleet, dict):
        fleet_efficiency = float(fleet.get("total_fleet_efficiency_score") or 50.0)
        redundancy = float((fleet.get("redundancy_analysis") or {}).get("redundancy_score") or 0)
        coverage = fleet.get("mission_coverage_map") or {}
        if isinstance(coverage, dict) and coverage:
            mission_efficiency = sum(float(v) for v in coverage.values()) / len(coverage)

    lifecycle_avg = 50.0
    risk_exposure = 50.0
    if isinstance(own, dict):
        reports = own.get("ownership_reports") or []
        if reports:
            lifecycle_avg = sum(float(r.get("lifecycle_score") or 50) for r in reports if isinstance(r, dict)) / len(
                reports
            )
            risk_exposure = 100.0 - sum(
                float(r.get("ownership_risk_score") or 50) for r in reports if isinstance(r, dict)
            ) / len(reports)

    liquidity_score = 50.0
    if isinstance(market, dict):
        liquidity_score = float(market.get("liquidity_score") or 50.0)

    ranking_avg = 50.0
    if portfolio_ranking:
        ranking_avg = sum(float(r.get("total_score") or 0) for r in portfolio_ranking) / len(portfolio_ranking)

    fleet_health = _clamp(
        fleet_efficiency * 0.30 + lifecycle_avg * 0.25 + ranking_avg * 0.25 + (100.0 - risk_exposure) * 0.20
    )
    diversification = _clamp(100.0 - redundancy)

    return {
        "fleet_health_score": round(fleet_health, 2),
        "diversification_score": round(diversification, 2),
        "risk_exposure_index": round(_clamp(risk_exposure), 2),
        "liquidity_position": _liquidity_label(liquidity_score),
        "mission_efficiency": round(_clamp(mission_efficiency), 2),
    }


def _build_strategic_actions(
    data_used: Dict[str, Any],
    portfolio_ranking: Sequence[Dict[str, Any]],
) -> Dict[str, List[str]]:
    fleet = data_used.get("fleet_portfolio_strategy")
    own = data_used.get("ownership_intelligence")
    opt = data_used.get("optimization_result")

    replacement: List[str] = []
    acquire: List[str] = []
    if isinstance(fleet, dict):
        replacement = [str(a) for a in (fleet.get("replacement_priority_order") or []) if a]
        for rec in fleet.get("optimization_recommendation") or []:
            if isinstance(rec, str) and rec.startswith("add "):
                acquire.append(rec.replace("add ", "", 1).strip())

    own_map = _index_by_aircraft((own or {}).get("ownership_reports") or []) if isinstance(own, dict) else {}

    sell: List[str] = []
    for ac in replacement[:2]:
        row = own_map.get(ac) or {}
        lifecycle = float(row.get("lifecycle_score") or 50)
        if lifecycle < 55 or ac in replacement:
            sell.append(ac)

    upgrade: List[str] = list(replacement[:2])

    keep: List[str] = []
    ranked_names = [str(r.get("aircraft") or "") for r in portfolio_ranking]
    for ac in ranked_names:
        if ac and ac not in sell and ac not in upgrade:
            keep.append(ac)

    if isinstance(opt, dict):
        winner = str(opt.get("winner") or "")
        if winner and winner not in acquire:
            acquire.append(f"mission-optimized candidate: {winner}")

    return {
        "keep": keep[:5],
        "upgrade": upgrade[:3],
        "sell": list(dict.fromkeys(sell))[:3],
        "acquire": acquire[:3],
    }


def _build_insights(
    data_used: Dict[str, Any],
    portfolio_ranking: Sequence[Dict[str, Any]],
    strategic_actions: Dict[str, List[str]],
) -> List[str]:
    insights: List[str] = []
    fleet = data_used.get("fleet_portfolio_strategy")
    market = data_used.get("market_intelligence")
    own = data_used.get("ownership_intelligence")

    if isinstance(fleet, dict):
        redundancy = fleet.get("redundancy_analysis") or {}
        if float(redundancy.get("redundancy_score") or 0) >= 45:
            pairs = redundancy.get("duplicated_mission_pairs") or redundancy.get("overlapping_cabin_pairs") or []
            if pairs:
                insights.append(f"redundancy cluster detected: {pairs[0]}")
            else:
                insights.append("redundancy cluster detected across fleet mission overlap")

        util = (fleet.get("utilization_assumptions") or {}).get("annual_utilization_hours")
        if isinstance(util, int) and util < 200:
            insights.append("low utilization aircraft detected relative to fleet hour assumptions")

        gaps = fleet.get("gap_analysis") or []
        for gap in gaps[:2]:
            if "missing" in str(gap).lower():
                insights.append(str(gap))

    if isinstance(market, dict):
        if str(market.get("sell_timing") or "") == "favorable":
            insights.append("high liquidity exit opportunity in current market window")
        if str(market.get("market_state") or "") == "SELLER_MARKET":
            insights.append("seller-favorable market supports selective liquidation timing")

    if isinstance(own, dict):
        for row in own.get("ownership_reports") or []:
            if not isinstance(row, dict):
                continue
            if float(row.get("lifecycle_score") or 0) < 45:
                insights.append(f"elevated ownership cost burden on {row.get('aircraft')}")

    if strategic_actions.get("upgrade"):
        insights.append(f"upgrade priority aircraft: {', '.join(strategic_actions['upgrade'][:2])}")

    if portfolio_ranking:
        top = portfolio_ranking[0]
        insights.append(
            f"portfolio leader {top.get('aircraft')} at composite score {top.get('total_score')}"
        )

    return list(dict.fromkeys(insights))[:8]


def build_executive_synthesis(data_used: Any) -> Dict[str, Any]:
    """
    Build executive synthesis from full pipeline ``data_used`` payload.

    Consumes Phases 21–26 outputs only; does not mutate *data_used*.
    """
    du = data_used if isinstance(data_used, dict) else {}

    portfolio_ranking = _build_portfolio_ranking(du)
    fleet_summary = _build_fleet_summary(du, portfolio_ranking)
    strategic_actions = _build_strategic_actions(du, portfolio_ranking)
    insights = _build_insights(du, portfolio_ranking, strategic_actions)

    synthesis_id = hashlib.sha256(
        "|".join(
            [
                str(round(fleet_summary.get("fleet_health_score", 0), 2)),
                ",".join(r["aircraft"] for r in portfolio_ranking[:5]),
                ",".join(strategic_actions.get("keep") or [])[:80],
            ]
        ).encode("utf-8")
    ).hexdigest()[:12]

    result = ExecutiveIntelligenceSynthesis(
        fleet_summary=fleet_summary,
        portfolio_ranking=portfolio_ranking,
        strategic_actions=strategic_actions,
        insights=insights,
        synthesis_id=synthesis_id,
    )

    out = result.to_dict()
    out["executive_panel"] = {
        "fleet_health_score": fleet_summary.get("fleet_health_score"),
        "top_aircraft": portfolio_ranking[0]["aircraft"] if portfolio_ranking else "",
        "strategic_actions": strategic_actions,
        "insight_count": len(insights),
    }
    return out


def attach_executive_synthesis_if_enabled(
    query: str,
    response: Any,
) -> Any:
    """Attach ``data_used.executive_synthesis`` when env flag enabled."""
    if not executive_synthesis_enabled():
        return response
    if not isinstance(response, dict):
        return response
    out = dict(response)
    du = dict(out.get("data_used") or {}) if isinstance(out.get("data_used"), dict) else {}
    du["executive_synthesis"] = build_executive_synthesis(du)
    out["data_used"] = du
    return out


def evaluate_executive_synthesis_hooks(response: Any) -> List[str]:
    """Optional evaluation hooks for synthesis consistency."""
    if not isinstance(response, dict):
        return []
    du = response.get("data_used")
    if not isinstance(du, dict):
        return []
    bundle = du.get("executive_synthesis")
    if not isinstance(bundle, dict):
        return []

    failures: List[str] = []
    summary = bundle.get("fleet_summary") or {}
    ranking = bundle.get("portfolio_ranking") or []
    actions = bundle.get("strategic_actions") or {}

    for key in ("fleet_health_score", "diversification_score", "risk_exposure_index", "mission_efficiency"):
        val = summary.get(key)
        if val is not None and (float(val) > 100 or float(val) < 0):
            failures.append("synthesis_score_consistency")

    if summary.get("liquidity_position") not in (None, "strong", "moderate", "weak"):
        failures.append("synthesis_score_consistency")

    prev = 101.0
    for row in ranking:
        if not isinstance(row, dict):
            continue
        score = float(row.get("total_score") or 0)
        if score > 100 or score < 0:
            failures.append("portfolio_ranking_consistency")
        if score > prev + 0.01:
            failures.append("portfolio_ranking_consistency")
        prev = score

    for bucket in ("keep", "upgrade", "sell", "acquire"):
        if bucket not in actions:
            failures.append("strategic_actions_completeness")

    sell_set = set(actions.get("sell") or [])
    keep_set = set(actions.get("keep") or [])
    if sell_set & keep_set:
        failures.append("strategic_actions_consistency")

    if (os.getenv("CONSULTANT_RANKING_CONSISTENCY_ASSERT") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        dispatch_models = du.get("authority_dispatch_models") or []
        synth_winner = ""
        if ranking and isinstance(ranking[0], dict):
            synth_winner = str(ranking[0].get("aircraft") or "").strip()
        if dispatch_models and synth_winner and synth_winner != dispatch_models[0]:
            failures.append("dispatch_synthesis_rank_mismatch")

    return list(dict.fromkeys(failures))


__all__ = [
    "ExecutiveIntelligenceSynthesis",
    "attach_executive_synthesis_if_enabled",
    "build_executive_synthesis",
    "evaluate_executive_synthesis_hooks",
    "executive_synthesis_enabled",
]
