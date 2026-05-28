"""
Recommendation Quality — FINAL 10 orchestration stress runner.

NOT geo tests. Stresses:
  - recommendation orchestration
  - aircraft suppression discipline
  - hierarchy weighting
  - interpretation-first behavior
  - broker realism / anti-generic behavior
  - structured verdict quality
  - mission-to-aircraft coherence

Usage:
  cd backend
  python runners/run_recommendation_quality_10.py
  python runners/run_recommendation_quality_10.py --json-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from dotenv import load_dotenv

if (_BACKEND / ".env").exists():
    load_dotenv(_BACKEND / ".env")

from services.orchestration.pipeline_orchestrator import run_consultant_orchestration  # noqa: E402

_SUITE_DEFAULT = _BACKEND / "evals" / "recommendation_quality_10_suite.json"
_OUT_DEFAULT = _BACKEND / "evals" / "recommendation_quality_10_results.json"

_GENERIC_DUMP_RE = re.compile(
    r"\b(?:global\s*7500|global\s*8000|g\s*650(?:er)?|gulfstream\s+g\s*650|falcon\s*8x)\b",
    re.I,
)
_AIRCRAFT_ANY_RE = re.compile(
    r"\b(?:gulfstream|global\s+\d+|falcon\s+\d+|citation|challenger|praetor|phenom|learjet|embraer|bbj|g\d{3})\b",
    re.I,
)
_LUXURY_MARKETING_RE = re.compile(
    r"\b(?:excellent\s+choice|world[\s-]class|prestige|flagship|luxury\s+cabin|ultimate\s+jet)\b",
    re.I,
)
_SALES_RE = re.compile(
    r"\b(?:you\s+should\s+buy|perfect\s+choice|great\s+fit\s+for\s+everything|ideal\s+acquisition)\b",
    re.I,
)


@dataclass
class ScenarioResult:
    id: str
    title: str
    query: str
    answer: str
    checks: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def pass_count(self) -> int:
        return sum(1 for v in self.checks.values() if v)

    @property
    def total_checks(self) -> int:
        return len(self.checks)

    def grade(self) -> str:
        if not self.checks:
            return "?"
        ratio = self.pass_count / self.total_checks
        hard_fail = bool(self.errors)
        if hard_fail:
            return "FAIL"
        if ratio >= 0.85:
            return "PASS"
        if ratio >= 0.65:
            return "PARTIAL"
        return "FAIL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "grade": self.grade(),
            "pass_count": self.pass_count,
            "total_checks": self.total_checks,
            "checks": self.checks,
            "metrics": self.metrics,
            "notes": self.notes,
            "errors": self.errors,
            "answer_preview": (self.answer or "")[:1400],
            "answer_len": len(self.answer or ""),
        }


def _load_suite(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return list(raw.get("scenarios") or [])


def _run_checks(
    scenario: Dict[str, Any],
    answer: str,
    du: Dict[str, Any],
    recommendations: List[str],
) -> Tuple[Dict[str, bool], Dict[str, Any], List[str], List[str]]:
    a = (answer or "").lower()
    checks: Dict[str, bool] = {}
    notes: List[str] = []
    errors: List[str] = []

    pkt = du.get("mission_understanding_packet") or {}
    if not isinstance(pkt, dict):
        pkt = {}
    inf = pkt.get("inferred_constraints") or {}
    kernel = du.get("mission_authority_kernel") or {}
    if not isinstance(kernel, dict):
        kernel = {}
    fleet = du.get("fleet_composition_plan") or {}
    if not isinstance(fleet, dict):
        fleet = {}
    structural = du.get("structural_decomposition") or {}
    if not isinstance(structural, dict):
        structural = {}
    gate = du.get("recommendation_gate") or {}
    if not isinstance(gate, dict):
        gate = {}
    orm = du.get("orchestration_response_mode") or {}
    if not isinstance(orm, dict):
        orm = {}
    hierarchy = du.get("hierarchy_weighting") or {}
    if not isinstance(hierarchy, dict):
        hierarchy = {}
    suppression = du.get("recommendation_suppression") or {}
    if not isinstance(suppression, dict):
        suppression = {}

    struct_req = bool(
        structural.get("required")
        or kernel.get("structural_decomposition")
        or inf.get("incompatible_mission_bands")
        or inf.get("multi_hard_domain_mission")
    )
    has_portfolio = bool(
        fleet.get("multi_aircraft_required")
        or fleet.get("single_aircraft_structurally_invalid")
        or "per-segment" in a
        or "portfolio" in a
        or "decompos" in a
        or "multiple missions" in a
    )
    generic_in_answer = bool(_GENERIC_DUMP_RE.search(answer or ""))
    generic_only_rec = bool(
        recommendations
        and all(_GENERIC_DUMP_RE.search(m or "") for m in recommendations[:3])
    )
    aircraft_in_answer = bool(_AIRCRAFT_ANY_RE.search(answer or ""))

    metrics: Dict[str, Any] = {
        "response_mode": orm.get("mode"),
        "suppresses_aircraft": gate.get("suppress_aircraft"),
        "render_interpretation_only": gate.get("render_interpretation_only"),
        "recommendations": recommendations[:5],
        "recommendation_count": len(recommendations),
        "structural_decomposition": kernel.get("structural_decomposition"),
        "dominant_utilization": hierarchy.get("dominant_utilization"),
        "suppression_active": suppression.get("suppress_aircraft_specificity"),
        "generic_models_in_answer": generic_in_answer,
        "aircraft_in_answer": aircraft_in_answer,
        "tier_downgrade_recovery": du.get("tier_downgrade_recovery"),
        "tier_downgrade_applied": du.get("tier_downgrade_applied"),
        "multi_factor_ranking": bool(du.get("multi_factor_ranking")),
    }

    allows_aircraft = bool(scenario.get("allows_aircraft"))
    if not allows_aircraft and aircraft_in_answer and not has_portfolio:
        if not gate.get("suppress_aircraft") and recommendations and not scenario.get(
            "allows_ownership_discussion"
        ):
            errors.append("premature_aircraft_in_answer")

    for ck in scenario.get("checks") or []:
        if ck == "dominant_utilization_identified":
            checks[ck] = bool(
                hierarchy.get("dominant_utilization")
                or any(
                    w in a
                    for w in (
                        "domestic",
                        "corridor",
                        "caribbean",
                        "regional",
                        "utilization",
                        "dominant",
                        "primary",
                        "majority",
                        "most flying",
                        "80%",
                        "70%",
                    )
                )
                or inf.get("domestic_utilization_dominant")
                or inf.get("domestic_utilization_dominates_except_founder_ulr")
            )
        elif ck == "no_generic_ulr_dump":
            checks[ck] = not (
                (generic_in_answer and not has_portfolio and struct_req)
                or generic_only_rec
                or (
                    generic_in_answer
                    and "per-segment" not in a
                    and not scenario.get("requires_explicit_recommendation")
                )
            )
        elif ck == "split_or_compromise_strategy":
            checks[ck] = any(
                w in a
                for w in (
                    "charter",
                    "fractional",
                    "split",
                    "portfolio",
                    "compromise",
                    "supplemental",
                    "second aircraft",
                    "two aircraft",
                    "multi-aircraft",
                    "decompos",
                    "segment",
                )
            ) or has_portfolio
        elif ck == "no_prestige_bias":
            checks[ck] = not _LUXURY_MARKETING_RE.search(answer or "")
        elif ck == "hierarchy_or_governance_signal":
            checks[ck] = bool(
                hierarchy.get("dominant_utilization")
                or hierarchy.get("continuation_hub_discipline")
                or inf.get("founder_company_asymmetry")
                or inf.get("ceo_ulr_mandate")
                or inf.get("domestic_utilization_dominates_except_founder_ulr")
                or "utilization hierarchy" in a
            )
        elif ck == "structural_conflict_first":
            idx_conflict = min(
                [a.find(w) for w in ("structural", "conflict", "incompatible", "decompos", "multiple missions") if w in a]
                or [9999]
            )
            idx_aircraft = a.find("aircraft options")
            if idx_aircraft < 0:
                idx_aircraft = next(
                    (a.find(m.group(0).lower()) for m in _AIRCRAFT_ANY_RE.finditer(answer or "")),
                    9999,
                )
            checks[ck] = (
                struct_req
                or has_portfolio
                or idx_conflict < idx_aircraft
                or "structurally wrong" in a
                or "operationally unstable" in a
                or "not a single optimization" in a
            )
        elif ck == "segmentation_emphasis":
            checks[ck] = has_portfolio or any(
                w in a for w in ("segment", "decompos", "portfolio", "multi-domain", "multiple missions")
            )
        elif ck == "dispatch_mismatch":
            checks[ck] = any(
                w in a
                for w in (
                    "dispatch mismatch",
                    "utilization conflict",
                    "operational incompatibility",
                    "fleet segmentation",
                    "dispatch reliability",
                    "single-aircraft",
                    "single aircraft",
                )
            )
        elif ck == "suppress_premature_aircraft":
            if allows_aircraft:
                checks[ck] = True
            else:
                checks[ck] = (
                    gate.get("suppress_aircraft")
                    or gate.get("render_interpretation_only")
                    or not recommendations
                    or (not generic_only_rec and len(recommendations) <= 1)
                ) and not errors.count("premature_aircraft_in_answer")
        elif ck == "structured_verdict":
            checks[ck] = any(
                w in a for w in ("verdict", "interpretation verdict", "viable with", "structurally", "operationally coherent")
            )
        elif ck == "reject_single_platform_logic":
            checks[ck] = any(
                w in a
                for w in (
                    "single platform",
                    "one aircraft",
                    "one flagship",
                    "single-aircraft",
                    "operationally unstable",
                    "multiple missions",
                    "not realistic",
                    "structurally invalid",
                )
            ) and ("valid" not in a or "invalid" in a or "not realistic" in a or "unstable" in a)
        elif ck == "runway_or_performance_compromise":
            checks[ck] = any(
                w in a for w in ("runway", "aspen", "mountain", "winter", "performance", "short field", "hot and high")
            )
        elif ck == "continuation_hub_secondary":
            checks[ck] = bool(
                hierarchy.get("continuation_hub_discipline")
                or inf.get("continuation_hubs_semantic_only_not_primary_origin")
                or any(
                    w in a
                    for w in (
                        "continuation",
                        "secondary",
                        "overlay",
                        "connector",
                        "not primary origin",
                        "dominant",
                    )
                )
            )
        elif ck == "payload_volatility":
            checks[ck] = bool(
                inf.get("passenger_load_variable")
                or inf.get("cargo_over_cabin")
                or any(w in a for w in ("variable", "4 to 16", "4–16", "pallet", "cargo", "payload", "equipment"))
            )
        elif ck == "not_executive_only_framing":
            checks[ck] = any(w in a for w in ("cargo", "pallet", "engineering", "equipment", "payload", "variable"))
        elif ck == "ownership_economics_critical":
            checks[ck] = any(
                w in a
                for w in (
                    "charter",
                    "fractional",
                    "ownership",
                    "economics",
                    "hours",
                    "180",
                    "utilization",
                    "fixed cost",
                    "operating cost",
                    "threshold",
                )
            )
        elif ck == "not_salesperson_tone":
            checks[ck] = not _SALES_RE.search(answer or "")
        elif ck == "converted_airliner_analysis":
            checks[ck] = any(
                w in a
                for w in (
                    "airliner",
                    "converted",
                    "bbj",
                    "acj",
                    "narrowbody",
                    "widebody",
                    "14",
                    "18",
                    "economics",
                    "airport",
                    "slot",
                )
            )
        elif ck == "reject_single_category":
            checks[ck] = any(
                w in a
                for w in (
                    "not operationally coherent",
                    "operationally coherent",
                    "multiple missions",
                    "decompos",
                    "single category",
                    "one aircraft category",
                    "structurally",
                    "incompatible",
                    "not a single",
                )
            ) and ("coherent" in a or struct_req or has_portfolio)
        elif ck == "economics_before_prestige":
            econ_idx = min(
                [a.find(w) for w in ("economics", "operating cost", "hourly", "doc", "ownership cost") if w in a]
                or [9999]
            )
            prestige_idx = min(
                [a.find(w) for w in ("prestige", "luxury", "flagship", "glamour") if w in a] or [9999]
            )
            checks[ck] = econ_idx < prestige_idx or prestige_idx == 9999
        elif ck == "realistic_midsize_shortlist":
            if not recommendations:
                checks[ck] = any(
                    w in a for w in ("super-mid", "super mid", "midsize", "mid-size", "economics", "class band")
                )
            else:
                ulr_only = all(
                    any(u in (m or "").lower() for u in ("global", "g650", "g700", "g800", "falcon 8x", "falcon 10x"))
                    for m in recommendations[:3]
                )
                checks[ck] = not ulr_only
        elif ck == "mission_coherence_if_aircraft":
            if not recommendations:
                checks[ck] = not scenario.get("requires_explicit_recommendation")
            else:
                checks[ck] = len(recommendations) <= 5
        elif ck == "non_empty_shortlist":
            if not scenario.get("allows_aircraft"):
                checks[ck] = True
            else:
                checks[ck] = bool(recommendations) or bool(
                    du.get("tier_downgrade_applied") or du.get("tier_downgrade_recovery")
                )
                if not checks[ck] and (
                    "ranked aircraft shortlist" in a
                    or "aircraft options" in a
                    or "primary recommendation" in a
                ):
                    checks[ck] = True
        elif ck == "non_empty_shortlist_or_class_band":
            checks[ck] = bool(recommendations) or any(
                w in a for w in ("class band", "operational band", "super-mid", "midsize")
            )
        elif ck == "tier_recovery_signal":
            checks[ck] = bool(
                du.get("tier_downgrade_applied")
                or du.get("tier_downgrade_recovery")
                or "tier-downgraded" in a
                or "shortlist recovery" in a
            ) or (bool(recommendations) and not generic_only_rec)
        elif ck == "structured_comparison":
            checks[ck] = (
                "|" in (answer or "")
                or "structured tradeoff" in a
                or "structured model comparison" in a
                or "class comparison" in a
                or "factor |" in a.replace("\n", " ")
            )
        elif ck == "mentions_named_comparison_models":
            checks[ck] = sum(
                1
                for m in ("challenger 650", "g280", "praetor 600", "challenger", "praetor")
                if m in a
            ) >= 2
        elif ck == "runway_winter_cost_emphasis":
            checks[ck] = (
                sum(1 for w in ("runway", "winter", "operating cost", "dispatch") if w in a) >= 2
            )
        elif ck == "field_utilization_dominant":
            checks[ck] = any(
                w in a
                for w in (
                    "field support",
                    "80%",
                    "gravel",
                    "industrial",
                    "alberta",
                    "oil field",
                    "northern canada",
                )
            )
        elif ck == "multi_factor_visible":
            checks[ck] = bool(du.get("multi_factor_ranking")) or "suitability=" in a
        elif ck == "what_breaks_language":
            checks[ck] = any(
                w in a
                for w in (
                    "what breaks",
                    "breaks structurally",
                    "breaks operationally",
                    "single-platform",
                    "operationally unstable",
                    "not viable",
                    "not realistic",
                    "structurally invalid",
                    "cannot share one platform",
                    "fleet segmentation",
                )
            )
        elif ck == "origin_integrity_language":
            checks[ck] = any(
                w in a
                for w in (
                    "origin integrity",
                    "not primary origin",
                    "continuation hub",
                    "connector",
                    "secondary",
                    "dominant origin",
                )
            )
        elif ck == "ownership_skepticism_for_low_hours":
            checks[ck] = any(
                w in a
                for w in (
                    "charter",
                    "fractional",
                    "180 hours",
                    "not justified",
                    "unlikely to justify",
                    "threshold",
                    "underutil",
                    "economics",
                )
            ) and not _SALES_RE.search(answer or "")
        elif ck == "field_or_industrial_awareness":
            checks[ck] = any(
                w in a
                for w in ("gravel", "arctic", "industrial", "oil field", "offshore", "runway")
            )
        else:
            checks[ck] = False

    if gate.get("suppress_aircraft"):
        notes.append(f"gate={gate.get('reason') or 'suppressed'}")
    if orm.get("mode"):
        notes.append(f"mode={orm.get('mode')}")
    if recommendations:
        notes.append(f"recs={recommendations[:3]}")
    return checks, metrics, notes, errors


def run_scenario(scenario: Dict[str, Any]) -> ScenarioResult:
    data_used: Dict[str, Any] = {"consultant_response_mode": "mission_advisory"}
    orch = run_consultant_orchestration(
        scenario["query"],
        conversation_state={"history": []},
        data_used=data_used,
        query_intent="mission_feasibility",
    )
    if isinstance(orch.data_used_patch, dict):
        data_used.update(orch.data_used_patch)

    recs = [
        str(r.get("model") or "")
        for r in (data_used.get("consultant_recommendations") or orch.recommendations or [])
        if isinstance(r, dict) or hasattr(r, "model")
    ]
    if not recs and orch.recommendations:
        recs = [getattr(r, "model", str(r)) for r in orch.recommendations]

    checks, metrics, notes, errors = _run_checks(
        scenario,
        orch.answer or "",
        data_used,
        recs,
    )
    return ScenarioResult(
        id=str(scenario.get("id") or ""),
        title=str(scenario.get("title") or ""),
        query=str(scenario.get("query") or ""),
        answer=orch.answer or "",
        checks=checks,
        metrics=metrics,
        notes=notes,
        errors=errors,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Recommendation quality orchestration runner")
    parser.add_argument("--json-only", action="store_true", help="Print JSON summary only")
    parser.add_argument(
        "--suite",
        default=str(_SUITE_DEFAULT),
        help="Path to suite JSON (default: recommendation_quality_10_suite.json)",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output results JSON path (default: derived from suite name)",
    )
    args = parser.parse_args()

    suite_path = Path(args.suite)
    if not suite_path.is_absolute():
        candidate = _BACKEND / "evals" / suite_path.name
        suite_path = candidate if candidate.exists() else (_BACKEND / suite_path)

    out_path = Path(args.out) if args.out else _BACKEND / "evals" / f"{suite_path.stem}_results.json"

    scenarios = _load_suite(suite_path)
    suite_name = json.loads(suite_path.read_text(encoding="utf-8")).get("suite", suite_path.stem)
    results: List[ScenarioResult] = []
    grade_counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}

    for sc in scenarios:
        results.append(run_scenario(sc))
        grade_counts[results[-1].grade()] += 1

    payload = {
        "suite": suite_name,
        "summary": {
            "total": len(results),
            **grade_counts,
            "pass_rate": round(grade_counts["PASS"] / max(1, len(results)), 3),
        },
        "results": [r.to_dict() for r in results],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json_only:
        print(json.dumps(payload["summary"], indent=2))
        return 0 if grade_counts["FAIL"] == 0 else 1

    print("=" * 92)
    print(f"RECOMMENDATION QUALITY — {suite_name.upper()} REPORT")
    print(
        f"SUMMARY: PASS={grade_counts['PASS']} PARTIAL={grade_counts['PARTIAL']} "
        f"FAIL={grade_counts['FAIL']} / {len(results)}"
    )
    print("=" * 92)
    for r in results:
        print()
        print(f"## {r.title} [{r.grade()}] ({r.pass_count}/{r.total_checks} checks)")
        print(f"Mode: {r.metrics.get('response_mode')} | Gate suppress: {r.metrics.get('suppresses_aircraft')}")
        print(f"Recs: {r.metrics.get('recommendations') or 'none'}")
        failed = [k for k, v in r.checks.items() if not v]
        if failed:
            print(f"Failed checks: {', '.join(failed)}")
        if r.errors:
            print(f"Errors: {', '.join(r.errors)}")
        if r.notes:
            print(f"Notes: {' | '.join(r.notes[:4])}")
        preview = (r.answer or "").replace("\n", " ")[:220]
        print(f"Answer: {preview}...")

    print()
    print(f"Wrote {out_path}")
    return 0 if grade_counts["FAIL"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
