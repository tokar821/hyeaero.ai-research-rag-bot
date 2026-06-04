"""
Structured comparative analysis for consultant advisory turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.consultant.mission_state import MissionState
from services.consultant.recommendation_engine import (
    AircraftRecommendation,
    _AIRCRAFT_PROFILES,
    rank_aircraft_recommendations,
    score_aircraft_for_mission,
)


@dataclass
class ComparisonRow:
    model: str
    range_practical_nm: float
    mission_fit: str
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    verified: bool = True
    data_note: str = ""


@dataclass
class StructuredComparison:
    title: str
    models: List[str]
    rows: List[ComparisonRow]
    operational_tradeoffs: List[str]
    acquisition_vs_operating: str
    workload_note: str
    airport_capability: str
    markdown_table: str
    json_schema: Dict[str, Any]
    visual_normalized: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "models": list(self.models),
            "rows": [
                {
                    "model": r.model,
                    "range_practical_nm": r.range_practical_nm,
                    "mission_fit": r.mission_fit,
                    "pros": list(r.pros),
                    "cons": list(r.cons),
                }
                for r in self.rows
            ],
            "operational_tradeoffs": list(self.operational_tradeoffs),
            "acquisition_vs_operating": self.acquisition_vs_operating,
            "workload_note": self.workload_note,
            "airport_capability": self.airport_capability,
            "markdown_table": self.markdown_table,
            "json_schema": dict(self.json_schema),
            "visual_normalized": dict(self.visual_normalized),
        }


def _pros_cons_for_model(model: str, rec: Optional[AircraftRecommendation]) -> tuple[List[str], List[str]]:
    pros: List[str] = []
    cons: List[str] = []
    prof = _AIRCRAFT_PROFILES.get(model) or {}
    if prof.get("cabin_score", 0) >= 0.85:
        pros.append("Stand-up cabin / large-cabin comfort for the class.")
    if prof.get("operating_index", 1) <= 0.55:
        pros.append("Lower direct operating cost vs large-cabin peers.")
    from services.aircraft_truth import validate_aircraft_truth

    truth = validate_aircraft_truth(model, prof)
    if truth.verified and truth.facts and truth.facts.practical_range_nm >= 5000:
        pros.append(
            f"Practical range near {int(truth.facts.practical_range_nm)} nm — long-stage capable in class."
        )
    if prof.get("operating_index", 1) >= 0.9:
        cons.append("Higher fuel and crew cost — ownership efficiency depends on utilization.")
    if prof.get("runway_ft", 5000) >= 5200:
        cons.append("Runway and hot/high flexibility is not the primary strength.")
    if rec and rec.explanation:
        pros.extend(rec.explanation.strengths[:2])
        cons.extend(rec.explanation.penalties[:2])
    return pros[:3], cons[:3]


def build_structured_comparison(
    models: List[str],
    mission: MissionState,
    *,
    recommendations: Optional[List[AircraftRecommendation]] = None,
) -> StructuredComparison:
    """Side-by-side comparison with mission-fit table and visual-ready JSON."""
    uniq = list(dict.fromkeys(m for m in models if m))[:4]
    if len(uniq) < 2:
        ranked = rank_aircraft_recommendations(mission, max_results=4)
        uniq = [r.model for r in ranked[:4]]

    rec_by_model = {r.model: r for r in (recommendations or [])}
    rows: List[ComparisonRow] = []
    from services.aircraft_truth import validate_aircraft_truth

    from services.broker.graceful_degradation import degraded_comparison_note

    for model in uniq:
        truth = validate_aircraft_truth(model)
        from services.aircraft.aircraft_authority_service import get_authority_profile_dict

        prof = get_authority_profile_dict(model) or _AIRCRAFT_PROFILES.get(model)
        rec = rec_by_model.get(model)
        if truth.verified and truth.facts and prof:
            if not rec:
                rec = score_aircraft_for_mission(model, prof, mission)
            pros, cons = _pros_cons_for_model(model, rec)
            from services.recommendation.fit_policy import normalize_fit_label, score_to_fit_label

            if rec:
                fit_label = normalize_fit_label(
                    rec.fit or score_to_fit_label(rec.total_score, avoid=rec.avoid)
                )
            else:
                fit_label = "Good Fit"
            rows.append(
                ComparisonRow(
                    model=model,
                    range_practical_nm=float(truth.facts.practical_range_nm),
                    mission_fit=fit_label,
                    pros=pros,
                    cons=cons,
                    verified=True,
                )
            )
            continue

        rows.append(
            ComparisonRow(
                model=model,
                range_practical_nm=0.0,
                mission_fit="Unverified",
                pros=[],
                cons=[degraded_comparison_note(model)],
                verified=False,
                data_note="Directional comparison — brochure numbers not verified in-band.",
            )
        )

    tradeoffs: List[str] = []
    if mission.nonstop_requirement:
        tradeoffs.append("Nonstop requirement favors higher practical range — light jets drop on long legs.")
    if (mission.operating_cost_priority or "") == "high":
        tradeoffs.append("Operating-cost priority pulls toward super-midsize vs ultra-long unless utilization is high.")
    if mission.mountain_airport_requirement:
        tradeoffs.append("Mountain/hot-high missions penalize payload — verify runway and climb performance.")

    acq_note = (
        "Full ownership spreads fixed costs over hours; fractional preserves balance-sheet flexibility "
        "but sacrifices tail-specific customization."
        if (mission.acquisition_strategy or "") == "fractional"
        else "Acquisition economics should be weighed against annual hours — under ~200–250 hrs/year, "
        "fractional or charter may dominate fully burdened ownership cost."
    )

    workload = (
        "Light/super-midsize jets typically reduce crew coordination complexity; "
        "large-cabin/ULR adds cabin crew, international handling, and maintenance program scrutiny."
    )
    airport = (
        "Shorter runway or mountain missions favor aircraft with lower landing speeds and "
        "strong hot/high data — verify POH limits, not brochure range alone."
        if mission.runway_constraints or mission.mountain_airport_requirement
        else "Airport compatibility is adequate for typical Part 91 U.S. destinations in this comparison set."
    )

    if len(rows) < 2:
        ranked = rank_aircraft_recommendations(mission, max_results=4)
        for r in ranked:
            if r.model not in {x.model for x in rows}:
                rows.append(
                    ComparisonRow(
                        model=r.model,
                        range_practical_nm=0.0,
                        mission_fit=r.fit or "Partial",
                        pros=[],
                        cons=[],
                        verified=True,
                    )
                )
            if len(rows) >= 2:
                break

    md_lines = [
        "| Model | Practical range (nm) | Mission fit |",
        "|---|---:|---|",
    ]
    for r in rows:
        rng = str(int(r.range_practical_nm)) if r.verified else "—"
        md_lines.append(f"| {r.model} | {rng} | {r.mission_fit} |")
    markdown_table = "\n".join(md_lines)

    json_schema = {
        "comparison_type": "mission_fit_table",
        "mission": mission.to_dict(),
        "aircraft": [
            {
                "model": r.model,
                "metrics": {
                    "practical_range_nm": r.range_practical_nm,
                    "mission_fit": r.mission_fit,
                },
                "pros": r.pros,
                "cons": r.cons,
            }
            for r in rows
        ],
        "tradeoffs": tradeoffs,
    }

    fit_order = {"Strong Fit": 0, "Good Fit": 1, "Partial Fit": 2, "Not Recommended": 3}
    visual_normalized = {
        "comparison_cards": [
            {
                "model": r.model,
                "mission_fit": r.mission_fit,
                "badge": "lead_option" if i == 0 else "alternative",
                "pros": r.pros,
                "cons": r.cons,
            }
            for i, r in enumerate(sorted(rows, key=lambda x: fit_order.get(x.mission_fit, 9)))
        ],
        "mission_fit_table": json_schema["aircraft"],
    }

    title = " vs ".join(r.model for r in rows[:3])
    if len(rows) > 3:
        title += " (+others)"

    return StructuredComparison(
        title=title,
        models=[r.model for r in rows],
        rows=rows,
        operational_tradeoffs=tradeoffs,
        acquisition_vs_operating=acq_note,
        workload_note=workload,
        airport_capability=airport,
        markdown_table=markdown_table,
        json_schema=json_schema,
        visual_normalized=visual_normalized,
    )
