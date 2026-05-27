"""
Orchestration hard-suite — operational trust regressions before feature expansion.

Run: python -m evals.orchestration_hard_suite
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Enable orchestration path in tests
os.environ.setdefault("CONSULTANT_ORCHESTRATION", "1")


@dataclass
class HardCase:
    case_id: str
    query: str
    description: str
    assert_fn: Callable[[Dict[str, Any]], Optional[str]]


@dataclass
class HardSuiteResult:
    passed: int = 0
    failed: int = 0
    failures: List[str] = field(default_factory=list)


def _run_case(query: str) -> Dict[str, Any]:
    from services.orchestration.pipeline_orchestrator import run_deterministic_stages

    data_used: Dict[str, Any] = {}
    pipeline, trace = run_deterministic_stages(query, data_used=data_used)
    rec_models = [r.model for r in (pipeline.recommendations or []) if not r.avoid]
    rec_verdicts = [r.fit_verdict for r in (pipeline.recommendations or [])]
    return {
        "pipeline": pipeline,
        "trace": trace,
        "data_used": data_used,
        "recommendations": pipeline.recommendations or [],
        "rec_models": rec_models,
        "rec_verdicts": rec_verdicts,
        "feasible_models": list(pipeline.feasible_models or []),
        "elimination_log": list(pipeline.elimination_log or []),
    }


def _super_mid_survivor(models: List[str]) -> bool:
    sm = (
        "challenger 350",
        "g280",
        "gulfstream g280",
        "praetor",
        "longitude",
        "citation latitude",
        "super-mid",
    )
    blob = " ".join(m.lower() for m in models)
    return any(s in blob for s in sm)


def _has_primary_outside_ulr(ctx: Dict[str, Any]) -> bool:
    from services.broker.broker_verdicts import BrokerVerdict

    primary = BrokerVerdict.PRIMARY_RECOMMENDATION.value
    for r in ctx["recommendations"]:
        if r.avoid:
            continue
        if (r.fit_verdict or "") == primary:
            cat = str(getattr(r, "category", "") or "").lower()
            if cat not in ("ultra-long", "ultra long range") and "g650" not in r.model.lower():
                if "global 7500" not in r.model.lower() and "falcon 8" not in r.model.lower():
                    return True
    return False


HARD_CASES: List[HardCase] = [
    HardCase(
        "teb_london_winter",
        "8 passengers Teterboro to London nonstop February recommend",
        "TEB–London winter — no super-mid in ranked shortlist; verified distance only",
        lambda ctx: (
            "super-mid in shortlist: " + ", ".join(ctx["rec_models"])
            if _super_mid_survivor(ctx["rec_models"])
            else (
                "ranked shortlist on unresolved route distance"
                if ctx["rec_models"]
                and any(
                    r.get("source") == "unresolved"
                    for r in (ctx["data_used"].get("route_distance_authority") or [])
                )
                else None
            )
        ),
    ),
    HardCase(
        "nyc_dubai",
        "6 pax New York to Dubai nonstop recommend",
        "NYC–Dubai — ULR band only or empty; no light jet PRIMARY",
        lambda ctx: (
            "light jet recommended: " + ", ".join(ctx["rec_models"])
            if any("cj" in m.lower() or "phenom" in m.lower() for m in ctx["rec_models"])
            else None
        ),
    ),
    HardCase(
        "dallas_aspen",
        "4 passengers Dallas to Aspen hot and high recommend",
        "Dallas–Aspen — field-flexible types; no super-mid default",
        lambda ctx: (
            "super-mid recommended on mountain leg: " + ", ".join(ctx["rec_models"])
            if _super_mid_survivor(ctx["rec_models"])
            and not any(
                k in " ".join(ctx["rec_models"]).lower()
                for k in ("pc-24", "pc-12", "cj4", "cj3")
            )
            else None
        ),
    ),
    HardCase(
        "aspen_telluride",
        "Aspen to Telluride ski trip 6 passengers recommend",
        "Aspen–Telluride — route extracted; short-field survivors",
        lambda ctx: (
            "no recommendations and route not blocked"
            if not ctx["rec_models"]
            and not ctx["data_used"].get("route_blocks_ranking")
            and not (ctx["pipeline"].mission_state.routes or [])
            else None
        ),
    ),
    HardCase(
        "sfo_tokyo_westbound",
        "6 executives San Francisco to Tokyo nonstop westbound January recommend",
        "SFO–Tokyo westbound — ULR / large only",
        lambda ctx: (
            "super-mid on transpacific westbound: " + ", ".join(ctx["rec_models"])
            if _super_mid_survivor(ctx["rec_models"])
            else None
        ),
    ),
    HardCase(
        "fractional_ownership",
        "I fly about 220 hours a year — fractional vs full ownership on a Challenger 350?",
        "Ownership economics — dedicated branch, non-empty, no ranked shortlist",
        lambda ctx: None,  # handled in run_ownership_case
    ),
    HardCase(
        "caribbean_short",
        "Miami to Nassau 6 passengers short runway recommend",
        "Caribbean short runway — no ULR PRIMARY",
        lambda ctx: (
            "ULR primary on short caribbean: " + ", ".join(ctx["rec_models"])
            if any("global 7500" in m.lower() or "g650" in m.lower() for m in ctx["rec_models"][:1])
            else None
        ),
    ),
]


def _run_ownership_case() -> Optional[str]:
    from services.orchestration.pipeline_orchestrator import run_consultant_orchestration

    result = run_consultant_orchestration(
        "I fly about 220 hours a year — fractional vs full ownership on a Challenger 350?"
    )
    if not (result.answer or "").strip():
        return "empty ownership response"
    if result.recommendations:
        return "ownership query produced ranked recommendations"
    if "Ownership Economics" not in result.answer:
        return "missing ownership economics structure"
    return None


def run_orchestration_hard_suite(*, verbose: bool = True) -> HardSuiteResult:
    out = HardSuiteResult()
    for case in HARD_CASES:
        if case.case_id == "fractional_ownership":
            err = _run_ownership_case()
        else:
            try:
                ctx = _run_case(case.query)
                err = case.assert_fn(ctx)
            except Exception as exc:
                err = f"exception: {exc}"
        if err:
            out.failed += 1
            msg = f"[FAIL] {case.case_id}: {err} — {case.description}"
            out.failures.append(msg)
            if verbose:
                print(msg)
        else:
            out.passed += 1
            if verbose:
                print(f"[PASS] {case.case_id}: {case.description}")
    if verbose:
        print(f"\nHard suite: {out.passed} passed, {out.failed} failed")
    return out


if __name__ == "__main__":
    import sys

    result = run_orchestration_hard_suite()
    sys.exit(1 if result.failed else 0)
