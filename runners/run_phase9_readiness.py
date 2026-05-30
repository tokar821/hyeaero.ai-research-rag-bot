#!/usr/bin/env python3
"""Phase 9 production readiness report — operational rollout observability only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from monitoring.failure_analysis import build_live_failure_reports
from monitoring.production_health_score import compute_production_health
from monitoring.unified_rollout_dashboard import build_rollout_dashboard_snapshot
from rollout.rollout_plan import get_rollout_stages, recommend_next_stage


def run_phase9_readiness(*, output_dir: Path | None = None) -> int:
    out = output_dir or (_ROOT / "evals" / "phase9")
    out.mkdir(parents=True, exist_ok=True)

    dashboard = build_rollout_dashboard_snapshot()
    health = compute_production_health(dashboard)
    stage_rec = recommend_next_stage()
    failures = build_live_failure_reports()

    report = {
        "rollout_health": {
            "dashboard": dashboard,
            "production_health": health.to_dict(),
        },
        "production_health": health.to_dict(),
        "rollout_stages": [s.to_dict() for s in get_rollout_stages()],
        "recommended_rollout_stage": stage_rec.to_dict(),
        "top_failing_aircraft": failures["top_failing_aircraft"],
        "top_failing_execution_paths": failures["top_failing_execution_paths"],
        "top_failing_categories": failures["top_failing_categories"],
    }

    report_path = out / "phase9_readiness_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Phase 9 Production Readiness ===\n")
    print("Production Health:", json.dumps(health.to_dict(), indent=2))
    print("\nRecommended Rollout Stage:", json.dumps(stage_rec.to_dict(), indent=2))
    print("\nRollout Dashboard (summary):")
    print(json.dumps(dashboard.get("rollout") or {}, indent=2))
    print("\nTop Failing Aircraft:")
    print(json.dumps(failures["top_failing_aircraft"][:5], indent=2))
    print("\nTop Failing Execution Paths:")
    print(json.dumps(failures["top_failing_execution_paths"][:5], indent=2))
    print(f"\nFull report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_phase9_readiness())
