"""Run full Phase 32 production validation and generate reports."""

from __future__ import annotations

from tests.production_validation.broker_quality_score import compute_broker_quality_report
from tests.production_validation.report_generator import generate_all_reports
from tests.production_validation.validation_runner import run_validation


def main() -> None:
    print("Running 500-query production validation (dispatch-level)...")
    results = run_validation(limit=None, use_retrieval=False)
    report = compute_broker_quality_report(results)
    paths = generate_all_reports(report)
    print(f"Routing accuracy: {report['routing_accuracy_pct']}%")
    print(f"Dispatch accuracy: {report['dispatch_accuracy_pct']}%")
    print(f"Hallucination rate: {report['hallucination_rate_pct']}%")
    print(f"Broker quality score: {report['broker_quality_score']}")
    print("Reports written:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
