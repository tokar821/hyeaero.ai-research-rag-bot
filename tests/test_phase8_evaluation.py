"""Phase 8 golden dataset and evaluation framework tests."""

from evaluation.aircraft_failure_report import build_aircraft_failure_report
from evaluation.golden_dataset import GoldenTestCase, dataset_summary, load_golden_cases
from evaluation.legacy_unified_benchmark import benchmark_case, benchmark_summary
from evaluation.path_accuracy_report import build_path_accuracy_report, format_path_accuracy_console
from evaluation.rollout_readiness import compute_rollout_readiness, RolloutReadiness
from evaluation.unified_evaluator import evaluate_unified_case, evaluate_unified_cases


def test_dataset_loads_minimum_cases():
    cases = load_golden_cases()
    summary = dataset_summary(cases)
    assert summary["total"] >= 175
    assert summary.get("FACT", 0) >= 50
    assert summary.get("MARKET", 0) >= 25
    assert summary.get("CAPABILITY", 0) >= 25
    assert summary.get("COMPARISON", 0) >= 25
    assert summary.get("ALTERNATIVE", 0) >= 25
    assert summary.get("MISSION", 0) >= 25
    assert summary.get("BUY_DECISION", 0) >= 25


def test_all_cases_use_valid_categories_and_tags():
    cases = load_golden_cases()
    for c in cases:
        assert c.id
        assert c.query
        assert c.category in {
            "FACT",
            "MARKET",
            "CAPABILITY",
            "COMPARISON",
            "ALTERNATIVE",
            "MISSION",
            "BUY_DECISION",
        }


def test_evaluator_falcon_8x_seats():
    case = GoldenTestCase(
        id="test-fact-1",
        category="FACT",
        query="How many seats does a Falcon 8X have?",
        expected_execution_path="aircraft_fact",
        expected_models=["Falcon 8X"],
        expected_behavior_tags=["factual_only", "no_mission_synthesis"],
    )
    result = evaluate_unified_case(case, enforce=True)
    assert result.route_correct is True
    assert result.model_correct is True
    assert result.behavior_correct is True
    assert result.score >= 0.99


def test_evaluator_mission_stays_none():
    case = GoldenTestCase(
        id="test-mission-1",
        category="MISSION",
        query="Recommend the best jet for New York to London weekly",
        expected_execution_path="none",
        expected_behavior_tags=["no_mission_synthesis"],
    )
    result = evaluate_unified_case(case, enforce=True)
    assert result.route_correct is True
    assert result.actual_execution_path == "none"


def test_benchmark_produces_winner():
    cases = load_golden_cases()
    result = benchmark_case(cases[0], enforce_unified=True)
    assert result.winner in ("legacy", "unified", "tie")
    assert 0.0 <= result.legacy_score <= 1.0
    assert 0.0 <= result.unified_score <= 1.0


def test_path_accuracy_report_generates():
    cases = load_golden_cases()[:10]
    results = evaluate_unified_cases(cases, enforce=True)
    report = build_path_accuracy_report(cases, results)
    assert "overall" in report
    assert "by_category" in report
    console = format_path_accuracy_console(report)
    assert "Path Accuracy Report" in console


def test_aircraft_failure_report_generates():
    cases = load_golden_cases()[:20]
    results = evaluate_unified_cases(cases, enforce=True)
    report = build_aircraft_failure_report(cases, results)
    assert "aircraft_failures" in report


def test_rollout_readiness_scoring():
    readiness = compute_rollout_readiness(
        path_report={"overall": {"total": 100, "pass_rate": 0.95}},
        aircraft_report={"total_failure_events": 5},
        benchmark={"unified_win_rate": 0.8},
        rollback_metrics={"authority_divergence_rate": 0.05, "hardening_failure_count": 2},
    )
    assert isinstance(readiness, RolloutReadiness)
    assert readiness.recommendation in {
        "NOT_READY",
        "LIMITED_ROLLOUT",
        "SAFE_FOR_25_PERCENT",
        "SAFE_FOR_50_PERCENT",
        "SAFE_FOR_100_PERCENT",
    }
    assert 0.0 <= readiness.score <= 1.0


def test_full_evaluation_subset_runs():
    cases = load_golden_cases()[:30]
    results = evaluate_unified_cases(cases, enforce=True)
    assert len(results) == 30
    bench = [benchmark_case(c) for c in cases]
    summary = benchmark_summary(bench)
    assert summary["total"] == 30


def test_readiness_from_evaluation_pipeline():
    cases = load_golden_cases()[:40]
    results = evaluate_unified_cases(cases, enforce=True)
    from evaluation.rollout_readiness import compute_rollout_readiness_from_evaluation
    from evaluation.legacy_unified_benchmark import benchmark_cases

    bench = benchmark_cases(cases)
    readiness = compute_rollout_readiness_from_evaluation(cases, results, bench)
    assert readiness.recommendation
