"""Aviation QA evaluator and improvement loop tests."""

from evals.aviation_qa.evaluator_agent import evaluate_advisor_response
from evals.aviation_qa.improvement_loop import build_improvement_plan, classify_failure_source
from evals.aviation_qa.repetition_detection import BatchRepetitionTracker, score_answer_repetition
from evals.aviation_qa.runner import run_aviation_qa_suite
from evals.aviation_qa.schemas import EvaluatorVerdict
from evals.aviation_qa.tone_evaluation import score_tone


def test_repetition_detects_banned_opening():
    rep = score_answer_repetition("On my list I'd start with the Challenger 350.")
    assert rep.repetition_score > 0.2
    assert "on my list" in rep.overused_phrases


def test_tone_penalizes_diagnostic_headers():
    hum, broker, fake, brochure = score_tone(
        "Mission Summary\nBest Fit Aircraft: Challenger 350\nConfidence: 0.92"
    )
    assert broker < 0.6
    assert brochure >= 0


def test_evaluator_flags_forbidden_aircraft():
    case = {
        "id": "qa_test",
        "category": "asia_nonstop",
        "input": "New York to Tokyo nonstop westbound winter 10 pax",
        "golden": {
            "route_must_include": ["New York", "Tokyo"],
            "forbidden_any_models": ["Praetor 600", "Challenger 350"],
            "expected_any_models": ["Global 7500", "Gulfstream G650"],
        },
    }
    verdict = evaluate_advisor_response(
        case=case,
        answer="I'd start with the Praetor 600 for this leg — great range and comfort.",
        turn_profile={
            "routes": [{"origin": "New York", "destination": "Tokyo"}],
            "passengers": 10,
            "nonstop_required": True,
            "westbound_sensitive": True,
        },
        merged_profile={},
        mission_state={},
        recommendations=[{"model": "Praetor 600", "fit": "Good Fit"}],
        mission_category="ultra_long_range",
    )
    assert verdict.aircraft_realism == "FAIL"
    assert not verdict.passed
    assert verdict.main_failure


def test_evaluator_output_schema():
    case = {
        "id": "tone_ok",
        "input": "8 pax LA to Miami",
        "golden": {"route_must_include": ["Los Angeles", "Miami"]},
    }
    v = evaluate_advisor_response(
        case=case,
        answer=(
            "In practice, LA–Miami is a transcon mission — you don't need ULR. "
            "Most operators would look at super-mids with real payload margin after NBAA reserves."
        ),
        turn_profile={"routes": [{"origin": "Los Angeles", "destination": "Miami"}], "passengers": 8},
        merged_profile={},
        mission_state={},
        recommendations=[{"model": "Challenger 350"}],
    )
    d = v.to_dict()
    assert "route_realism" in d
    assert "humanness_score" in d
    assert "trust_score" in d


def test_improvement_plan_prioritizes_aircraft_failures():
    rows = [
        {
            "id": "x",
            "category": "asia_nonstop",
            "evaluator": {
                "passed": False,
                "main_failure": "Unrealistic aircraft recommended",
                "aircraft_realism": "FAIL",
                "route_realism": "PASS",
                "hallucination_risk": 0.1,
                "repetition_score": 0.1,
                "humanness_score": 0.5,
                "operational_realism": 0.4,
                "tone_broker_score": 0.5,
                "fake_confidence_risk": 0.2,
                "brochure_language_risk": 0.3,
                "missing_tradeoffs": False,
                "trust_score": 0.3,
                "sub_failures": ["impossible_aircraft_recommended:Praetor 600"],
            },
        }
    ]
    plan = build_improvement_plan(rows)
    assert plan["failure_sources"].get("impossible_aircraft", 0) >= 1
    assert plan["suggested_fixes_by_source"].get("impossible_aircraft")


def test_qa_suite_smoke():
    report = run_aviation_qa_suite(case_ids=["asia_003", "range_002"])
    assert report["summary"]["total_cases"] == 2
    assert "improvement_plan" in report
    assert "cases" in report
