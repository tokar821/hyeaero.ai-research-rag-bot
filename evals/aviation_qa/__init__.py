"""
Automated aviation QA / evaluation — scenario engine, evaluator agent, improvement loop.
"""

from evals.aviation_qa.evaluator_agent import evaluate_advisor_response
from evals.aviation_qa.improvement_loop import build_improvement_plan
from evals.aviation_qa.runner import run_aviation_qa_suite

__all__ = [
    "evaluate_advisor_response",
    "build_improvement_plan",
    "run_aviation_qa_suite",
]
