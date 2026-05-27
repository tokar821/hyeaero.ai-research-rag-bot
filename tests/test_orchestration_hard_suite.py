"""CI wrapper for orchestration hard-suite."""

from evals.orchestration_hard_suite import run_orchestration_hard_suite


def test_orchestration_hard_suite_passes():
    result = run_orchestration_hard_suite(verbose=False)
    assert result.failed == 0, "\n".join(result.failures)
