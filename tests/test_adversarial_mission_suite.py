"""Adversarial mission suite — operational credibility checks."""

from evals.adversarial_mission_suite import run_adversarial_suite


def test_adversarial_mission_suite_passes():
    result = run_adversarial_suite(random_cases=8, verbose=False)
    assert result.failed == 0, "\n".join(result.failures)
