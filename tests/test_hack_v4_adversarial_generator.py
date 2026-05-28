"""HACK v4 — adversarial test generator (no inference)."""

from __future__ import annotations

import re

import pytest

from evals.hack_v4_adversarial_generator import (
    ALL_TRAP_CLASSES,
    compute_stress_score,
    generate_adversarial_suite,
    severity_from_stress,
    validate_adversarial_suite,
)


def test_generate_suite_meets_acceptance_criteria():
    rows = generate_adversarial_suite(seed=42, mutations_per_test=2)
    assert len(rows) >= 15
    validate_adversarial_suite(rows)


def test_at_least_five_trap_classes():
    rows = generate_adversarial_suite(seed=7, mutations_per_test=2)
    traps = {r["trap_class"] for r in rows}
    assert len(traps) >= 5
    assert traps <= set(ALL_TRAP_CLASSES)


def test_critical_severity_share():
    rows = generate_adversarial_suite(seed=42, mutations_per_test=2)
    critical = sum(1 for r in rows if r["severity"] == "CRITICAL")
    assert critical / len(rows) >= 0.30


def test_no_aircraft_catalog_names_in_prompts():
    rows = generate_adversarial_suite(seed=99, mutations_per_test=2)
    forbidden = (
        "citation cj",
        "learjet",
        "gulfstream g650",
        "global 7500",
        "challenger 350",
        "praetor 600",
    )
    for row in rows:
        blob = row["prompt"].lower()
        for name in forbidden:
            assert name not in blob, f"{name} in {row['test_id']}"


def test_contract_keys_only():
    rows = generate_adversarial_suite(seed=1, mutations_per_test=1)
    allowed = {
        "test_id",
        "prompt",
        "trap_class",
        "severity",
        "stress_score",
        "expected_failure_modes",
    }
    for row in rows:
        assert set(row.keys()) == allowed


def test_stress_score_severity_alignment():
    assert severity_from_stress(20) == "LOW"
    assert severity_from_stress(45) == "MEDIUM"
    assert severity_from_stress(70) == "HIGH"
    assert severity_from_stress(90) == "CRITICAL"
    score, _ = compute_stress_score(
        domain_count=5,
        region_tokens=("a", "b", "c", "d", "e"),
        class_pressure=20,
        hub_ambiguity=15,
        economic_physics_tension=20,
    )
    assert score >= 86


def test_non_repetitive_prompts():
    rows = generate_adversarial_suite(seed=42, mutations_per_test=2)
    prompts = [r["prompt"] for r in rows]
    assert len(prompts) == len(set(prompts))


def test_expected_failure_modes_present():
    rows = generate_adversarial_suite(seed=42, mutations_per_test=1)
    for row in rows:
        assert isinstance(row["expected_failure_modes"], list)
        assert len(row["expected_failure_modes"]) >= 1
        for mode in row["expected_failure_modes"]:
            assert re.match(r"^[a-z_]+$", mode)
