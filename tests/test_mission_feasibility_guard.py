"""Phase 56 — ultra-long missions must not surface short-range jets as candidates."""

from __future__ import annotations

import pytest

from tests.e2e.broker_certification_helpers import broker_certify

_INFEASIBLE = (
    "cj4",
    "latitude",
    "longitude",
    "g280",
    "praetor 600",
    "challenger 350",
)


@pytest.mark.parametrize(
    "query",
    [
        "8 passengers New York to Tokyo nonstop",
        "8 passengers New York to Tokyo",
    ],
)
def test_mission_feasibility_guard(query: str):
    answer, du, path = broker_certify(query, prefer_e2e=False)
    assert path == "layers"
    assert du.get("mission_feasibility_checked") is True
    infeasible = du.get("mission_infeasible_models") or []
    assert infeasible, "expected infeasible model list for NYC-Tokyo"

    br = du.get("broker_reasoning") or {}
    candidates = []
    for block in (br.get("category"), br.get("alternatives"), br.get("mission")):
        if isinstance(block, dict):
            candidates.extend(block.get("candidates") or block.get("models") or [])
    comp = br.get("comparison") or {}
    if isinstance(comp, dict):
        candidates.extend(comp.get("models") or [])

    low_join = " ".join(str(c).lower() for c in candidates)
    for tag in _INFEASIBLE:
        assert tag not in low_join, f"infeasible aircraft {tag!r} still in candidates: {candidates}"

    low_ans = answer.lower()
    for tag in ("citation cj4", "citation latitude"):
        if tag in low_ans and "not feasible" not in low_ans and "cannot" not in low_ans:
            pytest.fail(f"answer promotes infeasible {tag!r} without caveat: {answer[:400]}")
