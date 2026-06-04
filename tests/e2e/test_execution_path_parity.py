"""
Phase 54 — measure and enforce divergence between e2e and layers execution paths.
"""

from __future__ import annotations

import pytest

from tests.e2e.broker_certification_helpers import broker_certify
from tests.e2e.execution_path_parity_helpers import (
    evaluate_parity,
    parity_strict_enabled,
    write_parity_report,
)

pytestmark = pytest.mark.deterministic

_PARITY_QUERIES = [
    ("coast_buy", "Coast-to-coast nonstop, 6 passengers, $20M — what should I buy?"),
    ("g650_fair", "G650 asking $42M — fair price?"),
    ("best_supermid", "Best super-midsize jet under $18M"),
    ("cheap_g650", "I want a G650 but only have $12M"),
    ("compare", "G650 vs Falcon 8X"),
]

_SESSION_OBS: list = []


@pytest.fixture(scope="session", autouse=True)
def _parity_report_session():
    global _SESSION_OBS
    _SESSION_OBS = []
    yield
    write_parity_report(_SESSION_OBS)
    if parity_strict_enabled():
        critical = [o for o in _SESSION_OBS if o.critical_failures]
        assert not critical, (
            "Execution path parity critical failures: "
            + "; ".join(f"{o.scenario_id}:{o.critical_failures}" for o in critical)
        )


@pytest.mark.parametrize("scenario_id,query", _PARITY_QUERIES)
def test_execution_path_parity_sample(scenario_id: str, query: str):
    ans_layers, du_layers, path_layers = broker_certify(query, prefer_e2e=False)
    assert path_layers == "layers", f"{scenario_id}: expected layers path"
    assert du_layers.get("broker_certify_path") == "layers"

    ans_e2e, du_e2e, path_e2e = broker_certify(query, prefer_e2e=True)
    obs = evaluate_parity(
        scenario_id,
        query,
        ans_e2e=ans_e2e,
        du_e2e=du_e2e,
        path_e2e=path_e2e,
        ans_layers=ans_layers,
        du_layers=du_layers,
        path_layers=path_layers,
    )
    _SESSION_OBS.append(obs)

    assert ans_layers.strip(), f"{scenario_id}: layers answer empty"
    if path_e2e == "e2e":
        assert du_e2e.get("broker_certify_path") == "e2e"
        assert ans_e2e.strip()

    if parity_strict_enabled() and obs.critical_failures:
        pytest.fail(
            f"{scenario_id} parity critical: {obs.critical_failures} "
            f"(e2e_primary={obs.primary_e2e!r} layers_primary={obs.primary_layers!r})"
        )
