"""P0 — pipeline must not present eliminated aircraft after ranking."""

from __future__ import annotations

import logging

from services.elimination.elimination_invariant import (
    assert_elimination_invariant,
    collect_eliminated_models,
)
from services.orchestration.pipeline_orchestrator import run_deterministic_stages


def test_one_aircraft_only_no_latitude_in_ranking(caplog):
    caplog.set_level(logging.ERROR)
    q = "one aircraft only: 6 pax TEB London nonstop and KASE Telluride hot and high"
    du: dict = {}
    pipeline, _ = run_deterministic_stages(q, data_used=du)
    recs = [r.model for r in pipeline.recommendations if not getattr(r, "avoid", False)]
    from services.elimination.elimination_invariant import collect_hard_eliminated_models

    eliminated = collect_hard_eliminated_models(
        data_used=du,
        elimination_log=pipeline.elimination_log,
        explicit_eliminated=pipeline.eliminated_models,
    )
    for model, fr in (pipeline.feasibility_map or {}).items():
        feasible = getattr(fr, "feasible", None)
        if feasible is None and isinstance(fr, dict):
            feasible = fr.get("feasible")
        if feasible is False:
            eliminated.add(model.lower())
    assert_elimination_invariant(recs, eliminated)
    assert not any("latitude" in m.lower() for m in recs)
    assert not any(
        "ELIMINATION_INVARIANT_VIOLATION" in r.message for r in caplog.records
    )
