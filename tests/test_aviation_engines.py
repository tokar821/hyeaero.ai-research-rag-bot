"""Aviation reasoning engines: distance, feasibility, recommendation catalog."""

import math

from rag.aviation_engines.capabilities import (
    filter_by_mission_pax_budget,
    filter_by_pax_budget,
    load_capability_rows,
    mission_possible_for_row,
)
from rag.aviation_engines.context import build_aviation_engines_block
from rag.aviation_engines.geo import mission_endpoints_from_text, nm_between, required_range_nm
from rag.consultant_fine_intent import ConsultantFineIntent, ConsultantFineIntentResult


def test_great_circle_kjfk_klax_approx_nm():
    d = nm_between((40.6398, -73.7789), (33.9425, -118.4081))
    assert 2100 <= d <= 2200


def test_city_pair_new_york_to_los_angeles():
    r = mission_endpoints_from_text("Can a Citation XLS+ fly New York to Los Angeles?", [])
    assert r is not None
    _, _, nm = r
    assert 2100 <= nm <= 2200


def test_required_range_margin():
    assert math.isclose(required_range_nm(2000), 2300.0)


def test_catalog_load():
    rows = load_capability_rows()
    assert len(rows) >= 3
    assert any("Citation" in str(r.get("aircraft_model", "")) for r in rows)


def test_mission_possible_challenger_300_vs_xls():
    rows = {r["aircraft_model"]: r for r in load_capability_rows()}
    assert rows
    mission = 2145.0
    assert not mission_possible_for_row(mission, rows["Citation XLS+"])  # 2102250 * 1.15 > 2100
    assert mission_possible_for_row(mission, rows["Challenger 300"])


def test_recommendation_filter():
    req = required_range_nm(2145.0)
    rec = filter_by_mission_pax_budget(req, passengers=8, budget_usd=12_000_000, limit=5)
    models = [r.get("aircraft_model") for r in rec]
    assert "Challenger 300" in models or "Citation Sovereign+" in models
    assert "Falcon 2000" not in models
    # Closest feasible range match should lead the list.
    assert rec[0]["aircraft_model"] in ("Hawker 800XP", "Citation Sovereign+", "Challenger 300")


def test_recommendation_budget_cap_excludes_above_buffer():
    req = required_range_nm(1500.0)
    rec = filter_by_mission_pax_budget(req, passengers=6, budget_usd=9_000_000, limit=5)
    cap = 9_000_000 * 0.85
    assert rec
    for r in rec:
        p = float(r.get("typical_market_price") or r.get("typical_market_price_usd") or 0)
        assert p <= cap


def test_budget_advisory_no_mission_assumption_recommends_under_budget():
    # Budget-only advisory should work without a route/range; do not require transcon-class range.
    rec = filter_by_pax_budget(passengers=6, budget_usd=5_000_000, limit=8)
    models = [r.get("aircraft_model") for r in rec]
    assert "Citation CJ2" in models or "Citation Ultra" in models or "Learjet 45" in models


def test_city_alias_new_york_uses_kteb():
    r = mission_endpoints_from_text("New York to Los Angeles nonstop", [])
    assert r is not None
    assert "KTEB" in (r[0], r[1])
    assert "KLAX" in (r[0], r[1])


def test_build_engines_block_recommendation():
    fine = ConsultantFineIntentResult(
        ConsultantFineIntent.AIRCRAFT_RECOMMENDATION,
        0.9,
        {"icaos": ["KJFK", "KLAX"], "passengers": 8, "budget_usd": 12},
    )
    blk = build_aviation_engines_block(fine, "8 pax $12mm budget NYC to LA")
    assert "required usable range" in blk.lower() or "mission" in blk.lower()
    assert "Citation" in blk or "Challenger" in blk or "Falcon" in blk
