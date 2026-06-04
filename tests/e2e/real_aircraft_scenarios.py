"""
Phase 53 — ground-truth scenarios from real market transaction bands.

Derived from verified catalog tiers and observed pre-owned transaction ranges (not synthetic certification).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RealAircraftScenario:
    scenario_id: str
    query: str
    expected_primary: str = ""
    expected_alternatives: Tuple[str, ...] = ()
    expect_infeasible: bool = False
    expect_no_ultra_long: bool = False
    budget_musd: Optional[float] = None


def _gulfstream_budget_scenarios() -> List[RealAircraftScenario]:
    rows: List[RealAircraftScenario] = []
    for musd in (8, 10, 12, 14, 16, 18, 20, 25, 30):
        rows.append(
            RealAircraftScenario(
                f"gs_under_{musd}m",
                f"I want a Gulfstream under ${musd}M",
                expected_primary="G280" if musd <= 14 else "",
                budget_musd=float(musd),
                expect_no_ultra_long=musd < 20,
            )
        )
    return rows


def _named_model_plausibility() -> List[RealAircraftScenario]:
    return [
        RealAircraftScenario("g650_18m_plausible", "G650 for $18M — is that plausible?", expected_primary="G650", budget_musd=18.0),
        RealAircraftScenario("g650_45m", "G650 for $45M — realistic?", expected_primary="G650", budget_musd=45.0),
        RealAircraftScenario("g700_65m", "G700 at $65M asking — fair?", expected_primary="G700", budget_musd=65.0),
        RealAircraftScenario("g700_5m_infeasible", "Can I buy a G700 for $5M?", expect_infeasible=True, budget_musd=5.0),
        RealAircraftScenario("g700_12m_infeasible", "G700 listed at $12M — realistic?", expect_infeasible=True, budget_musd=12.0),
        RealAircraftScenario("longitude_22m", "Citation Longitude for $22M — good buy?", expected_primary="Longitude", budget_musd=22.0),
        RealAircraftScenario("challenger350_18m", "Challenger 350 at $18M — worth it?", expected_primary="Challenger 350", budget_musd=18.0),
        RealAircraftScenario("falcon8x_50m", "Falcon 8X for $50M — market realistic?", expected_primary="Falcon 8X", budget_musd=50.0),
        RealAircraftScenario("praetor_18m", "Praetor 600 for $18M — plausible?", expected_primary="Praetor 600", budget_musd=18.0),
        RealAircraftScenario("cj4_7m", "Citation CJ4 under $7M — what should I buy?", expected_primary="Citation CJ4", budget_musd=7.0),
    ]


def _mission_buy_scenarios() -> List[RealAircraftScenario]:
    missions = [
        ("coast_6pax_20m", "Coast-to-coast nonstop, 6 passengers, $20M — what should I buy?", "Longitude", ("Challenger 350",), 20.0),
        ("coast_8pax_25m", "Coast-to-coast, 8 pax, $25M budget", "Citation Longitude", ("Challenger 650", "G280"), 25.0),
        ("regional_4pax_8m", "Regional US, 4 passengers, $8M budget", "", ("Citation CJ4", "Phenom 300"), 8.0),
        ("europe_us_12pax_40m", "Europe to US nonstop, 12 passengers, $40M", "G650", ("Falcon 8X",), 40.0),
        ("supermid_15m", "Best super-midsize jet under $15M", "", ("Longitude", "Challenger 350", "Praetor 600"), 15.0),
        ("supermid_18m", "Best super-midsize under $18M", "", ("Longitude", "Challenger 350"), 18.0),
        ("entry_jet_6m", "Best light jet under $6M", "", ("Citation CJ4", "Learjet 75"), 6.0),
        ("midsize_12m", "Best midsize jet for $12M", "", ("Citation Latitude", "Praetor 600"), 12.0),
    ]
    return [
        RealAircraftScenario(sid, q, expected_primary=prim, expected_alternatives=alts, budget_musd=b)
        for sid, q, prim, alts, b in missions
    ]


def _cheap_discovery() -> List[RealAircraftScenario]:
    return [
        RealAircraftScenario("cheap_gulfstream", "cheap gulfstream", expected_primary="G280", expected_alternatives=("G280",)),
        RealAircraftScenario("cheap_g650_probe", "I want a G650 but only have $12M", expect_infeasible=True, budget_musd=12.0),
        RealAircraftScenario("g280_vs_g650_budget", "G280 vs G650 — I have $14M", expected_alternatives=("G280", "G650")),
    ]


def _transaction_listing_style() -> List[RealAircraftScenario]:
    """Observed listing-band probes (realistic / suspicious / impossible)."""
    pairs = [
        ("txn_g650_19m", "Found a G650 for $19M — should I pursue?", "G650", 19.0, False),
        ("txn_g650_14m", "G650 asking $14M — too good to be true?", "G650", 14.0, False),
        ("txn_longitude_11m", "Longitude at $11M — good deal?", "Longitude", 11.0, False),
        ("txn_longitude_24m", "Longitude listed $24M", "Longitude", 24.0, False),
        ("txn_global7500_40m", "Global 7500 for $40M", "Global 7500", 40.0, False),
        ("txn_global7500_20m", "Global 7500 at $20M — realistic?", "", 20.0, True),
        ("txn_falcon7x_22m", "Falcon 7X for $22M", "Falcon 7X", 22.0, False),
        ("txn_cj4_5m", "CJ4 for $5M", "Citation CJ4", 5.0, False),
        ("txn_phenom_9m", "Phenom 300 at $9M", "Phenom 300", 9.0, False),
        ("txn_challenger650_28m", "Challenger 650 for $28M", "Challenger 650", 28.0, False),
    ]
    return [
        RealAircraftScenario(sid, q, expected_primary=prim or "", expect_infeasible=infeas, budget_musd=m)
        for sid, q, prim, m, infeas in pairs
    ]


def _year_ask_transactions() -> List[RealAircraftScenario]:
    rows: List[RealAircraftScenario] = []
    specs = [
        ("2018", "Citation Longitude", 19),
        ("2019", "Challenger 350", 17),
        ("2017", "Gulfstream G280", 13),
        ("2020", "Praetor 600", 16),
        ("2016", "Falcon 2000", 14),
        ("2019", "Gulfstream G650", 42),
        ("2018", "Citation CJ4", 6),
        ("2021", "Pilatus PC-24", 9),
        ("2015", "Learjet 75", 5),
        ("2017", "Falcon 7X", 28),
    ]
    for year, model, ask in specs:
        rows.append(
            RealAircraftScenario(
                f"txn_{year}_{model.replace(' ', '_').lower()}_{ask}m",
                f"{year} {model} for ${ask}M — good deal?",
                expected_alternatives=(model,),
                budget_musd=float(ask),
            )
        )
    templates = [
        ("G650", 35), ("G650", 48), ("G700", 58), ("Longitude", 20), ("Challenger 350", 16),
        ("G280", 12), ("Falcon 8X", 45), ("Global 6500", 38), ("Phenom 300", 8), ("Citation Latitude", 13),
    ]
    for model, ask in templates:
        rows.append(
            RealAircraftScenario(
                f"deal_{model.replace(' ', '_').lower()}_{ask}m",
                f"Is {model} at ${ask}M a good deal?",
                expected_primary=model,
                budget_musd=float(ask),
            )
        )
    return rows


def _budget_sweep() -> List[RealAircraftScenario]:
    rows: List[RealAircraftScenario] = []
    for musd in range(3, 36, 1):
        rows.append(
            RealAircraftScenario(
                f"best_jet_{musd}m",
                f"What is the best jet I can buy for ${musd}M?",
                budget_musd=float(musd),
                expect_no_ultra_long=musd < 22,
            )
        )
    return rows


def _extra_probes() -> List[RealAircraftScenario]:
    return [
        RealAircraftScenario("wait_buy", "should I buy now or wait for a G280?", budget_musd=12.0),
        RealAircraftScenario("stretch_g650", "Is it worth stretching to a G650 from $18M?", budget_musd=18.0),
        RealAircraftScenario("vs_longitude_challenger", "Longitude vs Challenger 350 for $20M ops", expected_alternatives=("Longitude", "Challenger 350")),
    ]


def _manufacturer_focus() -> List[RealAircraftScenario]:
    specs = [
        ("cessna_15m", "Best Citation for $15M", ("Citation Longitude", "Citation Latitude")),
        ("bombardier_20m", "Best Challenger under $20M", ("Challenger 350",)),
        ("embraer_18m", "Best Embraer jet for $18M", ("Praetor 600",)),
        ("dassault_25m", "Best Falcon under $25M", ("Falcon 2000", "Falcon 7X")),
    ]
    return [
        RealAircraftScenario(sid, q, expected_alternatives=alts, budget_musd=15.0 if "15" in sid else 20.0)
        for sid, q, alts in specs
    ]


def build_real_aircraft_scenarios() -> List[RealAircraftScenario]:
    parts = (
        _gulfstream_budget_scenarios()
        + _named_model_plausibility()
        + _mission_buy_scenarios()
        + _cheap_discovery()
        + _transaction_listing_style()
        + _budget_sweep()
        + _manufacturer_focus()
        + _year_ask_transactions()
        + _extra_probes()
    )
    seen: set[str] = set()
    out: List[RealAircraftScenario] = []
    for s in parts:
        if s.scenario_id not in seen:
            seen.add(s.scenario_id)
            out.append(s)
    return out


REAL_AIRCRAFT_SCENARIOS: List[RealAircraftScenario] = build_real_aircraft_scenarios()
