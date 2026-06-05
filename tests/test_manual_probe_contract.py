"""
Manual probe contract — 10 user-verified queries (architecture + full-stack).

Architecture signals run fast; broker_certify layers tests validate end-to-end prose.
"""

from __future__ import annotations

import re

import pytest

from services.broker_execution.execution_intent_lock import ExecutionProfile, attach_execution_intent_lock
from services.broker_execution.fact_pack_builder import build_fact_pack
from services.broker_execution.mission_broker_answer import build_deterministic_mission_answer
from services.broker_execution.mission_feasibility_broker import build_mission_feasibility_broker_note
from services.broker_execution.output_governance import apply_governed_client_answer
from services.broker_execution.tail_answer_shaper import shape_tail_client_answer
from services.broker_execution.comparison_broker_facts import render_comparison_client_answer
from services.broker_execution.tail_depth_mode import TailDepthMode, classify_tail_depth_mode
from rag.consultant_query_anchor import gallery_user_query_for_image_pipeline
from tests.e2e.broker_certification_helpers import FORBIDDEN_HEADERS, broker_certify

_FORBIDDEN_SCAFFOLD = re.compile(
    r"(?is)\b(?:INSUFFICIENT_DATA|OPERATIONAL SYNTHESIS|Mission Fit|Overview|Analysis|Recommendation|Risks)\b"
)
_ACQUISITION_LEAK = re.compile(r"(?is)\b(?:logbooks?|engine\s+program|due\s+diligence)\b")
_SCAFFOLD_HEADERS = re.compile(
    r"(?is)\b(?:Overview|Analysis|Recommendation|Risks|Mission Fit|Aircraft Options|Verdict)\s*:"
)


PROBES = [
    {
        "id": "tail_owner",
        "query": "Who owns N807JS?",
        "expect_profile": ExecutionProfile.TAIL_OWNER,
        "expect_depth": TailDepthMode.OWNER,
        "forbid": _ACQUISITION_LEAK,
        "expect_in_answer": ("owned", "807JS"),
    },
    {
        "id": "tail_sale",
        "query": "Is N807JS for sale?",
        "expect_profile": ExecutionProfile.TAIL_SALE_STATUS,
        "expect_depth": TailDepthMode.SALE_STATUS,
        "forbid": _ACQUISITION_LEAK,
        "expect_lead_yes_no": True,
    },
    {
        "id": "mission_boston_denver",
        "query": "Boston to Denver 6 passengers $10M nonstop",
        "expect_profile": ExecutionProfile.MISSION,
        "expect_depth": TailDepthMode.NONE,
        "forbid": _SCAFFOLD_HEADERS,
        "expect_mission_models": True,
    },
    {
        "id": "mission_nyc_tokyo",
        "query": "8 passengers New York to Tokyo nonstop under $30M budget",
        "expect_profile": ExecutionProfile.MISSION,
        "expect_depth": TailDepthMode.NONE,
        "require_mission_note": True,
        "expect_in_answer": ("tokyo", "30"),
        "expect_budget_no": True,
    },
    {
        "id": "comparison_praetor_longitude",
        "query": "Praetor 600 vs Citation Longitude",
        "expect_profile": ExecutionProfile.COMPARISON,
        "expect_depth": TailDepthMode.NONE,
        "require_comparison_broker": True,
        "expect_in_answer": ("praetor", "longitude"),
    },
    {
        "id": "comparison_g280_longitude",
        "query": "Gulfstream G280 vs Citation Longitude for transcon",
        "expect_profile": ExecutionProfile.COMPARISON,
        "expect_depth": TailDepthMode.NONE,
        "require_comparison_broker": True,
    },
    {
        "id": "tail_detail",
        "query": "Tell me everything about N807JS",
        "expect_profile": ExecutionProfile.TAIL_DETAIL,
        "expect_depth": TailDepthMode.DETAIL,
        "require_tail_profile": True,
    },
    {
        "id": "tail_summary",
        "query": "What aircraft is N807JS?",
        "expect_profile": ExecutionProfile.TAIL_SUMMARY,
        "expect_depth": TailDepthMode.SUMMARY,
    },
    {
        "id": "comparison_multi_route",
        "query": "Challenger 350 vs Praetor 500 for NYC to LA",
        "expect_profile": ExecutionProfile.COMPARISON,
        "expect_depth": TailDepthMode.NONE,
        "require_comparison_broker": True,
    },
    {
        "id": "mission_coast",
        "query": "Recommend a jet for 6 passengers Boston to Miami nonstop under $12M",
        "expect_profile": ExecutionProfile.MISSION,
        "expect_depth": TailDepthMode.NONE,
        "forbid": _SCAFFOLD_HEADERS,
    },
]


@pytest.mark.parametrize("probe", PROBES, ids=[p["id"] for p in PROBES])
def test_probe_architecture_signals(probe):
    q = probe["query"]
    du: dict = {}
    profile = attach_execution_intent_lock(du, q)
    depth, _reg = classify_tail_depth_mode(q)

    assert profile == probe["expect_profile"], f"profile={profile.value}"
    assert depth == probe["expect_depth"], f"depth={depth.value}"

    if probe.get("require_mission_note"):
        note = build_mission_feasibility_broker_note(q)
        assert note
        assert "tokyo" in note.lower()

    if probe.get("require_comparison_broker"):
        pack = build_fact_pack(q, {})
        kinds = [f.get("kind") for f in pack.get("facts") or []]
        assert "comparison_broker" in kinds

    if probe.get("require_tail_profile"):
        pack = build_fact_pack(q, {"tail_depth_mode": "detail", "tail_registration": "N807JS"})
        assert any(f.get("kind") == "tail_profile" for f in pack.get("facts") or [])

    if probe["id"] == "mission_boston_denver":
        depth2, reg2 = classify_tail_depth_mode(q)
        assert reg2 is None
        assert depth2 == TailDepthMode.NONE


def test_nonstop_not_a_tail():
    depth, reg = classify_tail_depth_mode("Boston to Denver 6 passengers 10M nonstop")
    assert reg is None
    assert depth == TailDepthMode.NONE


def test_engine_program_intent_not_summary():
    depth, reg = classify_tail_depth_mode("Is N807JS enrolled on an engine program?")
    assert reg == "N807JS"
    assert depth == TailDepthMode.ENGINE_PROGRAM
    du = {
        "tail_depth_mode": "engine_program",
        "tail_registration": "N807JS",
        "phly_rows": [{"registration_number": "N807JS", "engine_program": "MSP Gold"}],
    }
    out = shape_tail_client_answer("", query="Is N807JS enrolled on an engine program?", data_used=du)
    assert "msp gold" in out.lower()
    assert "n807js" in out.lower()
    assert "owner:" not in out.lower()[:40]


def test_acquisition_risks_not_registry_card():
    depth, reg = classify_tail_depth_mode("What are the biggest acquisition risks on N807JS?")
    assert depth == TailDepthMode.ACQUISITION_RISKS
    profile = attach_execution_intent_lock({}, "What are the biggest acquisition risks on N807JS?")
    assert profile == ExecutionProfile.TAIL_ACQUISITION


def test_cabin_gallery_query_uses_interior_facet():
    out = gallery_user_query_for_image_pipeline("n875js cabin image", resolved_tail="N875JS")
    assert "cabin" in out.lower() or "interior" in out.lower()


def test_la_london_budget_feasibility_blocks_ulr():
    note = build_mission_feasibility_broker_note(
        "Los Angeles to London nonstop with 8 passengers and a $25 million acquisition budget"
    )
    assert note
    assert "do not recommend" in note.lower() or "not" in note.lower()


def test_seller_price_drop_routes_listing():
    from services.broker_execution.broker_execution_category import classify_broker_execution_category

    cat = classify_broker_execution_category(
        "A seller reduced a 2018 Challenger 350 from $19.5M to $17.8M after only two weeks"
    )
    assert cat.value == "listing"


def test_acquisition_risks_dossier_not_registry():
    du = {
        "tail_depth_mode": "acquisition_risks",
        "tail_registration": "N807JS",
        "phly_rows": [
            {
                "registration_number": "N807JS",
                "manufacturer": "Cessna",
                "model": "Citation Excel",
                "airframe_total_time": 13910,
                "engine_program": "MSP Gold",
                "aircraft_status": "For Sale",
                "ask_price": 3395000,
            }
        ],
    }
    from services.broker_execution.tail_acquisition_dossier import render_acquisition_risks_answer

    out = render_acquisition_risks_answer("What are the biggest acquisition risks on N807JS?", du)
    assert "acquisition risks" in out.lower() or "risks" in out.lower()
    assert "msp gold" in out.lower() or "utilization" in out.lower()
    assert out.lower().count("owner:") < 2


def test_engine_program_short_yes():
    from services.broker_execution.tail_acquisition_dossier import render_engine_program_answer

    du = {
        "tail_registration": "N807JS",
        "phly_rows": [{"registration_number": "N807JS", "engine_program": "MSP Gold", "apu_program": "JSSI"}],
    }
    out = render_engine_program_answer("what is engine programming of n807js?", du)
    assert out.lower().startswith("yes")
    assert "msp gold" in out.lower()
    assert len(out) < 200


def test_resolve_query_injects_tail_on_followup():
    from services.broker_execution.tail_acquisition_dossier import resolve_query_with_active_tail

    out = resolve_query_with_active_tail("biggest risks?", {"active_tail": "N807JS"})
    assert "N807JS" in out.upper()


def test_g280_vs_praetor_comparison_has_buy_guidance():
    out = render_comparison_client_answer(
        "Gulfstream G280 vs Praetor 600 for a corporate flight department",
        {},
    )
    low = out.lower()
    assert "buy" in low or "wins on" in low


def test_tail_owner_shaped_answer():
    du = {
        "execution_profile": "tail_owner",
        "tail_depth_mode": "owner",
        "tail_registration": "N807JS",
        "tail_facts": [
            {"kind": "ownership", "label": "Owner", "value": "HRL VENTURES LLC"},
            {"kind": "aircraft_model", "label": "Aircraft", "value": "Cessna Citation Excel"},
        ],
    }
    out = shape_tail_client_answer("", query="Who owns N807JS?", data_used=du)
    assert "HRL VENTURES" in out
    assert "**HRL VENTURES LLC**" in out
    assert "N807JS" in out
    assert "Key registration details" in out
    assert out.lower().startswith("the aircraft registered")


def test_tail_sale_yes_no_first():
    du = {
        "execution_profile": "tail_sale_status",
        "tail_depth_mode": "sale_status",
        "tail_registration": "N807JS",
        "phly_rows": [
            {
                "registration_number": "N807JS",
                "aircraft_status": "For Sale",
                "ask_price": 8500000,
                "manufacturer": "Cessna",
                "model": "Citation Excel",
            }
        ],
    }
    out = shape_tail_client_answer("", query="Is N807JS for sale?", data_used=du)
    assert out.strip().lower().startswith("yes")


def test_mission_nyc_tokyo_deterministic_no():
    du: dict = {}
    attach_execution_intent_lock(du, "8 passengers New York to Tokyo nonstop under $30M budget")
    out = build_deterministic_mission_answer(
        "8 passengers New York to Tokyo nonstop under $30M budget",
        du,
    )
    assert out
    low = out.lower()
    assert "tokyo" in low
    assert "30" in out or "does not realistically" in low or "one-stop" in low


def test_tail_owner_governance_no_insufficient():
    du = {
        "execution_profile": "tail_owner",
        "suppress_broker_reasoning_overlay": 1,
        "tail_registration": "N807JS",
        "tail_facts": [
            {"kind": "ownership", "label": "Owner", "value": "HRL VENTURES LLC"},
            {"kind": "aircraft_model", "label": "Aircraft", "value": "Citation Excel"},
        ],
    }
    out = apply_governed_client_answer("Citation Excel owner data", query="Who owns N807JS?", data_used=du)
    assert "INSUFFICIENT_DATA" not in out.upper()
    assert not _FORBIDDEN_SCAFFOLD.search(out)


@pytest.fixture(autouse=True)
def _mission_lightweight_for_layers(monkeypatch):
    """Keep layer certification fast — rank fallback covers mission prose."""
    monkeypatch.setenv("MISSION_BROKER_LIGHTWEIGHT", "1")


@pytest.mark.slow
@pytest.mark.parametrize("probe", PROBES, ids=[p["id"] for p in PROBES])
def test_probe_broker_certify_layers(probe):
    import os

    if os.getenv("BROKER_CERTIFY_USE_LLM", "1").strip().lower() not in ("0", "false", "no"):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY or BROKER_CERTIFY_USE_LLM=0 required")

    answer, du, path = broker_certify(probe["query"], prefer_e2e=False)
    assert path == "layers"
    assert answer.strip()
    assert not _FORBIDDEN_SCAFFOLD.search(answer)

    for header in FORBIDDEN_HEADERS:
        assert header not in answer, f"forbidden header {header!r}"

    forbid = probe.get("forbid")
    if forbid is not None:
        assert not forbid.search(answer), f"forbidden pattern in {probe['id']}"

    assert "INSUFFICIENT_DATA" not in answer.upper()

    low = answer.lower()
    if probe.get("expect_lead_yes_no"):
        assert low.startswith("yes") or low.startswith("no")

    for token in probe.get("expect_in_answer") or ():
        assert token.lower() in low, f"missing {token!r} in {probe['id']}"

    if probe.get("expect_budget_no"):
        assert any(x in low for x in ("does not realistically", "one-stop", "not realistically", "no "))

    if probe.get("expect_mission_models"):
        assert any(
            x in low
            for x in ("start with", "i'd", "passengers", "boston", "denver", "kbos", "kden", "confirm")
        )

    if probe["id"].startswith("comparison"):
        assert "wins on" in low or "tradeoff" in low or "buy" in low

    assert du.get("deterministic_answer_renderer_applied") or du.get("execution_profile")
