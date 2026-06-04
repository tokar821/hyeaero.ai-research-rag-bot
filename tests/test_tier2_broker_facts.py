"""Tier 2 broker facts — comparison depth, tail profile, tail-first gallery."""

from __future__ import annotations

from rag.consultant_market_lookup import build_aircraft_photo_focus_tavily_query
from rag.consultant_query_anchor import gallery_user_query_for_image_pipeline
from services.broker_execution.comparison_broker_facts import build_comparison_broker_facts_block
from services.broker_execution.fact_pack_builder import build_fact_pack, render_fact_pack_for_llm_context
from services.broker_execution.intent_answer_contract import build_intent_answer_contract_suffix
from services.broker_execution.tail_aircraft_profile import build_tail_aircraft_profile_block


def test_comparison_broker_facts_g280_vs_longitude():
    q = "Gulfstream G280 vs Citation Longitude for transcon"
    block = build_comparison_broker_facts_block(
        q,
        {"comparison_v2": {"status": "OK", "models": ["Gulfstream G280", "Citation Longitude"]}},
    )
    assert "G280" in block or "g280" in block.lower()
    assert "wins on" in block.lower()
    assert "Buy" in block
    assert "Tradeoff" in block


def test_fact_pack_includes_comparison_broker_block():
    du = {
        "comparison_v2": {"status": "OK", "models": ["Praetor 600", "Citation Longitude"]},
    }
    pack = build_fact_pack("Praetor 600 vs Citation Longitude", du)
    kinds = [f.get("kind") for f in pack.get("facts") or []]
    assert "comparison_broker" in kinds
    block = render_fact_pack_for_llm_context(pack)
    assert "COMPARISON BROKER FACTS" in block
    assert "buy" in block.lower()


def test_tail_profile_detail_query():
    du = {
        "tail_depth_mode": "detail",
        "tail_registration": "N807JS",
        "tail_facts": [
            {"kind": "registration", "label": "Registration", "value": "N807JS"},
            {"kind": "aircraft_model", "label": "Aircraft", "value": "Citation Excel"},
            {"kind": "ownership", "label": "Owner", "value": "Example Holdings LLC"},
        ],
        "phly_rows": [
            {
                "registration_number": "N807JS",
                "aircraft_status": "For Sale",
                "ask_price": 8500000,
                "owner": "Example Holdings LLC",
                "manufacturer": "Cessna",
                "model": "Citation Excel",
            }
        ],
    }
    block = build_tail_aircraft_profile_block("Tell me everything about N807JS", du)
    assert "N807JS" in block
    assert "For Sale" in block
    assert "Example Holdings" in block


def test_fact_pack_includes_tail_profile_for_detail():
    du = {
        "tail_depth_mode": "detail",
        "tail_registration": "N807JS",
        "tail_facts": [{"kind": "registration", "label": "Registration", "value": "N807JS"}],
        "phly_rows": [{"registration_number": "N807JS", "aircraft_status": "Available"}],
    }
    pack = build_fact_pack("Tell me everything about N807JS", du)
    assert any(f.get("kind") == "tail_profile" for f in pack.get("facts") or [])


def test_intent_contract_comparison_buy_if():
    suffix = build_intent_answer_contract_suffix(
        "G280 vs Longitude",
        data_used={"comparison_v2": {"status": "OK"}},
    )
    assert "buy X if" in suffix.lower() or "buy" in suffix.lower()
    assert "tradeoff" in suffix.lower()


def test_gallery_user_query_tail_led():
    out = gallery_user_query_for_image_pipeline("show me n140kw", resolved_tail="N140KW")
    assert out.upper().startswith("N140KW")
    assert "photos" in out.lower() or "aircraft" in out.lower()


def test_photo_focus_tail_before_phly_model():
    phly = [
        {
            "registration_number": "N807JS",
            "manufacturer": "Cessna",
            "model": "Citation Excel",
        }
    ]
    q = build_aircraft_photo_focus_tavily_query("show me N807JS", phly, None)
    assert q
    assert '"N807JS"' in q.upper() or "N807JS" in q.upper()
    assert "Citation Excel" not in q
