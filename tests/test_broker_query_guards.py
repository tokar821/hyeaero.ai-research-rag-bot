"""Broker query guards — segment liquidity, pre-offer, skepticism, comparison recovery."""

from __future__ import annotations

from services.broker_execution.broker_query_guards import (
    is_portfolio_mission_buy_query,
    is_resale_maximization_query,
    is_segment_liquidity_query,
    is_single_aircraft_mission_query,
    render_segment_liquidity_answer,
    try_broker_query_guard,
)
from services.broker_execution.output_governance import apply_governed_client_answer
from services.mission.route_extractor import extract_itinerary_chain


def test_segment_liquidity_guard():
    q = "Which has stronger liquidity today: G280, Challenger 350, Citation Longitude, or Falcon 2000LXS?"
    assert is_segment_liquidity_query(q)
    out = try_broker_query_guard(q, "", {})
    assert out and "Challenger 350" in out


def test_liquidity_via_governance_not_mission_fault():
    q = "Which has stronger liquidity today: G280, Challenger 350?"
    out = apply_governed_client_answer("", query=q, data_used={})
    assert "Challenger 350" in out
    assert "restate the city pair" not in out.lower()


def test_pre_offer_verification_guard():
    q = "A 2020 Citation Longitude is listed at $24.5M. What's the first thing you would verify before making an offer?"
    out = try_broker_query_guard(q, "", {})
    assert out and "logbook" in out.lower()


def test_cosmetic_skepticism_guard():
    q = "The aircraft has fresh paint, fresh interior, and a recent price reduction. Does that make you more interested or less?"
    out = try_broker_query_guard(q, "more attractive", {})
    assert out and "cautious" in out.lower()


def test_comparison_incomplete_recovery():
    q = "Falcon 2000LXS vs Praetor 600. Ignore purchase price. Which aircraft would you rather operate and why?"
    body = (
        "When comparing the Falcon 2000LXS and the Praetor 600, the choice largely depends on your "
        "operational priorities.\n\nHere's a structured comparison:"
    )
    du: dict = {}
    out = try_broker_query_guard(q, body, du)
    assert out and "operate" in out.lower()
    ui = du.get("comparison_broker_ui") or {}
    assert ui.get("model_a") == "Falcon 2000LXS"
    assert len(ui.get("specs") or []) >= 6


def test_resale_guard():
    q = "You're given $18M today and must maximize resale value over the next five years. What aircraft do you buy?"
    assert is_resale_maximization_query(q)
    out = apply_governed_client_answer("Global 5000 list dump", query=q, data_used={})
    assert "Challenger 350" in out


def test_owner_buy_concern_guard():
    q = "Who owns N7509 and what would concern you most if I were buying it today?"
    registry = "- Aircraft: BD-700-2A12\n- Owner: BANK OF UTAH TRUSTEE\n- Year: 2022\n- Registration: N7509"
    du = {
        "phly_rows": [{"marketing_type": "Global 7500", "registered_owner": "BANK OF UTAH TRUSTEE", "year": "2022"}],
        "tail_registration": "N7509",
    }
    out = try_broker_query_guard(q, registry, du)
    assert out and "concern" in out.lower()
    assert "Trust registrant" in out
    assert "BANK OF UTAH" in out or "Registrant" in out


def test_owner_buy_parses_registry_body():
    q = "Who owns N7509 and what would concern you most if I were buying it today?"
    registry = "- Aircraft: BD-700-2A12\n- Owner: BANK OF UTAH TRUSTEE\n- Year: 2022\n- Registration: N7509"
    out = try_broker_query_guard(q, registry, {})
    assert "BD-700" in out
    assert "BANK OF UTAH" in out
    assert "2022" in out


def test_route_map_limiting_leg_denver_reykjavik():
    from services.broker_execution.broker_query_guards import render_route_map_broker_answer
    from services.consultant.route_feasibility import estimate_route_distance_nm

    q = "Show me a route map for a Challenger 350 flying: Boston -> Denver -> Reykjavik -> London"
    assert estimate_route_distance_nm("Denver -> Reykjavik") > 2700
    prose, _ = render_route_map_broker_answer(q, {})
    assert "Denver" in prose and "Reykjavik" in prose
    assert "BINDS" in prose or "2920" in prose
    assert "Range envelope (generated)" not in prose
    assert "Mission reachability" not in prose
    assert "Payload-range" not in prose


def test_gallery_contradiction_guard():
    q = "Show me every verified image you can find for N875JS"
    body = "I currently don't have verified images for N875JS, a Challenger 350"
    du = {
        "aircraft_images": [{"url": "http://x.jpg", "caption": "Listing cabin (exact tail) - N875JS"}],
        "phly_rows": [{"marketing_type": "Citation XLS"}],
        "tail_registration": "N875JS",
    }
    out = try_broker_query_guard(q, body, du)
    assert out and ("below" in out.lower() or "photo" in out.lower())
    assert "Citation XLS" in out


def test_route_map_stream_shortcircuit():
    from services.broker_execution.broker_query_guards import resolve_broker_guard_stream_payload

    q = "Show me a route map for a Challenger 350 flying: Boston -> Denver -> Reykjavik -> London"
    payload = resolve_broker_guard_stream_payload(q, {"answer": "", "data_used": {}})
    assert payload and len(payload.get("answer") or "") > 200
    assert payload["data_used"].get("broker_query_guard_applied") == 1


def test_extract_itinerary_chain_four_stops():
    legs = extract_itinerary_chain(
        "Show me a route map for a Challenger 350 flying: Boston -> Denver -> Reykjavik -> London"
    )
    assert len(legs) == 3
    assert "Boston" in legs[0]
    assert "London" in legs[-1]


def test_multiline_itinerary_chicago_dubai():
    legs = extract_itinerary_chain(
        "Create a route map showing:\nChicago ->\nReykjavik ->\nLondon ->\nDubai"
    )
    assert len(legs) == 3
    assert "Chicago" in legs[0]
    assert "Dubai" in legs[-1]


def test_multiline_city_list_without_arrows():
    legs = extract_itinerary_chain(
        "Create a route map showing:\nChicago\nReykjavik\nLondon\nDubai\nusing a G280 mission profile"
    )
    assert len(legs) == 3
    assert "Chicago" in legs[0]
    assert "Dubai" in legs[-1]


def test_gallery_guard_empty_images():
    q = "Show me every verified image you can find for N604WM"
    body = "No verified images are currently available for N604WM"
    out = try_broker_query_guard(q, body, {"tail_registration": "N604WM", "aircraft_images": []})
    assert out and ("jetphotos" in out.lower() or "listing" in out.lower() or "caption" in out.lower())
    assert "flightaware" not in out.lower()


def test_gallery_stream_shortcircuit_without_images():
    from services.broker_execution.broker_query_guards import resolve_broker_guard_stream_payload

    q = "show me n750LX"
    payload = resolve_broker_guard_stream_payload(q, {"answer": "", "data_used": {}})
    assert payload and len(payload.get("answer") or "") > 40
    assert payload["data_used"].get("broker_guard_stream_shortcircuit") == 1


def test_scottsdale_london_not_dallas_portfolio():
    q = "7 passengers. Scottsdale to London. One aircraft only. Budget $14M. What would you buy?"
    assert not is_portfolio_mission_buy_query(q)
    assert is_single_aircraft_mission_query(q)
    out = try_broker_query_guard(q, "", {})
    assert out and "Praetor" in out
    assert "Dallas" not in out


def test_liquidity_ranks_named_models():
    q = "Which aircraft has stronger resale liquidity: Falcon 2000LXS, Latitude, Praetor 600, or G280? Rank all four."
    out = render_segment_liquidity_answer(q)
    assert "Praetor" in out
    assert "Challenger 350" not in out or "Rank for your list" in out


def test_nyc_lisbon_nonstop_hard_no():
    q = "Can a Challenger 350 realistically fly 8 passengers from New York to Lisbon nonstop year-round?"
    out = try_broker_query_guard(q, "maybe with fuel", {})
    assert out and "not reliably" in out.lower()


def test_comparison_rather_operate():
    q = "If range didn't matter, would you rather operate a Falcon 2000LXS or a Challenger 3500? Why?"
    out = try_broker_query_guard(q, "larger windows and natural light", {})
    assert out and "Challenger 3500" in out
    assert "operate" in out.lower()


def test_dom_psychology_guard():
    q = "A 2020 Longitude has been on the market for 14 months. What assumptions are you making before even seeing the logs?"
    out = try_broker_query_guard(q, "generic list", {})
    assert out and "price" in out.lower()


def test_pre_offer_first_three():
    q = "If a 2018 Falcon 2000LXS is listed at $18.5M today, what would be the first three things you'd verify before making an offer?"
    out = try_broker_query_guard(q, "", {})
    assert out and out.lower().count("program") >= 1
    assert out.count("\n") < 12


def test_pre_offer_tail_prefix_clean_grammar():
    q = "N807JS - what would you verify before making an offer?"
    out = try_broker_query_guard(q, "", {})
    assert out
    assert "this the aircraft" not in out.lower()
    assert "N807JS" in out
    assert out.startswith("Before an LOI on N807JS")


def test_comparison_challenger_300_not_350():
    q = "Compare Challenger 300 vs Citation Latitude"
    du = {"comparison_v2": {"models": ["Challenger 350", "Citation Latitude"]}}
    out = try_broker_query_guard(q, "generic incomplete comparison", du)
    assert out
    ui = du.get("comparison_broker_ui") or {}
    assert ui.get("model_a") == "Challenger 300"
    assert ui.get("model_b") == "Citation Latitude"
    assert "Challenger 350" not in str(ui)
    specs = ui.get("specs") or []
    assert len(specs) >= 6
    dims = {row.get("dimension") for row in specs if isinstance(row, dict)}
    assert "Max range" in dims
    assert "Cabin width" in dims


def test_comparison_falcon_lxs_google_style_specs():
    from services.broker_execution.comparison_broker_facts import build_comparison_broker_ui_payload

    q = "Falcon 2000LXS vs Praetor 600"
    du: dict = {}
    ui = build_comparison_broker_ui_payload(q, du)
    assert ui
    specs = {row["dimension"]: row for row in ui.get("specs") or []}
    assert specs["Max range"]["a"] == "~3,344 nm"
    assert specs["Max range"]["b"] == "~3,256 nm"
    assert "564" in specs["Max speed"]["a"]
    assert "466" in specs["Max speed"]["b"]
    assert "7 ft 8 in" in specs["Cabin width"]["a"]
    assert "Takeoff distance" in specs


def test_comparison_catalog_broker_commentary_global_7500():
    from services.broker_execution.comparison_broker_facts import (
        build_comparison_broker_ui_payload,
        render_comparison_display_answer,
    )

    q = "Compare Global 7500 vs Gulfstream G650"
    du: dict = {}
    ui = build_comparison_broker_ui_payload(q, du)
    assert ui
    assert ui.get("verification_status") == "verified"
    assert ui.get("a_wins") or ui.get("b_wins")
    assert ui.get("broker_summary")
    # Lead-in lines are not repeated in structured commentary.
    assert not ui.get("commentary")
    assert ui.get("tradeoffs")
    assert "compare verified" not in str(ui).lower()
    out = render_comparison_display_answer(q, du)
    assert out and "vs Gulfstream G650" not in out
    assert "range" in out.lower() or "nm" in out.lower()


def test_comparison_light_jet_category_from_curated():
    from services.broker_execution.comparison_broker_facts import build_comparison_broker_ui_payload

    ui = build_comparison_broker_ui_payload("Compare Phenom 300E vs Citation CJ4", {})
    cats = [r for r in ui.get("specs") or [] if r.get("dimension") == "Category"]
    assert cats and cats[0]["a"] == "light" and cats[0]["b"] == "light"


def test_comparison_partial_unknown_no_placeholders():
    from services.broker_execution.comparison_broker_facts import build_comparison_broker_ui_payload

    ui = build_comparison_broker_ui_payload("Compare HondaJet Elite II vs Phenom 300E", {})
    assert ui
    assert ui.get("verification_status") == "partial"
    assert "HondaJet Elite II" in str(ui.get("broker_notice") or "")
    assert ui.get("specs") == []
    assert "compare verified" not in str(ui).lower()


def test_tail_buy_concern_n750lx():
    q = "What concerns would you have if I were buying N750LX today?"
    out = try_broker_query_guard(q, "generic chatgpt", {})
    assert out and "concern" in out.lower()


def test_image_tier_rejects_placeholder():
    from services.broker_execution.image_verification_tiers import classify_image_trust_tier

    tier = classify_image_trust_tier(
        {"url": "https://example.com/tailimages/G650.png", "gallery_label": "Listing cabin"},
        tail="N628TS",
        user_query="Show me the cabin of N628TS",
    )
    assert tier == "rejected"
    tier2 = classify_image_trust_tier(
        {
            "url": "https://elasticbeanstalk-us-east-1.amazonaws.com/tail_images_fixed/N5616.png",
            "gallery_label": "Listing cabin (exact tail)",
        },
        tail="N5616",
        user_query="N5616 cabin image",
    )
    assert tier2 == "rejected"


def test_tail_cabin_without_show_me_triggers_gallery_intent():
    from rag.consultant_market_lookup import wants_consultant_aircraft_images_in_answer

    assert wants_consultant_aircraft_images_in_answer("N51ND cabin") is True
    assert wants_consultant_aircraft_images_in_answer("N5616 cabin image") is True


def test_gallery_contradicts_image_refusal():
    from services.broker_execution.broker_query_guards import gallery_contradicts_answer

    du = {"aircraft_images": [{"url": "https://example.com/a.jpg"}]}
    assert gallery_contradicts_answer("I can't display images directly.", du) is True


def test_deictic_show_me_followup_uses_history_tail():
    from rag.consultant_market_lookup import wants_consultant_aircraft_images_in_answer

    hist = [
        {"role": "user", "content": "N51ND cabin"},
        {"role": "assistant", "content": "Bombardier Global 560 cabin overview."},
    ]
    assert wants_consultant_aircraft_images_in_answer("can you show me ?", hist) is True


def test_engine_program_guard_short_circuits_llm_dump():
    q = "Is N628TS enrolled on an engine program?"
    llm_dump = (
        "The Gulfstream G650ER is not enrolled. Mailing Address: 1 Rocket Rd, Hawthorne, CA. "
        "Before considering any purchase, it's crucial to verify enrollment."
    )
    du = {
        "tail_registration": "N628TS",
        "phly_rows": [
            {
                "registration_number": "N628TS",
                "engine_program": "MSP Gold",
                "apu_program": "JSSI",
            }
        ],
    }
    out = try_broker_query_guard(q, llm_dump, du)
    assert out
    assert "MSP Gold" in out
    assert "Mailing Address" not in out
    assert "crucial to verify" not in out.lower()


def test_gallery_replaces_listing_dump_on_interior():
    q = "show me the interior on N807JS"
    listing_dump = (
        "N807JS is a 2003 Cessna Citation Excel currently listed for sale.\n"
        "- Ask Price: $3,395,000\n- Airframe Total Time: 13,910 hours\n"
        "- Engine Program: MSP Gold\nWould you like to see more details about the interior?"
    )
    du = {
        "tail_registration": "N807JS",
        "aircraft_images": [{"url": "https://example.com/cabin.jpg", "caption": "Cabin - N807JS"}],
        "phly_rows": [{"marketing_type": "Citation Excel"}],
    }
    out = try_broker_query_guard(q, listing_dump, du)
    assert out
    assert "below" in out.lower()
    assert "$3,395,000" not in out
    assert "13,910" not in out


def test_cockpit_followup_triggers_gallery_intent():
    from rag.consultant_market_lookup import wants_consultant_aircraft_images_in_answer

    assert wants_consultant_aircraft_images_in_answer("what about the cockpit?") is True


def test_cockpit_followup_replaces_listing_prose():
    q = "what about the cockpit?"
    listing_dump = (
        "The cockpit of N807JS is equipped with modern avionics. Ask Price: $3,395,000. "
        "Airframe Total Time: 13,910 hours. recommend reaching out directly to the seller."
    )
    du = {
        "tail_registration": "N807JS",
        "aircraft_images": [{"url": "https://example.com/cockpit.jpg", "caption": "Cockpit - N807JS"}],
    }
    hist = [
        {"role": "user", "content": "show me the interior on N807JS"},
        {"role": "assistant", "content": "N807JS cabin photo below."},
    ]
    out = try_broker_query_guard(q, listing_dump, du, hist)
    assert out
    assert "cockpit" in out.lower()
    assert "reaching out" not in out.lower()


def test_nyc_tokyo_mission_guard_rejects_ulr_shortlist():
    q = "8 passengers New York to Tokyo nonstop under $30M"
    llm_dump = (
        "The great-circle distance is 6,740 nm. Consider Bombardier Global 7500 and Gulfstream G650ER. "
        "VP-CAH Global 7500 may be available."
    )
    out = try_broker_query_guard(q, llm_dump, {})
    assert out
    assert "does not realistically" in out.lower()
    assert "VP-CAH" not in out


def test_nyc_tokyo_governance_override():
    q = "8 passengers New York to Tokyo nonstop under $30M"
    llm_dump = (
        "Consider Bombardier Global 7500 and Gulfstream G650ER for this mission."
    )
    out = apply_governed_client_answer(llm_dump, query=q, data_used={"llm_executed": 1})
    assert "does not realistically" in out.lower()
    assert "Global 7500" not in out or "does not realistically" in out.lower()
