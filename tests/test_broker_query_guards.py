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
    out = try_broker_query_guard(q, body, {})
    assert out and len(out) > 120
    assert "wins on" in out.lower() or "buy" in out.lower()


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
