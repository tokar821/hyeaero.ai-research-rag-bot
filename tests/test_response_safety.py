from rag.response_safety import (
    answer_contains_banned_terms,
    enforce_consultant_quality,
    sanitize_user_facing_answer,
)


def test_sanitize_cant_show_images_strong_gallery_replacement():
    raw = "I can't show images, but the cabin is spacious."
    out = sanitize_user_facing_answer(raw, strong_aircraft_gallery=True)
    assert "no verified images" not in out.lower()
    assert "gallery" in out.lower()


def test_sanitize_removes_internal_terms_and_tags():
    raw = """
[AUTHORITATIVE — PhlyData (Hye Aero aircraft source): phlydata_aircraft + FAA MASTER (faa_master)]
Per PhlyData, the aircraft is for sale. Pinecone vector search found similar records.
FAA MASTER registrant (faa_master): Example LLC
Controller scrape shows an asking price.
"""
    out = sanitize_user_facing_answer(raw)
    banned = answer_contains_banned_terms(out)
    assert not banned, f"still contains banned terms: {banned} :: {out}"
    assert "aircraft registry" in out.lower() or "registration records" in out.lower()


def test_advisory_mode_does_not_append_stock_assuming_block():
    draft = "The Gulfstream G650 is a strong ultra-long-range option for your mission."
    out = enforce_consultant_quality(
        draft,
        query="What jet should I buy for long trips?",
        data_used={
            "consultant_response_mode": "advisory",
            "consultant_intelligence_layer": 1,
            "consultant_structured_formatter": 1,
            "consultant_recommendations": [{"model": "Gulfstream G650"}],
        },
    )
    assert "Assuming 6" not in out
    assert "here are a few realistic fits" not in out.lower()
    assert "Consultant Insight:" not in out


def test_strips_existing_assuming_block_from_llm_draft():
    dirty = (
        "Lead answer about the Falcon 8X.\n\n"
        "Assuming 6–8 passengers and typical business-use constraints (no extreme hot/high), "
        "here are a few realistic fits:\n"
        "- Challenger 350: balanced\n\n"
        "Consultant Insight: dispatch."
    )
    out = enforce_consultant_quality(
        dirty,
        query="Tell me about Falcon 8X",
        data_used={"consultant_response_mode": "advisory"},
    )
    assert "Assuming 6" not in out
    assert "here are a few realistic fits" not in out.lower()
    assert "Lead answer" in out


def test_high_budget_cabin_gate_replaces_light_jet_spam():
    bad = "Try Citation CJ2 and Learjet 45 in your range."
    q = "I want the cabin to feel expensive but not tacky — what should I look at around $40M?"
    out = enforce_consultant_quality(bad, query=q, data_used={})
    lo = out.lower()
    assert "gulfstream" in lo and "falcon" in lo and "global" in lo and "challenger" in lo
    assert "cj2" not in lo

