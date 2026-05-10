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


def test_high_budget_cabin_gate_replaces_light_jet_spam():
    bad = "Try Citation CJ2 and Learjet 45 in your range."
    q = "I want the cabin to feel expensive but not tacky — what should I look at around $40M?"
    out = enforce_consultant_quality(bad, query=q, data_used={})
    lo = out.lower()
    assert "gulfstream" in lo and "falcon" in lo and "global" in lo and "challenger" in lo
    assert "cj2" not in lo

