from rag.response_safety import answer_contains_banned_terms, sanitize_user_facing_answer


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

