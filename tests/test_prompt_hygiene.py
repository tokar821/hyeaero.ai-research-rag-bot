"""Prompt hygiene tests."""

from services.consultant.prompt_hygiene import apply_prompt_hygiene, detect_repeated_phrases, score_prompt_contamination


def test_detects_repeated_assuming_phrase():
    text = "Intro.\n\nAssuming 6-8 passengers fit A.\n\nAssuming 6-8 passengers fit B."
    assert "assuming 6-8 passengers" in detect_repeated_phrases(text)


def test_strips_retrieval_bleed():
    raw = "[AUTHORITATIVE — phlydata_aircraft] The aircraft is a Challenger 350."
    cleaned, report = apply_prompt_hygiene(raw)
    assert "phlydata" not in cleaned.lower()
    assert report.bleed_detected


def test_cross_turn_assuming_suppressed():
    prior = "Assuming 6-8 passengers, start with Challenger 350."
    current = "More detail please.\n\nAssuming 6-8 passengers, here are fits again."
    cleaned, report = apply_prompt_hygiene(current, prior_answer=prior)
    assert cleaned.lower().count("assuming 6") == 0
    assert "stripped_stock_advisory_fallback" in report.actions_taken


def test_contamination_score_increases_with_fallback():
    clean = "The Falcon 8X range is about 6,450 nm with margin for your leg."
    dirty = clean + "\n\nAssuming 6-8 passengers Consultant Insight: dispatch"
    assert score_prompt_contamination(dirty) > score_prompt_contamination(clean)
